import os
import uuid
import shutil
import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from dotenv import load_dotenv
from supabase import create_client, Client as SupabaseClient
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_KEY")

if not HF_TOKEN:
    logger.error("HF_TOKEN not found in environment variables")
    raise ValueError("HF_TOKEN is required")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("SUPABASE_URL or SUPABASE_SERVICE_KEY not found in environment variables")
    raise ValueError("Supabase credentials are required")

app = FastAPI(title="AI Video Generator", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
client = None
audio_client = None
supabase: SupabaseClient = None

# Standard video parameters to ensure consistency
STANDARD_WIDTH = 544
STANDARD_HEIGHT = 960
STANDARD_FPS = 24  # Use 24fps as standard (works well for most content)

@app.on_event("startup")
async def startup_event():
    global client, audio_client, supabase
    try:
        logger.info("Initializing Gradio client...")
        client = Client("Heartsync/wan2_2-I2V-14B-FAST", token=HF_TOKEN)
        logger.info("Gradio client initialized successfully")

        logger.info("Initializing Audio Gradio client...")
        audio_client = Client("chenxie95/MeanAudio", token=HF_TOKEN)
        logger.info("Audio Gradio client initialized successfully")
        
        logger.info("Initializing Supabase client...")
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "client_ready": client is not None,
        "audio_client_ready": audio_client is not None,
        "supabase_ready": supabase is not None
    }

def parse_prompt(prompt: str):
    """Parse prompt to extract magic prompt and caption"""
    if "!@#" not in prompt:
        return prompt.strip(), None
    
    parts = prompt.split("!@#", 1)
    magic_prompt = parts[0].strip()
    caption = parts[1].strip() if len(parts) > 1 else None
    
    # Replace ^ with empty string
    magic_prompt = magic_prompt if magic_prompt != "^" else ""
    caption = caption if caption != "^" else None
    
    return magic_prompt, caption

@app.post("/generate/")
async def generate_video(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    sender_uid: str = Form(...),
    receiver_uids: str = Form(...)
):
    """Generate video with middle frame + AI video + mixed audio"""
    temp_files = []  # Track all temp files for cleanup

    try:
        # Parse the prompt to extract magic prompt and caption
        magic_prompt, caption = parse_prompt(prompt)
        
        logger.info(f"Parsed prompt - Magic: '{magic_prompt}', Caption: '{caption}'")
        
        # Determine if we should skip API processing
        skip_api = (magic_prompt == "" or magic_prompt is None)
        
        # Improved video validation
        content_type = file.content_type or ""
        filename = file.filename or ""

        # Check content type OR file extension for video
        valid_content_types = ['video/mp4', 'video/mov', 'video/avi', 'video/webm', 'video/quicktime']
        valid_extensions = ['.mp4', '.mov', '.avi', '.webm', '.qt']

        is_valid_content_type = any(content_type.startswith(ct) for ct in valid_content_types)
        is_valid_extension = any(filename.lower().endswith(ext) for ext in valid_extensions)

        if not (is_valid_content_type or is_valid_extension):
            logger.warning(f"Invalid file - Content-Type: {content_type}, Filename: {filename}")
            raise HTTPException(status_code=400, detail="File must be a video (mp4, mov, avi, webm)")

        logger.info(f"Starting processing for user {sender_uid}")
        logger.info(f"Original Prompt: {prompt}")
        logger.info(f"Skip API: {skip_api}")
        logger.info(f"Receivers: {receiver_uids}")

        # Create temp directory if it doesn't exist
        temp_dir = Path("/tmp")
        temp_dir.mkdir(exist_ok=True)

        # Save uploaded video temporarily
        video_id = str(uuid.uuid4())
        file_extension = Path(filename).suffix or '.mp4'
        temp_video_path = temp_dir / f"{video_id}{file_extension}"
        temp_files.append(temp_video_path)

        # Save file
        with open(temp_video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Video saved to {temp_video_path}")

        # Validate file size (optional)
        file_size = temp_video_path.stat().st_size
        if file_size > 50 * 1024 * 1024:  # 50MB limit for videos
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")

        # Check if Supabase is available
        if supabase is None:
            raise HTTPException(status_code=503, detail="Storage service not available")

        video_url = None
        
        if skip_api:
            # Skip API processing, upload video directly to Supabase
            logger.info("Skipping API processing, uploading video directly to Supabase")
            video_url = await _upload_video_to_supabase(str(temp_video_path), sender_uid)
            logger.info(f"Video uploaded to Supabase: {video_url}")
        else:
            # Process with API
            # Check if clients are available
            if client is None:
                raise HTTPException(status_code=503, detail="AI video service not available")
            
            if audio_client is None:
                raise HTTPException(status_code=503, detail="AI audio service not available")

            # Process video with new structure: first_part + middle_frame + ai_video + last_part
            logger.info("Processing video: extracting parts with middle frame...")
            middle_frame_path, first_part_path, middle_frame_video_path, ai_video_duration, remaining_audio_path = await _process_video_with_middle_frame(str(temp_video_path))
            temp_files.extend([middle_frame_path, first_part_path, middle_frame_video_path, remaining_audio_path])

            # Start both AI video and AI audio generation concurrently
            logger.info("Starting AI video and audio generation concurrently...")
            
            # Create tasks for both generations
            ai_video_task = asyncio.create_task(
                asyncio.wait_for(
                    asyncio.to_thread(_predict_video, middle_frame_path, magic_prompt),
                    timeout=300.0  # 5 minutes timeout
                )
            )

            ai_audio_task = asyncio.create_task(
                asyncio.wait_for(
                    asyncio.to_thread(_predict_audio, magic_prompt),
                    timeout=300.0  # 5 minutes timeout
                )
            )

            # Wait for both tasks to complete
            ai_video_result, ai_audio_result = await asyncio.gather(ai_video_task, ai_audio_task)

            if not ai_video_result or len(ai_video_result) < 2:
                raise HTTPException(status_code=500, detail="Invalid response from video AI model")

            # Handle audio failure gracefully
            if not ai_audio_result:
                logger.warning("AI audio generation failed, will use only original audio")

            ai_video_path = ai_video_result[0].get("video") if isinstance(ai_video_result[0], dict) else ai_video_result[0]
            seed_used = ai_video_result[1] if len(ai_video_result) > 1 else "unknown"

            logger.info(f"AI Video generated locally: {ai_video_path}")
            logger.info(f"AI Audio result: {ai_audio_result}")

            # Standardize AI video to match our requirements
            ai_video_standardized = await _standardize_video(ai_video_path, is_ai_video=True)
            temp_files.append(ai_video_standardized)

            # Create mixed audio for AI video section (AI audio + original remaining audio)
            mixed_audio_path = await _create_mixed_audio(ai_audio_result, remaining_audio_path, ai_video_duration)
            temp_files.append(mixed_audio_path)

            # Merge AI video with mixed audio
            ai_merged_path = await _merge_video_audio_standardized(ai_video_standardized, mixed_audio_path)
            temp_files.append(ai_merged_path)
            logger.info(f"AI video merged with mixed audio: {ai_merged_path}")

            # Create final video by concatenating all four parts: first_part + middle_frame + ai_video + (remaining original video handled in first_part)
            final_video_path = await _concatenate_videos_standardized([first_part_path, middle_frame_video_path, ai_merged_path])
            temp_files.append(final_video_path)
            logger.info(f"Final video created: {final_video_path}")

            # Upload final video to Supabase storage
            video_url = await _upload_video_to_supabase(final_video_path, sender_uid)
            logger.info(f"Final video uploaded to Supabase: {video_url}")

        # Save chat messages to Firebase for each receiver
        receiver_list = [uid.strip() for uid in receiver_uids.split(",") if uid.strip()]
        await _save_chat_messages_to_firebase(sender_uid, receiver_list, video_url, magic_prompt or "", caption, skip_api)

        return JSONResponse({
            "success": True,
            "video_url": video_url,
            "sender_uid": sender_uid,
            "receiver_uids": receiver_list,
            "caption": caption,
            "skipped_api": skip_api
        })

    except asyncio.TimeoutError:
        logger.error("AI Video/Audio generation timed out after 5 minutes")
        raise HTTPException(
            status_code=408, 
            detail="AI Video/Audio generation timed out. Please try with a simpler prompt or smaller video."
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        logger.error(f"Error generating video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate video: {str(e)}"
        )

    finally:
        # Cleanup temporary files
        for temp_path in temp_files:
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                    logger.info(f"Cleaned up temp file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")

async def _process_video_with_middle_frame(video_path: str) -> tuple:
    """Extract parts: first_part + middle_frame + remaining_audio"""
    try:
        import subprocess
        
        temp_dir = Path("/tmp")
        process_id = str(uuid.uuid4())
        
        # Get video properties first
        cmd_info = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
            '-show_format', video_path
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd_info, capture_output=True, text=True, timeout=30
        )
        
        if result.returncode != 0:
            raise Exception(f"Failed to get video info: {result.stderr}")
        
        import json
        video_info = json.loads(result.stdout)
        
        # Find video stream
        video_stream = next((stream for stream in video_info['streams'] if stream['codec_type'] == 'video'), None)
        if not video_stream:
            raise Exception("No video stream found")
        
        # Get video properties
        duration = float(video_info['format']['duration'])
        original_fps = eval(video_stream['r_frame_rate'])  # This gives exact fps as fraction
        
        logger.info(f"Original video - Duration: {duration}s, FPS: {original_fps}")
        
        # Calculate split points
        middle_time = duration / 2
        ai_duration = 3.5  # AI video is 3.5 seconds (changed from 5.0)
        
        logger.info(f"Split points - Middle: {middle_time}s, AI duration: {ai_duration}s")
        
        # Extract high-quality middle frame (PNG format for lossless quality)
        middle_frame_path = temp_dir / f"{process_id}_middle_frame.png"
        cmd_frame = [
            'ffmpeg', '-y', '-i', video_path,
            '-ss', str(middle_time), '-vframes', '1',
            '-vf', 'scale=-1:1080:flags=lanczos,unsharp=5:5:1.0:5:5:0.0',  # Upscale + sharpen
            '-f', 'image2',
            '-pix_fmt', 'rgb24',
            str(middle_frame_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd_frame, capture_output=True, text=True, timeout=60
        )
        
        if result.returncode != 0:
            raise Exception(f"Failed to extract middle frame: {result.stderr}")
        
        # Extract first part (start to middle time) with original audio
        first_part_path = temp_dir / f"{process_id}_first_part.mp4"
        cmd_first = [
            'ffmpeg', '-y', '-i', video_path,
            '-t', str(middle_time),
            '-vf', f'scale={STANDARD_WIDTH}:{STANDARD_HEIGHT}:force_original_aspect_ratio=decrease,pad={STANDARD_WIDTH}:{STANDARD_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black',
            '-r', str(STANDARD_FPS),  # Standardize FPS
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-ac', '2',  # Standardize audio
            '-movflags', '+faststart',
            '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
            str(first_part_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd_first, capture_output=True, text=True, timeout=120
        )
        
        if result.returncode != 0:
            raise Exception(f"Failed to extract first part: {result.stderr}")
        
        # Create middle frame as a very short video (0.1 seconds) to maintain continuity
        middle_frame_video_path = temp_dir / f"{process_id}_middle_frame_video.mp4"
        cmd_middle_video = [
            'ffmpeg', '-y', 
            '-loop', '1', '-i', str(middle_frame_path),  # Loop the frame
            '-f', 'lavfi', '-i', f'anullsrc=channel_layout=stereo:sample_rate=48000',  # Silent audio
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            '-vf', f'scale={STANDARD_WIDTH}:{STANDARD_HEIGHT}:force_original_aspect_ratio=decrease,pad={STANDARD_WIDTH}:{STANDARD_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black',
            '-r', str(STANDARD_FPS),
            '-t', '0.1',  # Very short duration
            '-movflags', '+faststart',
            str(middle_frame_video_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd_middle_video, capture_output=True, text=True, timeout=60
        )
        
        if result.returncode != 0:
            raise Exception(f"Failed to create middle frame video: {result.stderr}")
        
        # Extract remaining audio (from middle time + 0.1 to end) for mixing with AI audio
        remaining_audio_start = middle_time + 0.1  # After middle frame
        remaining_audio_path = temp_dir / f"{process_id}_remaining_audio.wav"
        
        if remaining_audio_start < duration:
            cmd_remaining_audio = [
                'ffmpeg', '-y', '-i', video_path,
                '-ss', str(remaining_audio_start),
                '-vn',  # No video
                '-c:a', 'pcm_s16le',  # Uncompressed for mixing
                '-ar', '48000', '-ac', '2',
                '-t', str(ai_duration),  # Only for AI video duration
                str(remaining_audio_path)
            ]
            
            result = await asyncio.to_thread(
                subprocess.run, cmd_remaining_audio, capture_output=True, text=True, timeout=60
            )
            
            if result.returncode != 0:
                logger.warning(f"Failed to extract remaining audio, creating silence: {result.stderr}")
                # Create silence if no audio
                cmd_silence = [
                    'ffmpeg', '-y',
                    '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000',
                    '-t', str(ai_duration),
                    '-c:a', 'pcm_s16le',
                    str(remaining_audio_path)
                ]
                
                result = await asyncio.to_thread(
                    subprocess.run, cmd_silence, capture_output=True, text=True, timeout=30
                )
                
                if result.returncode != 0:
                    raise Exception(f"Failed to create silence audio: {result.stderr}")
        else:
            # Create silence if no remaining content
            cmd_silence = [
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000',
                '-t', str(ai_duration),
                '-c:a', 'pcm_s16le',
                str(remaining_audio_path)
            ]
            
            result = await asyncio.to_thread(
                subprocess.run, cmd_silence, capture_output=True, text=True, timeout=30
            )
            
            if result.returncode != 0:
                raise Exception(f"Failed to create silence audio: {result.stderr}")
        
        logger.info(f"Video processing complete:")
        logger.info(f"- Middle frame: {middle_frame_path}")
        logger.info(f"- First part: {first_part_path}")
        logger.info(f"- Middle frame video: {middle_frame_video_path}")
        logger.info(f"- Remaining audio: {remaining_audio_path}")
        
        return str(middle_frame_path), str(first_part_path), str(middle_frame_video_path), ai_duration, str(remaining_audio_path)
        
    except Exception as e:
        logger.error(f"Failed to process video: {e}")
        raise Exception(f"Video processing failed: {str(e)}")

async def _create_mixed_audio(ai_audio_path: str, original_audio_path: str, duration: float) -> str:
    """Mix AI audio with original audio"""
    try:
        import subprocess
        
        temp_dir = Path("/tmp")
        mix_id = str(uuid.uuid4())
        mixed_audio_path = temp_dir / f"{mix_id}_mixed_audio.wav"
        
        logger.info(f"Creating mixed audio for duration: {duration}s")
        
        if ai_audio_path and Path(ai_audio_path).exists():
            # First repair AI audio
            repaired_ai_audio = await _repair_audio(ai_audio_path)
            
            # Mix AI audio with original audio (50% each)
            cmd_mix = [
                'ffmpeg', '-y',
                '-i', repaired_ai_audio,      # AI audio
                '-i', original_audio_path,    # Original audio
                '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=shortest:weights=0.6 0.4',  # AI audio 60%, original 40%
                '-c:a', 'pcm_s16le',
                '-ar', '48000', '-ac', '2',
                '-t', str(duration),
                str(mixed_audio_path)
            ]
            
            result = await asyncio.to_thread(
                subprocess.run, cmd_mix, capture_output=True, text=True, timeout=60
            )
            
            # Clean up repaired AI audio
            try:
                Path(repaired_ai_audio).unlink()
            except:
                pass
            
            if result.returncode != 0:
                logger.warning(f"Audio mixing failed, using original audio only: {result.stderr}")
                # Fallback to original audio only
                cmd_original = [
                    'ffmpeg', '-y',
                    '-i', original_audio_path,
                    '-c:a', 'pcm_s16le',
                    '-ar', '48000', '-ac', '2',
                    '-t', str(duration),
                    str(mixed_audio_path)
                ]
                
                result = await asyncio.to_thread(
                    subprocess.run, cmd_original, capture_output=True, text=True, timeout=60
                )
                
                if result.returncode != 0:
                    raise Exception(f"Failed to process audio: {result.stderr}")
        else:
            logger.info("No AI audio available, using original audio only")
            # Use only original audio
            cmd_original = [
                'ffmpeg', '-y',
                '-i', original_audio_path,
                '-c:a', 'pcm_s16le',
                '-ar', '48000', '-ac', '2',
                '-t', str(duration),
                str(mixed_audio_path)
            ]
            
            result = await asyncio.to_thread(
                subprocess.run, cmd_original, capture_output=True, text=True, timeout=60
            )
            
            if result.returncode != 0:
                raise Exception(f"Failed to process original audio: {result.stderr}")
        
        logger.info(f"Mixed audio created: {mixed_audio_path}")
        return str(mixed_audio_path)
        
    except Exception as e:
        logger.error(f"Failed to create mixed audio: {e}")
        raise Exception(f"Mixed audio creation failed: {str(e)}")

async def _standardize_video(video_path: str, is_ai_video: bool = False) -> str:
    """Standardize video to consistent format"""
    try:
        import subprocess
        
        temp_dir = Path("/tmp")
        output_id = str(uuid.uuid4())
        standardized_path = temp_dir / f"{output_id}_standardized.mp4"
        
        logger.info(f"Standardizing video: {video_path}")
        
        if is_ai_video:
            # AI video should already be correct size but ensure FPS and encoding
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-r', str(STANDARD_FPS),  # Ensure standard FPS
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                '-movflags', '+faststart',
                '-t', '3.5',  # Ensure exactly 3.5 seconds
                str(standardized_path)
            ]
        else:
            # Regular video - scale and standardize
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vf', f'scale={STANDARD_WIDTH}:{STANDARD_HEIGHT}:force_original_aspect_ratio=decrease,pad={STANDARD_WIDTH}:{STANDARD_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black',
                '-r', str(STANDARD_FPS),
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                '-movflags', '+faststart',
                str(standardized_path)
            ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, timeout=120
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg standardization failed: {result.stderr}")
            raise Exception(f"Video standardization failed: {result.stderr}")
        
        logger.info(f"Successfully standardized video: {standardized_path}")
        return str(standardized_path)
        
    except Exception as e:
        logger.error(f"Failed to standardize video: {e}")
        raise Exception(f"Video standardization failed: {str(e)}")

async def _repair_audio(audio_path: str) -> str:
    """Repair and normalize potentially corrupted AI-generated audio"""
    try:
        import subprocess
        
        temp_dir = Path("/tmp")
        repair_id = str(uuid.uuid4())
        repaired_path = temp_dir / f"{repair_id}_repaired_audio.wav"
        
        logger.info(f"Repairing potentially corrupted audio: {audio_path}")
        
        # First, try to repair and convert to WAV (more robust format)
        cmd_repair = [
            'ffmpeg', '-y',
            '-i', audio_path,
            '-vn',  # No video
            '-ar', '48000',  # Standard sample rate
            '-ac', '2',  # Force stereo
            '-c:a', 'pcm_s16le',  # Uncompressed PCM (most compatible)
            '-af', 'aresample=48000:async=1:first_pts=0',  # Resample and fix timing
            '-t', '3.5',  # Exact 3.5 seconds
            str(repaired_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd_repair, capture_output=True, text=True, timeout=60
        )
        
        if result.returncode != 0:
            logger.warning(f"Audio repair failed, creating silence: {result.stderr}")
            # If repair fails, create 3.5 seconds of silence as fallback
            cmd_silence = [
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000',
                '-t', '3.5',
                '-c:a', 'pcm_s16le',
                str(repaired_path)
            ]
            
            result = await asyncio.to_thread(
                subprocess.run, cmd_silence, capture_output=True, text=True, timeout=30
            )
            
            if result.returncode != 0:
                raise Exception(f"Failed to create silence audio: {result.stderr}")
        
        logger.info(f"Audio repaired successfully: {repaired_path}")
        return str(repaired_path)
        
    except Exception as e:
        logger.error(f"Failed to repair audio: {e}")
        raise Exception(f"Audio repair failed: {str(e)}")

async def _merge_video_audio_standardized(video_path: str, audio_path: str) -> str:
    """Merge video and audio with exact duration matching"""
    try:
        import subprocess
        
        temp_dir = Path("/tmp")
        output_id = str(uuid.uuid4())
        merged_path = temp_dir / f"{output_id}_merged.mp4"
        
        logger.info(f"Merging video {video_path} with audio {audio_path}")
        
        # Merge with exact 3.5-second duration
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,  # Video input
            '-i', audio_path,  # Audio input
            '-c:v', 'copy',  # Copy video stream (already standardized)
            '-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-ac', '2',  # Standardize audio
            '-movflags', '+faststart',
            '-t', '3.5',  # Exact 3.5 seconds
            '-shortest',  # Stop when shortest stream ends
            '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
            str(merged_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, timeout=120
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg merge failed: {result.stderr}")
            # If merge still fails, create video without audio
            logger.warning("Creating video without mixed audio")
            cmd_no_audio = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-c:v', 'copy',
                '-an',  # No audio
                '-t', '3.5',
                str(merged_path)
            ]
            
            result = await asyncio.to_thread(
                subprocess.run, cmd_no_audio, capture_output=True, text=True, timeout=60
            )
            
            if result.returncode != 0:
                raise Exception(f"Failed to create video even without audio: {result.stderr}")
        
        logger.info(f"Successfully merged video and audio: {merged_path}")
        return str(merged_path)
        
    except Exception as e:
        logger.error(f"Failed to merge video and audio: {e}")
        raise Exception(f"Video-audio merge failed: {str(e)}")

async def _concatenate_videos_standardized(video_paths: list) -> str:
    """Concatenate videos with consistent encoding"""
    try:
        import subprocess
        
        temp_dir = Path("/tmp")
        output_id = str(uuid.uuid4())
        final_path = temp_dir / f"{output_id}_final.mp4"
        
        # Filter out empty or very short videos
        valid_paths = []
        for path in video_paths:
            if Path(path).exists():
                # Check if video has content
                cmd_duration = [
                    'ffprobe', '-v', 'error', '-show_entries',
                    'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                    path
                ]
                
                result = await asyncio.to_thread(
                    subprocess.run, cmd_duration, capture_output=True, text=True, timeout=30
                )
                
                if result.returncode == 0:
                    duration = float(result.stdout.strip())
                    if duration > 0.05:  # At least 0.05 seconds
                        valid_paths.append(path)
                        logger.info(f"Including video part: {path} (duration: {duration}s)")
                    else:
                        logger.info(f"Skipping very short video: {path} (duration: {duration}s)")
        
        if not valid_paths:
            raise Exception("No valid video parts to concatenate")
        
        # Create concat file
        concat_file = temp_dir / f"{output_id}_concat.txt"
        with open(concat_file, 'w') as f:
            for path in valid_paths:
                f.write(f"file '{path}'\n")
        
        logger.info(f"Concatenating {len(valid_paths)} video parts")
        
        # Use concat demuxer for perfect concatenation
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0', '-i', str(concat_file),
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart',
            '-avoid_negative_ts', 'make_zero',
            str(final_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, timeout=300
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg concatenation failed: {result.stderr}")
            raise Exception(f"Video concatenation failed: {result.stderr}")
        
        # Cleanup concat file
        concat_file.unlink()
        
        logger.info(f"Successfully created final video: {final_path}")
        return str(final_path)
        
    except Exception as e:
        logger.error(f"Failed to concatenate videos: {e}")
        raise Exception(f"Video concatenation failed: {str(e)}")

async def _upload_video_to_supabase(local_video_path: str, sender_uid: str) -> str:
    """Upload video to Supabase storage and return public URL"""
    try:
        video_path = Path(local_video_path)
        if not video_path.exists():
            raise Exception(f"Video file not found: {local_video_path}")

        # Generate unique filename for Supabase storage
        video_id = str(uuid.uuid4())
        storage_path = f"videos/{sender_uid}/{video_id}.mp4"

        # Read video file
        with open(video_path, "rb") as video_file:
            video_data = video_file.read()

        logger.info(f"Uploading video to Supabase: {storage_path}")

        # Upload to Supabase storage
        try:
            result = supabase.storage.from_("videos").upload(
                path=storage_path,
                file=video_data,
                file_options={
                    "content-type": "video/mp4",
                    "cache-control": "3600"
                }
            )
            logger.info(f"Upload result: {result}")

        except Exception as upload_error:
            logger.error(f"Upload failed: {upload_error}")
            raise Exception(f"Supabase upload failed: {upload_error}")

        # Get public URL
        try:
            url_result = supabase.storage.from_("videos").get_public_url(storage_path)
            logger.info(f"Generated public URL: {url_result}")

            if not url_result:
                raise Exception("Failed to get public URL")

            return url_result

        except Exception as url_error:
            logger.error(f"Failed to get public URL: {url_error}")
            raise Exception(f"Failed to get public URL: {url_error}")

    except Exception as e:
        logger.error(f"Failed to upload video to Supabase: {e}")
        raise Exception(f"Storage upload failed: {str(e)}")

async def _save_chat_messages_to_firebase(sender_uid: str, receiver_list: list, video_url: str, prompt: str, caption: str, is_video_only: bool):
    """Save chat messages with video URL to Firebase for each receiver"""
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        from datetime import datetime
        import pytz

        # Initialize Firebase Admin if not already done
        if not firebase_admin._apps:
            try:
                # Use the specified service account file path
                cred = credentials.Certificate("/etc/secrets/services")
                firebase_admin.initialize_app(cred)
            except Exception as e:
                logger.error(f"Failed to initialize Firebase with service account: {e}")
                raise Exception("Firebase initialization failed")

        db = firestore.client()

        # Current timestamp with timezone
        ist = pytz.timezone('Asia/Kolkata')
        timestamp = datetime.now(ist)

        logger.info(f"Saving video messages to Firebase for {len(receiver_list)} receivers")

        for receiver_id in receiver_list:
            if not receiver_id:  # Skip empty receiver IDs
                continue

            try:
                logger.info(f"Processing message for receiver: {receiver_id}")

                # Check if receiver_id ends with "(group)"
                if receiver_id.endswith("(group)"):
                    # Handle group receiver - save to existing group document (remove "(group)" suffix)
                    group_id = receiver_id.replace("(group)", "")  # Remove the "(group)" suffix
                    
                    # Create message document for group
                    group_message_data = {
                        "senderId": sender_uid,
                        "text": prompt,
                        "videoUrl": video_url,
                        "messageType": "video",
                        "timestamp": timestamp,
                        "isRead": False,
                        "createdAt": timestamp,
                        "updatedAt": timestamp,
                        "hasVideo": True,
                        "mediaType": "video",
                        "videoStatus": "uploaded"
                    }
                    
                    # Add caption field if caption exists
                    if caption:
                        group_message_data["caption"] = caption

                    # Save to groups/{group_id}/messages/ (without "(group)" in the path)
                    doc_ref = db.collection("groups").document(group_id).collection("messages").add(group_message_data)
                    message_id = doc_ref[1].id
                    logger.info(f"Video message saved to groups/{group_id}/messages/ with ID: {message_id}")

                else:
                    # Handle regular individual receiver (existing logic)
                    message_data = {
                        "senderId": sender_uid,
                        "receiverId": receiver_id,
                        "text": prompt,
                        "videoUrl": video_url,
                        "messageType": "video",
                        "timestamp": timestamp,
                        "isRead": False,
                        "createdAt": timestamp,
                        "updatedAt": timestamp,
                        "hasVideo": True,
                        "mediaType": "video",
                        "videoStatus": "uploaded"
                    }
                    
                    # Add caption field if caption exists
                    if caption:
                        message_data["caption"] = caption

                    # Save message to chats/{receiver_id}/messages/ collection
                    doc_ref = db.collection("chats").document(receiver_id).collection("messages").add(message_data)
                    message_id = doc_ref[1].id
                    logger.info(f"Video message saved to chats/{receiver_id}/messages/ with ID: {message_id}")

                    # Also save to sender's chat collection for their own reference
                    doc_ref_sender = db.collection("chats").document(sender_uid).collection("messages").add(message_data)
                    sender_message_id = doc_ref_sender[1].id
                    logger.info(f"Video message saved to chats/{sender_uid}/messages/ with ID: {sender_message_id}")

                    # Create or update chat document (keeping original chat logic for main chat list)
                    chat_participants = sorted([sender_uid, receiver_id])
                    chat_id = f"{chat_participants[0]}_{chat_participants[1]}"

                    chat_data = {
                        "participants": [sender_uid, receiver_id],
                        "participantIds": chat_participants,
                        "lastMessage": prompt,
                        "lastMessageType": "video",
                        "lastMessageTimestamp": timestamp,
                        "lastSenderId": sender_uid,
                        "lastVideoUrl": video_url,
                        "lastMediaType": "video",
                        "hasUnreadVideo": True,
                        "updatedAt": timestamp,
                        "unreadCount": {
                            receiver_id: firestore.Increment(1)
                        }
                    }
                    
                    # Add caption to chat data if it exists
                    if caption:
                        chat_data["lastCaption"] = caption

                    # Create chat if it doesn't exist, or update if it does
                    chat_ref = db.collection("chats").document(chat_id)

                    # Check if chat exists
                    chat_doc = chat_ref.get()
                    if chat_doc.exists:
                        # Update existing chat with video-specific fields
                        update_data = {
                            "lastMessage": prompt,
                            "lastMessageType": "video",
                            "lastMessageTimestamp": timestamp,
                            "lastSenderId": sender_uid,
                            "lastVideoUrl": video_url,
                            "lastMediaType": "video",
                            "hasUnreadVideo": True,
                            "updatedAt": timestamp,
                            f"unreadCount.{receiver_id}": firestore.Increment(1)
                        }
                        
                        # Add caption to update if it exists
                        if caption:
                            update_data["lastCaption"] = caption
                        
                        chat_ref.update(update_data)
                        logger.info(f"Updated existing chat with video: {chat_id}")
                    else:
                        # Create new chat with video data
                        chat_data["createdAt"] = timestamp
                        chat_data["unreadCount"] = {
                            sender_uid: 0,
                            receiver_id: 1
                        }
                        chat_ref.set(chat_data)
                        logger.info(f"Created new chat with video: {chat_id}")

            except Exception as e:
                logger.error(f"Failed to save video message for receiver {receiver_id}: {e}")
                continue  # Continue with other receivers even if one fails

        logger.info("Successfully saved all video messages with URLs to Firebase")

    except Exception as e:
        logger.error(f"Failed to save chat messages to Firebase: {e}", exc_info=True)
        # Don't raise exception here - video generation was successful
        # Just log the error and continue

def _predict_video(image_path: str, prompt: str):
    """Synchronous function to call the Gradio client for 3.5-second video"""
    try:
        enhanced_prompt = f"{prompt} bring this image to life with cinematic motion and smooth animation"
        negative_prompt = "vivid tone, overexposed, static, blurry details, subtitles, stylized, artwork, painting, screen, static, grayscale, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, distorted, malformed limbs, fused fingers, static frame, messy background, three legs, crowded background, walking backwards"
        
        return client.predict(
            input_image=handle_file(image_path),
            prompt=enhanced_prompt,
            steps=6,
            negative_prompt=negative_prompt,
            duration_seconds=3.5,
            guidance_scale=1,
            guidance_scale_2=1,
            seed=42,
            randomize_seed=True,
            api_name="/generate_video"
        )
    except Exception as e:
        logger.error(f"Gradio client prediction failed: {e}")
        raise

def _predict_audio(prompt: str):
    """Synchronous function to call the Audio Gradio client for 3.5-second audio"""
    try:
        logger.info(f"Generating 3.5-second audio with prompt: {prompt}")
        
        result = audio_client.predict(
            prompt=prompt,
            duration=2,  # 5 seconds
            cfg_strength=4.5,
            num_steps=3,
            variant="meanaudio_s_full",
            seed=42,
            api_name="/predict"
        )
        
        logger.info(f"Audio generation result: {result}")
        
        # Validate the result
        if not result or len(result) == 0:
            logger.warning("Audio generation returned empty result")
            return None
            
        audio_path = result[0]  # Return the first element (filepath) from the tuple
        
        # Check if the audio file actually exists and has content
        if not audio_path or not Path(audio_path).exists():
            logger.warning(f"Audio file does not exist: {audio_path}")
            return None
            
        # Check if audio file has reasonable size (at least 1KB)
        if Path(audio_path).stat().st_size < 1024:
            logger.warning(f"Audio file is too small: {Path(audio_path).stat().st_size} bytes")
            return None
            
        return audio_path
        
    except Exception as e:
        logger.error(f"Audio Gradio client prediction failed: {e}")
        return None  # Return None instead of raising, so we can handle gracefully

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=300,  # 5 minutes keep alive
        timeout_graceful_shutdown=30
    )
