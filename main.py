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
        client = Client("Lightricks/ltx-video-distilled", hf_token=HF_TOKEN)
        logger.info("Gradio client initialized successfully")

        logger.info("Initializing Audio Gradio client...")
        audio_client = Client("chenxie95/MeanAudio", hf_token=HF_TOKEN)
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

@app.post("/generate/")
async def generate_video(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    sender_uid: str = Form(...),
    receiver_uids: str = Form(...)
):
    """Generate video from video input and prompt, add audio, then merge them"""
    temp_files = []  # Track all temp files for cleanup

    try:
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

        if len(prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        logger.info(f"Starting video generation for user {sender_uid}")
        logger.info(f"Prompt: {prompt}")
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

        # Check if clients are available
        if client is None:
            raise HTTPException(status_code=503, detail="AI video service not available")
        
        if audio_client is None:
            raise HTTPException(status_code=503, detail="AI audio service not available")

        if supabase is None:
            raise HTTPException(status_code=503, detail="Storage service not available")

        # Process video with standardized parameters
        logger.info("Processing video: extracting middle frame and splitting...")
        middle_frame_path, first_part_path, last_part_path = await _process_video_standardized(str(temp_video_path))
        temp_files.extend([middle_frame_path, first_part_path, last_part_path])

        # Start both AI video and AI audio generation concurrently
        logger.info("Starting AI video and audio generation concurrently...")
        
        # Create tasks for both generations
        ai_video_task = asyncio.create_task(
            asyncio.wait_for(
                asyncio.to_thread(_predict_video, middle_frame_path, prompt),
                timeout=300.0  # 5 minutes timeout
            )
        )

        ai_audio_task = asyncio.create_task(
            asyncio.wait_for(
                asyncio.to_thread(_predict_audio, prompt),
                timeout=300.0  # 5 minutes timeout
            )
        )

        # Wait for both tasks to complete
        ai_video_result, ai_audio_result = await asyncio.gather(ai_video_task, ai_audio_task)

        if not ai_video_result or len(ai_video_result) < 2:
            raise HTTPException(status_code=500, detail="Invalid response from video AI model")

        if not ai_audio_result:
            raise HTTPException(status_code=500, detail="Invalid response from audio AI model")

        ai_video_path = ai_video_result[0].get("video") if isinstance(ai_video_result[0], dict) else ai_video_result[0]
        seed_used = ai_video_result[1] if len(ai_video_result) > 1 else "unknown"
        ai_audio_path = ai_audio_result

        logger.info(f"AI Video generated locally: {ai_video_path}")
        logger.info(f"AI Audio generated locally: {ai_audio_path}")

        # Standardize AI video to match our requirements
        ai_video_standardized = await _standardize_video(ai_video_path, is_ai_video=True)
        temp_files.append(ai_video_standardized)

        # Merge AI video with AI audio
        ai_merged_path = await _merge_video_audio_standardized(ai_video_standardized, ai_audio_path)
        temp_files.append(ai_merged_path)
        logger.info(f"AI video and audio merged: {ai_merged_path}")

        # Create final video by concatenating all three parts
        final_video_path = await _concatenate_videos_standardized([first_part_path, ai_merged_path, last_part_path])
        temp_files.append(final_video_path)
        logger.info(f"Final video created: {final_video_path}")

        # Upload final video to Supabase storage
        video_url = await _upload_video_to_supabase(final_video_path, sender_uid)
        logger.info(f"Final video uploaded to Supabase: {video_url}")

        # Save chat messages to Firebase for each receiver
        receiver_list = [uid.strip() for uid in receiver_uids.split(",") if uid.strip()]
        await _save_chat_messages_to_firebase(sender_uid, receiver_list, video_url, prompt)

        return JSONResponse({
            "success": True,
            "video_url": video_url,
            "seed": seed_used,
            "sender_uid": sender_uid,
            "receiver_uids": receiver_list
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

async def _process_video_standardized(video_path: str) -> tuple:
    """Extract middle frame and split video into standardized parts"""
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
        ai_duration = 5.0  # AI video is exactly 5 seconds
        
        logger.info(f"Split points - Middle: {middle_time}s, AI duration: {ai_duration}s")
        
        # Extract middle frame with rotation and resize to match AI output
        middle_frame_path = temp_dir / f"{process_id}_middle_frame.jpg"
        cmd_frame = [
            'ffmpeg', '-y', '-i', video_path,
            '-ss', str(middle_time), '-vframes', '1',
            '-f', 'image2',  # Specify image format
            '-pix_fmt', 'rgb24',  # Use RGB pixel format
            str(middle_frame_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd_frame, capture_output=True, text=True, timeout=60
        )
        
        if result.returncode != 0:
            raise Exception(f"Failed to extract middle frame: {result.stderr}")
        
        # Extract and standardize first part (start to middle)
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
        
        # Extract and standardize last part (middle + ai_duration to end)
        last_part_start = middle_time + ai_duration
        last_part_path = temp_dir / f"{process_id}_last_part.mp4"
        
        if last_part_start < duration:  # Only if there's content after AI part
            cmd_last = [
                'ffmpeg', '-y', '-i', video_path,
                '-ss', str(last_part_start),
                '-vf', f'scale={STANDARD_WIDTH}:{STANDARD_HEIGHT}:force_original_aspect_ratio=decrease,pad={STANDARD_WIDTH}:{STANDARD_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black',
                '-r', str(STANDARD_FPS),  # Standardize FPS
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-ac', '2',  # Standardize audio
                '-movflags', '+faststart',
                '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                str(last_part_path)
            ]
            
            result = await asyncio.to_thread(
                subprocess.run, cmd_last, capture_output=True, text=True, timeout=120
            )
            
            if result.returncode != 0:
                raise Exception(f"Failed to extract last part: {result.stderr}")
        else:
            # Create empty video file if no last part exists
            cmd_empty = [
                'ffmpeg', '-y', '-f', 'lavfi', 
                '-i', f'color=black:size={STANDARD_WIDTH}x{STANDARD_HEIGHT}:duration=0.1:rate={STANDARD_FPS}',
                '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000',
                '-c:v', 'libx264', '-c:a', 'aac', '-shortest',
                str(last_part_path)
            ]
            
            result = await asyncio.to_thread(
                subprocess.run, cmd_empty, capture_output=True, text=True, timeout=30
            )
            
            if result.returncode != 0:
                raise Exception(f"Failed to create empty last part: {result.stderr}")
        
        logger.info(f"Video processing complete:")
        logger.info(f"- Middle frame: {middle_frame_path}")
        logger.info(f"- First part: {first_part_path}")
        logger.info(f"- Last part: {last_part_path}")
        
        return str(middle_frame_path), str(first_part_path), str(last_part_path)
        
    except Exception as e:
        logger.error(f"Failed to process video: {e}")
        raise Exception(f"Video processing failed: {str(e)}")

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
                '-t', '5.0',  # Ensure exactly 5 seconds
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
            '-t', '5.0',  # Exact 5 seconds
            str(repaired_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd_repair, capture_output=True, text=True, timeout=60
        )
        
        if result.returncode != 0:
            logger.warning(f"Audio repair failed, creating silence: {result.stderr}")
            # If repair fails, create 5 seconds of silence as fallback
            cmd_silence = [
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000',
                '-t', '5.0',
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
        
        # First repair the audio to prevent corruption issues
        repaired_audio_path = await _repair_audio(audio_path)
        
        # Merge with exact 5-second duration using repaired audio
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,  # Video input
            '-i', repaired_audio_path,  # Repaired audio input
            '-c:v', 'copy',  # Copy video stream (already standardized)
            '-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-ac', '2',  # Standardize audio
            '-movflags', '+faststart',
            '-t', '5.0',  # Exact 5 seconds
            '-shortest',  # Stop when shortest stream ends
            '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
            str(merged_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, timeout=120
        )
        
        # Clean up repaired audio file
        try:
            Path(repaired_audio_path).unlink()
        except:
            pass
        
        if result.returncode != 0:
            logger.error(f"FFmpeg merge failed: {result.stderr}")
            # If merge still fails, create video without audio
            logger.warning("Creating video without AI audio due to corruption")
            cmd_no_audio = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-c:v', 'copy',
                '-an',  # No audio
                '-t', '5.0',
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
                    if duration > 0.1:  # At least 0.1 seconds
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

async def _save_chat_messages_to_firebase(sender_uid: str, receiver_list: list, video_url: str, prompt: str):
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
    """Synchronous function to call the Gradio client for 5-second video"""
    try:
        enhanced_prompt = f"{prompt} do not change the face of the person, do not change the face, keep the face clear, maintain teh facial appearance"
        return client.predict(
            prompt=enhanced_prompt,
            negative_prompt="worst quality, inconsistent motion, blurry face, artifacts,distorted face,distorted video,distorted motion,blurry video,blur face,changed face,new face,changed facial appearance",
            input_image_filepath=handle_file(image_path),
            input_video_filepath=None,
            height_ui=STANDARD_HEIGHT,  # Use consistent height
            width_ui=STANDARD_WIDTH,    # Use consistent width
            mode="image-to-video",
            duration_ui=2,  # 5 seconds
            ui_frames_to_use=5,  # 25 frames for 5 seconds at 5fps (AI model standard)
            seed_ui=42,
            randomize_seed=True,
            ui_guidance_scale=5,
            improve_texture_flag=True,
            api_name="/image_to_video"
        )
    except Exception as e:
        logger.error(f"Gradio client prediction failed: {e}")
        raise

def _predict_audio(prompt: str):
    """Synchronous function to call the Audio Gradio client for 5-second audio"""
    try:
        logger.info(f"Generating 5-second audio with prompt: {prompt}")
        
        result = audio_client.predict(
            prompt=prompt,
            duration=2,  # 5 seconds
            cfg_strength=4.5,
            num_steps=1,
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

async def generate_video(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    sender_uid: str = Form(...),
    receiver_uids: str = Form(...)
):
    """Generate video from video input and prompt, add audio, then merge them"""
    temp_files = []  # Track all temp files for cleanup

    try:
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

        if len(prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        logger.info(f"Starting video generation for user {sender_uid}")
        logger.info(f"Prompt: {prompt}")
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

        # Check if clients are available
        if client is None:
            raise HTTPException(status_code=503, detail="AI video service not available")
        
        if audio_client is None:
            raise HTTPException(status_code=503, detail="AI audio service not available")

        if supabase is None:
            raise HTTPException(status_code=503, detail="Storage service not available")

        # Process video with standardized parameters
        logger.info("Processing video: extracting middle frame and splitting...")
        middle_frame_path, first_part_path, last_part_path = await _process_video_standardized(str(temp_video_path))
        temp_files.extend([middle_frame_path, first_part_path, last_part_path])

        # Start both AI video and AI audio generation concurrently
        logger.info("Starting AI video and audio generation concurrently...")
        
        # Create tasks for both generations
        ai_video_task = asyncio.create_task(
            asyncio.wait_for(
                asyncio.to_thread(_predict_video, middle_frame_path, prompt),
                timeout=300.0  # 5 minutes timeout
            )
        )

        ai_audio_task = asyncio.create_task(
            asyncio.wait_for(
                asyncio.to_thread(_predict_audio, prompt),
                timeout=300.0  # 5 minutes timeout
            )
        )

        # Wait for both tasks to complete
        ai_video_result, ai_audio_result = await asyncio.gather(ai_video_task, ai_audio_task)

        if not ai_video_result or len(ai_video_result) < 2:
            raise HTTPException(status_code=500, detail="Invalid response from video AI model")

        # Handle audio failure gracefully
        if not ai_audio_result:
            logger.warning("AI audio generation failed, proceeding without AI audio")

        ai_video_path = ai_video_result[0].get("video") if isinstance(ai_video_result[0], dict) else ai_video_result[0]
        seed_used = ai_video_result[1] if len(ai_video_result) > 1 else "unknown"

        logger.info(f"AI Video generated locally: {ai_video_path}")
        logger.info(f"AI Audio result: {ai_audio_result}")

        # Standardize AI video to match our requirements
        ai_video_standardized = await _standardize_video(ai_video_path, is_ai_video=True)
        temp_files.append(ai_video_standardized)

        # Merge AI video with AI audio (if available)
        if ai_audio_result:
            ai_merged_path = await _merge_video_audio_standardized(ai_video_standardized, ai_audio_result)
            temp_files.append(ai_merged_path)
            logger.info(f"AI video and audio merged: {ai_merged_path}")
        else:
            # Use AI video without audio if audio generation failed
            ai_merged_path = ai_video_standardized
            logger.info(f"Using AI video without audio due to audio generation failure")

        # Create final video by concatenating all three parts
        final_video_path = await _concatenate_videos_standardized([first_part_path, ai_merged_path, last_part_path])
        temp_files.append(final_video_path)
        logger.info(f"Final video created: {final_video_path}")

        # Upload final video to Supabase storage
        video_url = await _upload_video_to_supabase(final_video_path, sender_uid)
        logger.info(f"Final video uploaded to Supabase: {video_url}")

        # Save chat messages to Firebase for each receiver
        receiver_list = [uid.strip() for uid in receiver_uids.split(",") if uid.strip()]
        await _save_chat_messages_to_firebase(sender_uid, receiver_list, video_url, prompt)

        return JSONResponse({
            "success": True,
            "video_url": video_url,
            "seed": seed_used,
            "sender_uid": sender_uid,
            "receiver_uids": receiver_list,
            "has_ai_audio": ai_audio_result is not None
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
