import os
import uuid
import shutil
import asyncio
import logging
import re
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from dotenv import load_dotenv
from supabase import create_client, Client as SupabaseClient
import uvicorn
import httpx

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
wan_client = None
audio_client = None
supabase: SupabaseClient = None

# Standard video parameters to ensure consistency
STANDARD_WIDTH = 544
STANDARD_HEIGHT = 960
STANDARD_FPS = 24  # Use 24fps as standard (works well for most content)

@app.on_event("startup")
async def startup_event():
    global wan_client, audio_client, supabase
    try:
        logger.info("Initializing WAN2_2 Video client...")
        # Add timeout configuration
        wan_client = Client(
            "Heartsync/wan2_2-I2V-14B-FAST", 
            token=HF_TOKEN,
            httpx_kwargs={"timeout": httpx.Timeout(60.0, connect=30.0, read=60.0)}
        )
        logger.info("WAN2_2 Video client initialized successfully")

        logger.info("Initializing Audio Gradio client...")
        # Add timeout configuration
        audio_client = Client(
            "hkchengrex/MMAudio", 
            token=HF_TOKEN,
            httpx_kwargs={"timeout": httpx.Timeout(60.0, connect=30.0, read=60.0)}
        )
        logger.info("Audio Gradio client initialized successfully")
        
        logger.info("Initializing Supabase client...")
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        # Don't raise - allow server to start even if clients fail
        logger.warning("Server starting with failed clients - they will be retried on first request")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    all_ready = (
        wan_client is not None and 
        audio_client is not None and 
        supabase is not None
    )
    
    return {
        "status": "healthy", 
        "client_ready": all_ready,
        "wan_client_ready": wan_client is not None,
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

def extract_verbs_and_nouns(prompt: str) -> str:
    """Extract action verbs and nouns from prompt using hardcoded lists (case-insensitive)"""
    # Hardcoded list of common action verbs (in gerund form for actions)
    action_verbs = [
        'running', 'walking', 'jumping', 'dancing', 'singing', 'playing', 'eating', 
        'drinking', 'swimming', 'flying', 'driving', 'riding', 'sleeping', 'working',
        'talking', 'laughing', 'crying', 'smiling', 'fighting', 'cooking', 'reading',
        'writing', 'drawing', 'painting', 'climbing', 'falling', 'sitting', 'standing',
        'kicking', 'throwing', 'catching', 'shooting', 'exploding', 'burning', 'flowing',
        'spinning', 'rotating', 'moving', 'shaking', 'vibrating', 'bouncing', 'rolling',
        'sliding', 'gliding', 'floating', 'sinking', 'rising', 'descending', 'ascending',
        'run', 'walk', 'jump', 'dance', 'sing', 'play', 'eat', 'drink', 'swim', 'fly',
        'drive', 'ride', 'sleep', 'work', 'talk', 'laugh', 'cry', 'smile', 'fight',
        'cook', 'read', 'write', 'draw', 'paint', 'climb', 'fall', 'sit', 'stand',
        'kick', 'throw', 'catch', 'shoot', 'explode', 'burn', 'flow', 'spin', 'rotate',
        'move', 'shake', 'vibrate', 'bounce', 'roll', 'slide', 'glide', 'float', 'sink'
    ]
    
    # Hardcoded list of common nouns for sound effects
    nouns = [
        'water', 'fire', 'wind', 'thunder', 'rain', 'snow', 'ice', 'storm', 'lightning',
        'ocean', 'river', 'waterfall', 'wave', 'bird', 'dog', 'cat', 'horse', 'car',
        'truck', 'plane', 'helicopter', 'train', 'boat', 'ship', 'motorcycle', 'bicycle',
        'drum', 'guitar', 'piano', 'bell', 'horn', 'siren', 'alarm', 'clock', 'door',
        'window', 'glass', 'metal', 'wood', 'stone', 'rock', 'explosion', 'gunshot',
        'footsteps', 'crowd', 'applause', 'laughter', 'scream', 'whistle', 'wind chime',
        'rain drop', 'heartbeat', 'breathing', 'coughing', 'sneezing', 'roar', 'growl',
        'chirp', 'meow', 'bark', 'neigh', 'moo', 'quack', 'tweet', 'buzz', 'hiss',
        'crackle', 'splash', 'drip', 'swoosh', 'whoosh', 'thud', 'crash', 'bang',
        'clang', 'ding', 'ring', 'beep', 'honk', 'screech', 'rumble', 'roar'
    ]
    
    # Convert prompt to lowercase for case-insensitive matching
    prompt_lower = prompt.lower()
    
    # Extract matching verbs and nouns
    found_words = []
    
    # Check for verbs
    for verb in action_verbs:
        # Use word boundary to match whole words only
        if re.search(r'\b' + re.escape(verb) + r'\b', prompt_lower):
            found_words.append(verb)
    
    # Check for nouns
    for noun in nouns:
        if re.search(r'\b' + re.escape(noun) + r'\b', prompt_lower):
            found_words.append(noun)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in found_words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)
    
    # Join with commas
    result = ', '.join(unique_words) if unique_words else prompt
    
    logger.info(f"Extracted audio prompt: {result}")
    return result

async def ensure_clients_ready():
    """Ensure clients are initialized, retry if needed"""
    global wan_client, audio_client
    
    if wan_client is None:
        logger.info("Retrying WAN2_2 Video client initialization...")
        try:
            wan_client = Client(
                "Heartsync/wan2_2-I2V-14B-FAST", 
                token=HF_TOKEN,
                httpx_kwargs={"timeout": httpx.Timeout(60.0, connect=30.0, read=60.0)}
            )
            logger.info("WAN2_2 Video client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WAN2_2 client: {e}")
            raise HTTPException(status_code=503, detail="AI video service unavailable")
    
    if audio_client is None:
        logger.info("Retrying Audio Gradio client initialization...")
        try:
            audio_client = Client(
                "hkchengrex/MMAudio", 
                token=HF_TOKEN,
                httpx_kwargs={"timeout": httpx.Timeout(60.0, connect=30.0, read=60.0)}
            )
            logger.info("Audio Gradio client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Audio client: {e}")
            raise HTTPException(status_code=503, detail="AI audio service unavailable")

@app.post("/generate/")
async def generate_video(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    sender_uid: str = Form(...),
    receiver_uids: str = Form(...)
):
    """Generate video with middle frame + AI video (with audio) merged back"""
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
            # Ensure clients are ready before processing
            await ensure_clients_ready()
            
            # Process video: extract middle frame and video parts
            logger.info("Processing video: extracting parts with middle frame...")
            middle_frame_path, first_part_path, middle_frame_video_path = await _process_video_with_middle_frame(str(temp_video_path))
            temp_files.extend([middle_frame_path, first_part_path, middle_frame_video_path])

            # Apply EXIF orientation to ensure middle frame is upright
            from PIL import Image, ImageOps
            try:
                with Image.open(middle_frame_path) as img:
                    # Apply EXIF orientation to correct rotation automatically
                    corrected_img = ImageOps.exif_transpose(img)
                    if corrected_img is None:
                        # If no EXIF data, use original image
                        corrected_img = img
                    corrected_img.save(middle_frame_path)
                    logger.info(f"Middle frame orientation corrected using EXIF data")
            except Exception as e:
                logger.warning(f"Failed to correct middle frame orientation: {e}, proceeding with original frame")

            # Generate video with WAN2_2 API using middle frame
            logger.info(f"Starting video generation with WAN2_2 API using middle frame and prompt: {magic_prompt}")
            
            video_result = await asyncio.wait_for(
                asyncio.to_thread(_predict_video_wan, str(middle_frame_path), magic_prompt),
                timeout=300.0  # 5 minutes timeout
            )

            if not video_result or len(video_result) < 2:
                raise HTTPException(status_code=500, detail="Invalid response from WAN2_2 video AI model")

            ai_video_path = video_result[0].get("video") if isinstance(video_result[0], dict) else video_result[0]
            seed_used = video_result[1] if len(video_result) > 1 else "unknown"

            logger.info(f"AI Video generated locally: {ai_video_path}")

            # Generate audio using the AI video file
            logger.info("Starting audio generation with AI video...")
            
            audio_result = await asyncio.wait_for(
                asyncio.to_thread(_predict_audio, ai_video_path, magic_prompt),
                timeout=300.0  # 5 minutes timeout
            )

            if not audio_result:
                raise HTTPException(status_code=500, detail="Invalid response from audio AI model")

            ai_audio_path = audio_result
            logger.info(f"AI Audio generated locally: {ai_audio_path}")

            # Merge AI video with AI audio
            ai_merged_path = await _merge_video_audio(ai_video_path, ai_audio_path)
            temp_files.append(ai_merged_path)
            logger.info(f"AI video merged with AI audio: {ai_merged_path}")

            # Create final video by concatenating: first_part + middle_frame_video + ai_merged_video
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
    """Extract parts: first_part + middle_frame + middle_frame_video"""
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
        
        logger.info(f"Split points - Middle: {middle_time}s")
        
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
        
        logger.info(f"Video processing complete:")
        logger.info(f"- Middle frame: {middle_frame_path}")
        logger.info(f"- First part: {first_part_path}")
        logger.info(f"- Middle frame video: {middle_frame_video_path}")
        
        return str(middle_frame_path), str(first_part_path), str(middle_frame_video_path)
        
    except Exception as e:
        logger.error(f"Failed to process video: {e}")
        raise Exception(f"Video processing failed: {str(e)}")

def _predict_audio(video_path: str, prompt: str):
    """Synchronous function to call the MMAudio Gradio client"""
    try:
        # Extract verbs and nouns from the prompt
        audio_prompt = extract_verbs_and_nouns(prompt)
        
        logger.info(f"Original prompt: {prompt}")
        logger.info(f"Audio prompt (extracted): {audio_prompt}")
        
        result = audio_client.predict(
            video={"video": handle_file(video_path)},
            prompt=audio_prompt,
            negative_prompt="music,artifacts,fuzzy audio,distortion",
            seed=-1,
            num_steps=25,
            cfg_strength=4.5,
            duration=5,
            api_name="/predict"
        )
        
        logger.info(f"Audio generation result: {result}")
        
        # Extract the audio file path from the result
        if isinstance(result, dict) and "video" in result:
            return result["video"]
        else:
            return result
        
    except Exception as e:
        logger.error(f"Audio Gradio client prediction failed: {e}")
        raise

def _predict_video_wan(image_path: str, prompt: str):
    """Generate video using WAN2_2 API with new parameters"""
    try:
        return wan_client.predict(
            input_image=handle_file(image_path),
            prompt=prompt,
            steps=4,
            negative_prompt=" multiple bodies, overlapping bodies, ghost limbs, duplicate limbs, jitter, unstable movement, morphing face, morphing identity, extra head, extra arms, multiple poses, fast dancing, energetic dancing, motion blur, identity drift ",
            duration_seconds=3.5,
            guidance_scale=1,
            guidance_scale_2=1,
            seed=42,
            randomize_seed=False,
            api_name="/generate_video"
        )
    except Exception as e:
        logger.error(f"WAN2_2 video generation failed: {e}")
        raise

async def _merge_video_audio(video_path: str, audio_path: str) -> str:
    """Merge video and audio files using ffmpeg"""
    try:
        import subprocess
        
        # Generate output path
        temp_dir = Path("/tmp")
        output_id = str(uuid.uuid4())
        merged_path = temp_dir / f"{output_id}_merged.mp4"
        
        logger.info(f"Merging video {video_path} with audio {audio_path}")
        
        # Use ffmpeg to merge video and audio
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-i', video_path,  # input video
            '-i', audio_path,  # input audio
            '-c:v', 'copy',    # copy video codec (no re-encoding)
            '-c:a', 'aac',     # encode audio to AAC
            '-strict', 'experimental',
            '-shortest',       # finish when shortest stream ends
            str(merged_path)
        ]
        
        # Run ffmpeg command
        result = await asyncio.to_thread(
            subprocess.run, cmd, 
            capture_output=True, 
            text=True, 
            timeout=120  # 2 minute timeout for merging
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            raise Exception(f"Video-audio merging failed: {result.stderr}")
        
        if not merged_path.exists():
            raise Exception("Merged video file was not created")
        
        logger.info(f"Successfully merged video and audio: {merged_path}")
        return str(merged_path)
        
    except Exception as e:
        logger.error(f"Failed to merge video and audio: {e}")
        raise Exception(f"Video-audio merging failed: {str(e)}")

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
