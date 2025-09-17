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

app = FastAPI(title="AI Video Processor", version="1.0.0")

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
    """Process input video: extract middle frame, generate video from middle to end, merge with audio"""
    temp_video_path = None
    temp_middle_frame_path = None
    temp_first_part_path = None
    temp_second_part_audio_path = None
    temp_generated_video_path = None
    temp_generated_audio_path = None
    temp_merged_second_part_path = None
    temp_final_merged_path = None

    try:
        # Improved video validation
        content_type = file.content_type or ""
        filename = file.filename or ""

        # Check for video file types
        valid_content_types = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/webm']
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.3gp']

        is_valid_content_type = any(content_type.startswith(ct) for ct in valid_content_types)
        is_valid_extension = any(filename.lower().endswith(ext) for ext in valid_extensions)

        if not (is_valid_content_type or is_valid_extension):
            logger.warning(f"Invalid file - Content-Type: {content_type}, Filename: {filename}")
            raise HTTPException(status_code=400, detail="File must be a video (mp4, avi, mov, mkv, webm)")

        if len(prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        logger.info(f"Starting video processing for user {sender_uid}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Receivers: {receiver_uids}")
        logger.info(f"File info - Content-Type: {content_type}, Filename: {filename}")

        # Create temp directory if it doesn't exist
        temp_dir = Path("/tmp")
        temp_dir.mkdir(exist_ok=True)

        # Save uploaded video temporarily
        video_id = str(uuid.uuid4())
        file_extension = Path(filename).suffix or '.mp4'
        temp_video_path = temp_dir / f"{video_id}{file_extension}"

        # Save file
        with open(temp_video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Video saved to {temp_video_path}")

        # Validate file size (optional)
        file_size = temp_video_path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # 100MB limit for videos
            raise HTTPException(status_code=400, detail="File too large (max 100MB)")

        # Check if clients are available
        if client is None:
            raise HTTPException(status_code=503, detail="AI video service not available")
        
        if audio_client is None:
            raise HTTPException(status_code=503, detail="AI audio service not available")

        if supabase is None:
            raise HTTPException(status_code=503, detail="Storage service not available")

        # Process the video: extract middle frame and split video
        logger.info("Processing video: extracting middle frame and splitting...")
        middle_frame_path, first_part_path, second_part_duration = await _process_input_video(str(temp_video_path))
        
        temp_middle_frame_path = middle_frame_path
        temp_first_part_path = first_part_path

        logger.info(f"Middle frame extracted: {middle_frame_path}")
        logger.info(f"First part saved: {first_part_path}")
        logger.info(f"Second part duration: {second_part_duration} seconds")

        # Extract audio from the second part of original video
        temp_second_part_audio_path = await _extract_audio_from_second_part(str(temp_video_path), second_part_duration)
        logger.info(f"Second part audio extracted: {temp_second_part_audio_path}")

        # Start both video and audio generation concurrently for the second part
        logger.info("Starting video and audio generation for second part...")
        
        # Create tasks for both generations
        video_task = asyncio.create_task(
            asyncio.wait_for(
                asyncio.to_thread(_predict_video, middle_frame_path, prompt, second_part_duration),
                timeout=300.0  # 5 minutes timeout
            )
        )

        audio_task = asyncio.create_task(
            asyncio.wait_for(
                asyncio.to_thread(_predict_audio, prompt, second_part_duration),
                timeout=300.0  # 5 minutes timeout
            )
        )

        # Wait for both tasks to complete
        video_result, audio_result = await asyncio.gather(video_task, audio_task)

        if not video_result or len(video_result) < 2:
            raise HTTPException(status_code=500, detail="Invalid response from video AI model")

        if not audio_result:
            raise HTTPException(status_code=500, detail="Invalid response from audio AI model")

        generated_video_path = video_result[0].get("video") if isinstance(video_result[0], dict) else video_result[0]
        seed_used = video_result[1] if len(video_result) > 1 else "unknown"
        generated_audio_path = audio_result

        temp_generated_video_path = generated_video_path
        temp_generated_audio_path = generated_audio_path

        logger.info(f"Generated video: {generated_video_path}")
        logger.info(f"Generated audio: {generated_audio_path}")

        # Merge generated video with generated audio + original second part audio
        merged_second_part_path = await _merge_second_part_components(
            generated_video_path, 
            generated_audio_path, 
            temp_second_part_audio_path
        )
        temp_merged_second_part_path = merged_second_part_path
        logger.info(f"Second part merged: {merged_second_part_path}")

        # Finally, merge first part with the processed second part
        final_video_path = await _merge_first_and_second_parts(first_part_path, merged_second_part_path)
        temp_final_merged_path = final_video_path
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
        logger.error("Video/Audio generation timed out after 5 minutes")
        raise HTTPException(
            status_code=408, 
            detail="Video/Audio generation timed out. Please try with a simpler prompt or smaller video."
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process video: {str(e)}"
        )

    finally:
        # Cleanup temporary files
        temp_files = [
            temp_video_path, temp_middle_frame_path, temp_first_part_path,
            temp_second_part_audio_path, temp_generated_video_path, 
            temp_generated_audio_path, temp_merged_second_part_path, temp_final_merged_path
        ]
        
        for temp_path in temp_files:
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                    logger.info(f"Cleaned up temp file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")

async def _process_input_video(video_path: str):
    """Extract middle frame and split video into first part and calculate second part duration"""
    try:
        import subprocess
        import json
        
        # Get video duration and frame count
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"FFprobe failed: {result.stderr}")
        
        video_info = json.loads(result.stdout)
        duration = float(video_info['format']['duration'])
        middle_time = duration / 2
        second_part_duration = duration - middle_time
        
        logger.info(f"Video duration: {duration}s, Middle time: {middle_time}s, Second part duration: {second_part_duration}s")
        
        temp_dir = Path("/tmp")
        unique_id = str(uuid.uuid4())
        
        # Extract middle frame
        middle_frame_path = temp_dir / f"{unique_id}_middle_frame.jpg"
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-ss', str(middle_time), 
            '-frames:v', '1', '-q:v', '2', str(middle_frame_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"Middle frame extraction failed: {result.stderr}")
        
        # Extract first part (from start to middle)
        first_part_path = temp_dir / f"{unique_id}_first_part.mp4"
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-t', str(middle_time), 
            '-c', 'copy', str(first_part_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"First part extraction failed: {result.stderr}")
        
        return str(middle_frame_path), str(first_part_path), second_part_duration
        
    except Exception as e:
        logger.error(f"Failed to process input video: {e}")
        raise Exception(f"Video processing failed: {str(e)}")

async def _extract_audio_from_second_part(video_path: str, second_part_duration: float):
    """Extract audio from the second part of the original video"""
    try:
        import subprocess
        import json
        
        # Get video duration first
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', video_path]
        result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"FFprobe failed: {result.stderr}")
        
        video_info = json.loads(result.stdout)
        duration = float(video_info['format']['duration'])
        middle_time = duration / 2
        
        temp_dir = Path("/tmp")
        unique_id = str(uuid.uuid4())
        second_part_audio_path = temp_dir / f"{unique_id}_second_part_audio.wav"
        
        # Extract audio from middle to end
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-ss', str(middle_time),
            '-t', str(second_part_duration), '-vn', '-acodec', 'pcm_s16le',
            str(second_part_audio_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"Second part audio extraction failed: {result.stderr}")
        
        return str(second_part_audio_path)
        
    except Exception as e:
        logger.error(f"Failed to extract second part audio: {e}")
        raise Exception(f"Audio extraction failed: {str(e)}")

async def _merge_second_part_components(generated_video_path: str, generated_audio_path: str, original_second_audio_path: str):
    """Merge generated video with both generated audio and original second part audio"""
    try:
        import subprocess
        
        temp_dir = Path("/tmp")
        unique_id = str(uuid.uuid4())
        
        # First, let's check the actual duration of generated video
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', generated_video_path]
        result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            import json
            video_info = json.loads(result.stdout)
            actual_video_duration = float(video_info['format']['duration'])
            logger.info(f"Generated video actual duration: {actual_video_duration}s")
        else:
            actual_video_duration = 2.0  # fallback
            logger.warning(f"Could not get video duration, using fallback: {actual_video_duration}s")
        
        # Trim original second part audio to match generated video duration
        trimmed_original_audio_path = temp_dir / f"{unique_id}_trimmed_original_audio.wav"
        cmd = [
            'ffmpeg', '-y', '-i', original_second_audio_path, 
            '-t', str(actual_video_duration), '-c', 'copy', str(trimmed_original_audio_path)
        ]
        
        result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Failed to trim original audio, proceeding without it")
            trimmed_original_audio_path = None
        
        # Trim generated audio to match video duration
        trimmed_generated_audio_path = temp_dir / f"{unique_id}_trimmed_generated_audio.wav"
        cmd = [
            'ffmpeg', '-y', '-i', generated_audio_path, 
            '-t', str(actual_video_duration), '-c', 'copy', str(trimmed_generated_audio_path)
        ]
        
        result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Failed to trim generated audio")
            trimmed_generated_audio_path = generated_audio_path  # use original
        
        merged_path = temp_dir / f"{unique_id}_merged_second_part.mp4"
        
        logger.info(f"Merging generated video with audio tracks")
        
        # Build ffmpeg command based on available audio
        if trimmed_original_audio_path and Path(trimmed_original_audio_path).exists():
            # Mix both audio tracks
            cmd = [
                'ffmpeg', '-y',
                '-i', generated_video_path,  # Generated video
                '-i', str(trimmed_generated_audio_path),  # Generated audio
                '-i', str(trimmed_original_audio_path),  # Original second part audio
                '-filter_complex', '[1:a][2:a]amix=inputs=2:duration=shortest:dropout_transition=2[mixedaudio]',
                '-map', '0:v',  # Use video from first input
                '-map', '[mixedaudio]',  # Use mixed audio
                '-c:v', 'libx264',  # Re-encode video for compatibility
                '-c:a', 'aac',  # Encode audio to AAC
                '-shortest',  # Finish when shortest stream ends
                '-avoid_negative_ts', 'make_zero',  # Fix timing issues
                str(merged_path)
            ]
        else:
            # Use only generated audio
            cmd = [
                'ffmpeg', '-y',
                '-i', generated_video_path,  # Generated video
                '-i', str(trimmed_generated_audio_path),  # Generated audio only
                '-c:v', 'libx264',  # Re-encode video for compatibility
                '-c:a', 'aac',  # Encode audio to AAC
                '-shortest',  # Finish when shortest stream ends
                '-avoid_negative_ts', 'make_zero',  # Fix timing issues
                str(merged_path)
            ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, timeout=180
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg second part merge failed: {result.stderr}")
            raise Exception(f"Second part merging failed: {result.stderr}")
        
        if not merged_path.exists():
            raise Exception("Merged second part video was not created")
        
        # Cleanup temporary audio files
        for temp_audio in [trimmed_original_audio_path, trimmed_generated_audio_path]:
            if temp_audio and Path(temp_audio).exists() and temp_audio != generated_audio_path:
                try:
                    Path(temp_audio).unlink()
                except:
                    pass
        
        logger.info(f"Successfully merged second part: {merged_path}")
        return str(merged_path)
        
    except Exception as e:
        logger.error(f"Failed to merge second part components: {e}")
        raise Exception(f"Second part merging failed: {str(e)}")

async def _merge_first_and_second_parts(first_part_path: str, second_part_path: str):
    """Merge first part with processed second part"""
    try:
        import subprocess
        
        temp_dir = Path("/tmp")
        unique_id = str(uuid.uuid4())
        
        # First, get duration of both parts to verify they match expected lengths
        for part_name, part_path in [("first", first_part_path), ("second", second_part_path)]:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', part_path]
            result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                import json
                video_info = json.loads(result.stdout)
                duration = float(video_info['format']['duration'])
                logger.info(f"{part_name} part duration: {duration}s")
            else:
                logger.warning(f"Could not get {part_name} part duration")
        
        # Ensure both videos have compatible formats before concatenation
        temp_first_normalized = temp_dir / f"{unique_id}_first_normalized.mp4"
        temp_second_normalized = temp_dir / f"{unique_id}_second_normalized.mp4"
        
        # Normalize first part
        cmd = [
            'ffmpeg', '-y', '-i', first_part_path,
            '-c:v', 'libx264', '-c:a', 'aac',
            '-r', '24', '-pix_fmt', 'yuv420p',
            '-avoid_negative_ts', 'make_zero',
            str(temp_first_normalized)
        ]
        
        result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"First part normalization failed: {result.stderr}")
            temp_first_normalized = first_part_path  # use original
        
        # Normalize second part
        cmd = [
            'ffmpeg', '-y', '-i', second_part_path,
            '-c:v', 'libx264', '-c:a', 'aac',
            '-r', '24', '-pix_fmt', 'yuv420p',
            '-avoid_negative_ts', 'make_zero',
            str(temp_second_normalized)
        ]
        
        result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Second part normalization failed: {result.stderr}")
            temp_second_normalized = second_part_path  # use original
        
        final_path = temp_dir / f"{unique_id}_final_video.mp4"
        
        # Create a temporary file list for concatenation
        filelist_path = temp_dir / f"{unique_id}_filelist.txt"
        
        with open(filelist_path, 'w') as f:
            f.write(f"file '{temp_first_normalized}'\n")
            f.write(f"file '{temp_second_normalized}'\n")
        
        logger.info(f"Concatenating normalized parts")
        
        # Concatenate videos using concat demuxer (more reliable)
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
            '-i', str(filelist_path), 
            '-c:v', 'libx264', '-c:a', 'aac',  # Re-encode for consistency
            '-avoid_negative_ts', 'make_zero',
            str(final_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, timeout=180
        )
        
        # Cleanup temporary files
        for temp_file in [filelist_path, temp_first_normalized, temp_second_normalized]:
            if temp_file and Path(temp_file).exists() and temp_file not in [first_part_path, second_part_path]:
                try:
                    Path(temp_file).unlink()
                except:
                    pass
        
        if result.returncode != 0:
            logger.error(f"FFmpeg final merge failed: {result.stderr}")
            raise Exception(f"Final video merging failed: {result.stderr}")
        
        if not final_path.exists():
            raise Exception("Final merged video was not created")
        
        # Verify final video duration
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(final_path)]
        result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            import json
            video_info = json.loads(result.stdout)
            final_duration = float(video_info['format']['duration'])
            logger.info(f"Final video duration: {final_duration}s")
        else:
            logger.warning("Could not verify final video duration")
        
        logger.info(f"Successfully created final video: {final_path}")
        return str(final_path)
        
    except Exception as e:
        logger.error(f"Failed to merge first and second parts: {e}")
        raise Exception(f"Final video merging failed: {str(e)}")

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

def _predict_video(image_path: str, prompt: str, duration: float):
    """Synchronous function to call the Gradio client with specific duration"""
    try:
        # LTX typically generates 2-5 second videos, so we'll use minimum 2 seconds
        # and ensure we don't exceed typical limits
        target_duration = max(2.0, min(duration, 5.0))
        target_frames = max(9, int(target_duration * 24))  # 24fps is more common for LTX
        
        logger.info(f"Requesting video generation: duration={target_duration}s, frames={target_frames}")
        
        return client.predict(
            prompt=prompt,
            negative_prompt="worst quality, inconsistent motion, blurry, artifacts",
            input_image_filepath=handle_file(image_path),
            input_video_filepath=None,
            height_ui=960,
            width_ui=544,
            mode="image-to-video",
            duration_ui=5,  # Use constrained duration
            ui_frames_to_use=9,
            seed_ui=42,
            randomize_seed=True,
            ui_guidance_scale=5,
            improve_texture_flag=True,
            api_name="/image_to_video"
        )
    except Exception as e:
        logger.error(f"Gradio client prediction failed: {e}")
        raise

def _predict_audio(prompt: str, duration: float):
    """Synchronous function to call the Audio Gradio client with specific duration"""
    try:
        # Constrain audio duration to match video constraints
        target_duration = max(2.0, min(duration, 5.0))
        logger.info(f"Generating audio with prompt: {prompt}, duration: {target_duration}")
        
        result = audio_client.predict(
            prompt=prompt,
            duration=5,  # Use constrained duration
            cfg_strength=4.5,
            num_steps=1,
            variant="meanaudio_s_full",
            seed=42,
            api_name="/predict"
        )
        
        logger.info(f"Audio generation result: {result}")
        return result[0]  # Return the first element (filepath) from the tuple
        
    except Exception as e:
        logger.error(f"Audio Gradio client prediction failed: {e}")
        raise

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
