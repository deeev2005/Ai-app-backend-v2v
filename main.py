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
import subprocess

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
async def process_video(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    sender_uid: str = Form(...),
    receiver_uids: str = Form(...)
):
    """Process video by extracting middle frame, generating new video and audio, then stitching everything together"""
    temp_video_path = None
    temp_middle_frame_path = None
    temp_first_half_path = None
    temp_second_half_path = None
    temp_second_half_audio_path = None
    temp_generated_video_path = None
    temp_generated_audio_path = None
    temp_final_video_path = None

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

        # Check if clients are available
        if client is None:
            raise HTTPException(status_code=503, detail="AI video service not available")
        
        if audio_client is None:
            raise HTTPException(status_code=503, detail="AI audio service not available")

        if supabase is None:
            raise HTTPException(status_code=503, detail="Storage service not available")

        # Step 1: Extract middle frame and split video
        logger.info("Extracting middle frame and splitting video...")
        middle_frame_path, first_half_path, second_half_path, second_half_audio_path = await _extract_middle_frame_and_split(str(temp_video_path))

        # Step 2: Generate new video from middle frame and audio concurrently
        logger.info("Starting video and audio generation concurrently...")
        
        video_task = asyncio.create_task(
            asyncio.wait_for(
                asyncio.to_thread(_predict_video, middle_frame_path, prompt),
                timeout=300.0  # 5 minutes timeout
            )
        )

        audio_task = asyncio.create_task(
            asyncio.wait_for(
                asyncio.to_thread(_predict_audio, prompt),
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

        logger.info(f"Generated video: {generated_video_path}")
        logger.info(f"Generated audio: {generated_audio_path}")

        # Step 3: Stitch everything together
        final_video_path = await _stitch_final_video(
            first_half_path, 
            generated_video_path, 
            generated_audio_path, 
            second_half_path, 
            second_half_audio_path
        )

        logger.info(f"Final video stitched: {final_video_path}")

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
        logger.error("Video processing timed out after 5 minutes")
        raise HTTPException(
            status_code=408, 
            detail="Video processing timed out. Please try with a simpler prompt or smaller video."
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
        for temp_path in [temp_video_path, temp_middle_frame_path, temp_first_half_path, 
                         temp_second_half_path, temp_second_half_audio_path, 
                         temp_generated_video_path, temp_generated_audio_path, temp_final_video_path]:
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                    logger.info(f"Cleaned up temp file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")

async def _extract_middle_frame_and_split(video_path: str):
    """Extract middle frame and split video into two halves"""
    try:
        temp_dir = Path("/tmp")
        video_id = str(uuid.uuid4())
        
        # Get video duration first
        duration_cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
            '-of', 'csv=p=0', video_path
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, duration_cmd, 
            capture_output=True, text=True, timeout=30
        )
        
        if result.returncode != 0:
            raise Exception(f"Failed to get video duration: {result.stderr}")
        
        duration = float(result.stdout.strip())
        middle_time = duration / 2
        
        logger.info(f"Video duration: {duration}s, middle time: {middle_time}s")
        
        # Paths for outputs
        middle_frame_path = temp_dir / f"{video_id}_middle_frame.jpg"
        first_half_path = temp_dir / f"{video_id}_first_half.mp4"
        second_half_path = temp_dir / f"{video_id}_second_half.mp4"
        second_half_audio_path = temp_dir / f"{video_id}_second_half_audio.aac"
        
        # Extract middle frame and rotate 90 degrees clockwise
        frame_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-ss', str(middle_time),
            '-vframes', '1',
            '-vf', 'transpose=1',  # Rotate 90 degrees clockwise
            str(middle_frame_path)
        ]
        
        # Split into first half (0 to middle_time)
        first_half_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-t', str(middle_time),
            '-c', 'copy',
            str(first_half_path)
        ]
        
        # Split into second half (middle_time to end)
        second_half_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-ss', str(middle_time),
            '-c', 'copy',
            str(second_half_path)
        ]
        
        # Extract audio from second half
        second_audio_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-ss', str(middle_time),
            '-vn', '-acodec', 'aac',
            str(second_half_audio_path)
        ]
        
        # Run all commands concurrently
        tasks = [
            asyncio.to_thread(subprocess.run, frame_cmd, capture_output=True, text=True, timeout=60),
            asyncio.to_thread(subprocess.run, first_half_cmd, capture_output=True, text=True, timeout=60),
            asyncio.to_thread(subprocess.run, second_half_cmd, capture_output=True, text=True, timeout=60),
            asyncio.to_thread(subprocess.run, second_audio_cmd, capture_output=True, text=True, timeout=60)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Check all results
        for i, result in enumerate(results):
            if result.returncode != 0:
                cmd_names = ["frame extraction", "first half split", "second half split", "second half audio"]
                raise Exception(f"{cmd_names[i]} failed: {result.stderr}")
        
        # Verify files exist
        for path in [middle_frame_path, first_half_path, second_half_path, second_half_audio_path]:
            if not path.exists():
                raise Exception(f"Output file not created: {path}")
        
        logger.info("Successfully extracted middle frame and split video")
        return str(middle_frame_path), str(first_half_path), str(second_half_path), str(second_half_audio_path)
        
    except Exception as e:
        logger.error(f"Failed to extract middle frame and split video: {e}")
        raise Exception(f"Video processing failed: {str(e)}")

async def _stitch_final_video(first_half_path: str, generated_video_path: str, 
                            generated_audio_path: str, second_half_path: str, 
                            second_half_audio_path: str) -> str:
    """Stitch all video parts together with their respective audio tracks"""
    try:
        temp_dir = Path("/tmp")
        output_id = str(uuid.uuid4())
        
        # First, get the frame rate and resolution of the original video
        probe_cmd = [
            'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,width,height',
            '-of', 'csv=p=0', first_half_path
        ]
        
        probe_result = await asyncio.to_thread(
            subprocess.run, probe_cmd, 
            capture_output=True, text=True, timeout=30
        )
        
        if probe_result.returncode != 0:
            raise Exception(f"Failed to probe original video: {probe_result.stderr}")
        
        # Fix: Handle the probe output more carefully
        output_lines = probe_result.stdout.strip().split('\n')
        if not output_lines or not output_lines[0]:
            raise Exception("No video stream information found")
        
        # Parse the first line which should contain fps,width,height
        video_info = output_lines[0].split(',')
        if len(video_info) < 3:
            raise Exception(f"Unexpected probe output format: {output_lines[0]}")
        
        original_fps = video_info[0]
        original_width = video_info[1] 
        original_height = video_info[2]
        
        logger.info(f"Original video properties: {original_width}x{original_height} @ {original_fps} fps")
        
        # Step 1: Normalize all video segments to match original specs
        normalized_first_path = temp_dir / f"{output_id}_normalized_first.mp4"
        normalized_generated_path = temp_dir / f"{output_id}_normalized_generated.mp4"
        normalized_second_path = temp_dir / f"{output_id}_normalized_second.mp4"
        
        # Normalize first half (should already match, but ensure consistency)
        norm_first_cmd = [
            'ffmpeg', '-y',
            '-i', first_half_path,
            '-vf', f'scale={original_width}:{original_height}',
            '-r', str(eval(original_fps)),  # Convert fraction to decimal
            '-c:v', 'libx264', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            str(normalized_first_path)
        ]
        
        # Normalize generated video and add audio - this is the key fix
        norm_generated_cmd = [
            'ffmpeg', '-y',
            '-i', generated_video_path,
            '-i', generated_audio_path,
            '-vf', f'scale={original_width}:{original_height}',
            '-r', str(eval(original_fps)),  # Match original fps
            '-t', '5.0',  # Force exactly 5 seconds
            '-c:v', 'libx264', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            '-shortest',
            str(normalized_generated_path)
        ]
        
        # Normalize second half
        norm_second_cmd = [
            'ffmpeg', '-y',
            '-i', second_half_path,
            '-vf', f'scale={original_width}:{original_height}',
            '-r', str(eval(original_fps)),
            '-c:v', 'libx264', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            str(normalized_second_path)
        ]
        
        # Run normalization commands
        norm_tasks = [
            asyncio.to_thread(subprocess.run, norm_first_cmd, capture_output=True, text=True, timeout=120),
            asyncio.to_thread(subprocess.run, norm_generated_cmd, capture_output=True, text=True, timeout=120),
            asyncio.to_thread(subprocess.run, norm_second_cmd, capture_output=True, text=True, timeout=120)
        ]
        
        norm_results = await asyncio.gather(*norm_tasks)
        
        # Check normalization results
        for i, result in enumerate(norm_results):
            if result.returncode != 0:
                segment_names = ["first half", "generated", "second half"]
                logger.error(f"Normalization failed for {segment_names[i]}: {result.stderr}")
                raise Exception(f"Failed to normalize {segment_names[i]}: {result.stderr}")
        
        # Step 2: Concatenate normalized segments
        final_output_path = temp_dir / f"{output_id}_final.mp4"
        
        # Use filter_complex for seamless concatenation
        concat_cmd = [
            'ffmpeg', '-y',
            '-i', str(normalized_first_path),
            '-i', str(normalized_generated_path),
            '-i', str(normalized_second_path),
            '-filter_complex', '[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[outv][outa]',
            '-map', '[outv]',
            '-map', '[outa]',
            '-c:v', 'libx264', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            str(final_output_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, concat_cmd, 
            capture_output=True, text=True, timeout=240
        )
        
        if result.returncode != 0:
            logger.error(f"Concatenation failed: {result.stderr}")
            raise Exception(f"Failed to concatenate video segments: {result.stderr}")
        
        if not final_output_path.exists():
            raise Exception("Final stitched video file was not created")
        
        logger.info(f"Successfully stitched final video: {final_output_path}")
        
        # Cleanup intermediate files
        for cleanup_path in [normalized_first_path, normalized_generated_path, normalized_second_path]:
            if cleanup_path.exists():
                cleanup_path.unlink()
        
        return str(final_output_path)
        
    except Exception as e:
        logger.error(f"Failed to stitch final video: {e}")
        raise Exception(f"Video stitching failed: {str(e)}")

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
        return client.predict(
            prompt=prompt,
            negative_prompt="worst quality, inconsistent motion, blurry, artifacts",
            input_image_filepath=handle_file(image_path),
            input_video_filepath=None,
            height_ui=960,
            width_ui=544,
            mode="image-to-video",
            duration_ui=3,  # Changed to 5 seconds
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

def _predict_audio(prompt: str):
    """Synchronous function to call the Audio Gradio client for 5-second audio"""
    try:
        logger.info(f"Generating 5-second audio with prompt: {prompt}")
        
        result = audio_client.predict(
            prompt=prompt,
            duration=3,  # 5 seconds to match video
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
