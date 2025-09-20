async def _process_video_standardized(video_path: str) -> tuple:
    """Extract middle frame and split video, with middle frame as static image"""
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
            '-vf', 'scale=-1:1080:flags=lanczos,unsharp=5:5:1.0:5:5:0.0',
            '-f', 'image2',  # Specify image format
            '-pix_fmt', 'rgb24',  # Use RGB pixel format
            str(middle_frame_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd_frame, capture_output=True, text=True, timeout=60
        )
        
        if result.returncode != 0:
            raise Exception(f"Failed to extract middle frame: {result.stderr}")
        
        # Extract and standardize first part (start to middle) WITH ORIGINAL AUDIO
        first_part_path = temp_dir / f"{process_id}_first_part.mp4"
        cmd_first = [
            'ffmpeg', '-y', '-i', video_path,
            '-t', str(middle_time),
            '-vf', f'scale={STANDARD_WIDTH}:{STANDARD_HEIGHT}:force_original_aspect_ratio=decrease,pad={STANDARD_WIDTH}:{STANDARD_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black',
            '-r', str(STANDARD_FPS),  # Standardize FPS
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-ac', '2',  # Keep original audio but standardize format
            '-movflags', '+faststart',
            '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
            str(first_part_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd_first, capture_output=True, text=True, timeout=120
        )
        
        if result.returncode != 0:
            raise Exception(f"Failed to extract first part: {result.stderr}")
        
        # Create middle part as STATIC IMAGE VIDEO (brief duration, maybe 1-2 seconds)
        middle_static_duration = 1.0  # 1 second static image
        middle_part_path = temp_dir / f"{process_id}_middle_static.mp4"
        
        # Create static video from middle frame with silence
        cmd_static = [
            'ffmpeg', '-y',
            '-loop', '1', '-i', str(middle_frame_path),  # Loop the image
            '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000',  # Generate silence
            '-vf', f'scale={STANDARD_WIDTH}:{STANDARD_HEIGHT}:force_original_aspect_ratio=decrease,pad={STANDARD_WIDTH}:{STANDARD_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black',
            '-r', str(STANDARD_FPS),
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-ac', '2',
            '-t', str(middle_static_duration),  # Duration of static image
            '-movflags', '+faststart',
            str(middle_part_path)
        ]
        
        result = await asyncio.to_thread(
            subprocess.run, cmd_static, capture_output=True, text=True, timeout=60
        )
        
        if result.returncode != 0:
            raise Exception(f"Failed to create middle static part: {result.stderr}")
        
        # Extract original audio from middle frame to end for AI video mixing
        remaining_audio_start = middle_time
        remaining_audio_path = temp_dir / f"{process_id}_remaining_audio.wav"
        
        if remaining_audio_start < duration:
            cmd_remaining_audio = [
                'ffmpeg', '-y', '-i', video_path,
                '-ss', str(remaining_audio_start),
                '-vn',  # No video
                '-ar', '48000', '-ac', '2', '-c:a', 'pcm_s16le',
                '-t', str(ai_duration),  # Only 5 seconds for AI video duration
                str(remaining_audio_path)
            ]
            
            result = await asyncio.to_thread(
                subprocess.run, cmd_remaining_audio, capture_output=True, text=True, timeout=60
            )
            
            if result.returncode != 0:
                logger.warning(f"Failed to extract remaining audio, will create silence: {result.stderr}")
                # Create silence as fallback
                cmd_silence = [
                    'ffmpeg', '-y', '-f', 'lavfi', 
                    '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000',
                    '-t', str(ai_duration), '-c:a', 'pcm_s16le',
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
                'ffmpeg', '-y', '-f', 'lavfi', 
                '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000',
                '-t', str(ai_duration), '-c:a', 'pcm_s16le',
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
        logger.info(f"- Middle static part: {middle_part_path}")
        logger.info(f"- Remaining audio: {remaining_audio_path}")
        
        return str(middle_frame_path), str(first_part_path), str(middle_part_path), str(remaining_audio_path)
        
    except Exception as e:
        logger.error(f"Failed to process video: {e}")
        raise Exception(f"Video processing failed: {str(e)}")

async def _merge_ai_video_with_mixed_audio(ai_video_path: str, ai_audio_path: str, original_audio_path: str) -> str:
    """Merge AI video with MIXED audio (AI audio + original remaining audio)"""
    try:
        import subprocess
        
        temp_dir = Path("/tmp")
        output_id = str(uuid.uuid4())
        merged_path = temp_dir / f"{output_id}_merged.mp4"
        mixed_audio_path = temp_dir / f"{output_id}_mixed_audio.wav"
        
        logger.info(f"Mixing AI audio {ai_audio_path} with original remaining audio {original_audio_path}")
        
        # First repair the AI audio to prevent corruption issues
        repaired_ai_audio_path = await _repair_audio(ai_audio_path) if ai_audio_path else None
        
        if repaired_ai_audio_path and Path(original_audio_path).exists():
            # Mix AI audio with original remaining audio (60% AI, 40% original)
            cmd_mix = [
                'ffmpeg', '-y',
                '-i', repaired_ai_audio_path,  # AI audio
                '-i', original_audio_path,     # Original remaining audio
                '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=shortest:weights=0.6 0.4',  # AI audio slightly louder
                '-ar', '48000', '-ac', '2', '-c:a', 'pcm_s16le',
                '-t', '5.0',  # Exact 5 seconds
                str(mixed_audio_path)
            ]
            
            result = await asyncio.to_thread(
                subprocess.run, cmd_mix, capture_output=True, text=True, timeout=60
            )
            
            if result.returncode != 0:
                logger.warning(f"Audio mixing failed, using original audio only: {result.stderr}")
                # If mixing fails, use original audio only
                mixed_audio_path = original_audio_path
            else:
                logger.info(f"Successfully mixed AI and original remaining audio: {mixed_audio_path}")
        elif Path(original_audio_path).exists():
            # Use original remaining audio only if AI audio failed
            logger.info("Using original remaining audio only (AI audio unavailable)")
            mixed_audio_path = original_audio_path
        elif repaired_ai_audio_path:
            # Use AI audio only if original audio is unavailable
            logger.info("Using AI audio only (original remaining audio unavailable)")
            mixed_audio_path = repaired_ai_audio_path
        else:
            # Create silence if both failed
            logger.warning("Both audio sources failed, creating silence")
            cmd_silence = [
                'ffmpeg', '-y', '-f', 'lavfi', 
                '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000',
                '-t', '5.0', '-c:a', 'pcm_s16le',
                str(mixed_audio_path)
            ]
            
            result = await asyncio.to_thread(
                subprocess.run, cmd_silence, capture_output=True, text=True, timeout=30
            )
            
            if result.returncode != 0:
                raise Exception(f"Failed to create silence audio: {result.stderr}")
        
        # Now merge AI video with the mixed/selected audio
        cmd = [
            'ffmpeg', '-y',
            '-i', ai_video_path,  # AI Video input
            '-i', str(mixed_audio_path),  # Mixed audio input
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
        
        # Clean up temporary files
        try:
            if repaired_ai_audio_path and repaired_ai_audio_path != mixed_audio_path:
                Path(repaired_ai_audio_path).unlink()
            if str(mixed_audio_path) != original_audio_path and Path(mixed_audio_path).exists():
                Path(mixed_audio_path).unlink()
        except:
            pass
        
        if result.returncode != 0:
            logger.error(f"FFmpeg merge failed: {result.stderr}")
            # If merge still fails, create video without audio
            logger.warning("Creating AI video without audio due to merge failure")
            cmd_no_audio = [
                'ffmpeg', '-y',
                '-i', ai_video_path,
                '-c:v', 'copy',
                '-an',  # No audio
                '-t', '5.0',
                str(merged_path)
            ]
            
            result = await asyncio.to_thread(
                subprocess.run, cmd_no_audio, capture_output=True, text=True, timeout=60
            )
            
            if result.returncode != 0:
                raise Exception(f"Failed to create AI video even without audio: {result.stderr}")
        
        logger.info(f"Successfully merged AI video with mixed audio: {merged_path}")
        return str(merged_path)
        
    except Exception as e:
        logger.error(f"Failed to merge AI video with mixed audio: {e}")
        raise Exception(f"AI Video-audio merge failed: {str(e)}")

# Updated main function changes needed:
# In the generate_video function, update these lines:

        # Process video with standardized parameters
        logger.info("Processing video: extracting middle frame, first part, and remaining audio...")
        middle_frame_path, first_part_path, middle_static_path, remaining_audio_path = await _process_video_standardized(str(temp_video_path))
        temp_files.extend([middle_frame_path, first_part_path, middle_static_path, remaining_audio_path])

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

        ai_video_path = ai_video_result[0].get("video") if isinstance(ai_video_result[0], dict) else ai_video_result[0]
        seed_used = ai_video_result[1] if len(ai_video_result) > 1 else "unknown"

        logger.info(f"AI Video generated locally: {ai_video_path}")
        logger.info(f"AI Audio result: {ai_audio_result}")

        # Standardize AI video to match our requirements
        ai_video_standardized = await _standardize_video(ai_video_path, is_ai_video=True)
        temp_files.append(ai_video_standardized)

        # Merge AI video with MIXED audio (AI audio + original remaining audio)
        if ai_audio_result:
            ai_merged_path = await _merge_ai_video_with_mixed_audio(ai_video_standardized, ai_audio_result, remaining_audio_path)
            temp_files.append(ai_merged_path)
            logger.info(f"AI video merged with mixed audio (AI + original remaining): {ai_merged_path}")
        else:
            # Use AI video with original remaining audio only if AI audio generation failed
            ai_merged_path = await _merge_ai_video_with_mixed_audio(ai_video_standardized, None, remaining_audio_path)
            temp_files.append(ai_merged_path)
            logger.info(f"AI video merged with original remaining audio only: {ai_merged_path}")

        # Create final video by concatenating: first_part + middle_static + ai_merged
        final_video_path = await _concatenate_videos_standardized([first_part_path, middle_static_path, ai_merged_path])
        temp_files.append(final_video_path)
        logger.info(f"Final video created: {final_video_path}")
