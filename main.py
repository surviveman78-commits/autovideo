import os
import uuid
import shutil
import subprocess
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Header, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import yt_dlp
from dotenv import load_dotenv

# Import pipeline services
from services.transcriber import transcribe_audio
from services.translator import translate_transcript, generate_viral_translation_options
from services.tts import generate_segment_tts_tracks, generate_tts_sync, generate_tts_audio_file, DEFAULT_VOICES
from services.processor import (
    create_srt_file, create_master_audio_track, render_recap_video_gpu,
    extract_audio_from_video, cleanup_recap_temp_files, purge_entire_temp_directory
)
from services.settings import SettingsManager
from services.queue_manager import QueueManager

load_dotenv()

app = FastAPI(
    title="Auto Recap & Video Translation API",
    description="AI-powered Video Translation and Recap generator using Groq, Gemini/OpenAI, TTS, and RTX GPU acceleration.",
    version="3.0.0"
)

@app.on_event("startup")
def startup_event():
    """Starts background queue worker thread automatically on FastAPI server startup."""
    qm = QueueManager.get_instance()
    qm.start_worker()


# Output directory paths
DOWNLOADS_DIR = Path("downloads")
VIDEO_DIR = DOWNLOADS_DIR / "video"
AUDIO_DIR = DOWNLOADS_DIR / "audio"
RECAP_DIR = DOWNLOADS_DIR / "recap"
TEMP_DIR = DOWNLOADS_DIR / "temp"
FONTS_DIR = DOWNLOADS_DIR / "fonts"
VOICES_DIR = DOWNLOADS_DIR / "voices"
STATIC_DIR = Path("static")

# Ensure required directories exist
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
RECAP_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
FONTS_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files and downloaded files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/downloads", StaticFiles(directory="downloads"), name="downloads")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.svg", media_type="image/svg+xml")


class DownloadRequest(BaseModel):
    url: str
    download_type: str = "both"     # Options: "video", "audio", "both"
    resolution: str = "4k"          # Options: "4k", "1080", "720"


class DownloadResponse(BaseModel):
    status: str
    message: str
    video_path: Optional[str] = None
    audio_path: Optional[str] = None


class RecapProcessRequest(BaseModel):
    video_url: Optional[str] = None
    video_file_path: Optional[str] = None
    groq_api_key: Optional[str] = None
    ai_api_key: Optional[str] = None
    ai_provider: str = "gemini"     # "gemini" or "openai"
    target_language: str = "Burmese"
    mode: str = "recap"              # "recap" (🔥 TikTok Story Recap) or "dubbing" (🎙️ 1:1 Video Dubbing)
    voice: str = "f5-myanmar-v2"
    custom_voice_file: Optional[str] = None
    keep_original_audio: bool = False
    original_audio_volume: float = 0.1
    burn_subtitles: bool = True
    sub_font_size: int = 16
    sub_font_color: str = "#FFFFFF"
    sub_font_name: Optional[str] = None
    sub_font_file: Optional[str] = None
    sub_position: str = "bottom"
    sub_margin_v: int = 140
    sub_outline: bool = True
    sub_outline_color: str = "#000000"
    sub_outline_width: int = 2
    sub_shadow: bool = True
    sub_shadow_strength: int = 1
    sub_alignment: str = "center"
    sub_x: Optional[int] = None
    sub_y: Optional[int] = None


class RecapProcessResponse(BaseModel):
    status: str
    message: str
    output_video_path: str
    srt_path: str
    transcript_text: str
    translated_text: str


class HistoryItem(BaseModel):
    name: str
    type: str
    path: str
    time: float


DOWNLOAD_URL_CACHE: Dict[str, str] = {}


def process_video_download(url: str, file_id: str, resolution: str = "4k") -> str:
    """Downloads video in highest available resolution with MP4 merging, reusing cached downloads."""
    clean_url = url.strip()
    if clean_url in DOWNLOAD_URL_CACHE:
        cached_file = DOWNLOAD_URL_CACHE[clean_url]
        if os.path.exists(cached_file):
            print(f"[Video Download] Reusing existing downloaded video for URL: {cached_file}")
            return cached_file

    expected_filepath = str(VIDEO_DIR / f"{file_id}.mp4")
    if os.path.exists(expected_filepath):
        DOWNLOAD_URL_CACHE[clean_url] = expected_filepath
        return expected_filepath

    output_template = str(VIDEO_DIR / f"{file_id}.%(ext)s")

    if resolution == "1080":
        format_str = 'bestvideo[height<=1080]+bestaudio/bestvideo+bestaudio/best'
        sort_opts = ['res:1080', 'res', 'ext:mp4:m4a', 'vcodec:h264']
    elif resolution == "720":
        format_str = 'bestvideo[height<=720]+bestaudio/bestvideo+bestaudio/best'
        sort_opts = ['res:720', 'ext:mp4:m4a', 'vcodec:h264']
    else:
        format_str = 'bestvideo+bestaudio/best'
        sort_opts = ['res:2160', 'res:1440', 'res:1080', 'res', 'ext:mp4:m4a', 'vcodec:h264']

    ydl_opts = {
        'format': format_str,
        'format_sort': sort_opts,
        'merge_output_format': 'mp4',
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
        'nocheckcertificate': True,
        'retries': 5,
        'fragment_retries': 5,
        'socket_timeout': 30,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.tiktok.com/'
        }
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"[Downloader] Primary download failed ({e}). Attempting fallback format...")
        fallback_opts = dict(ydl_opts)
        fallback_opts['format'] = 'bestvideo+bestaudio/best'
        fallback_opts.pop('format_sort', None)
        with yt_dlp.YoutubeDL(fallback_opts) as ydl:
            ydl.download([url])

    final_path = expected_filepath
    if not os.path.exists(expected_filepath):
        matching_files = list(VIDEO_DIR.glob(f"{file_id}.*"))
        if matching_files:
            final_path = matching_files[0].as_posix()
        else:
            raise Exception("Video download completed but output file was not found.")

    DOWNLOAD_URL_CACHE[clean_url] = final_path
    return Path(final_path).as_posix()



@app.get("/", response_class=HTMLResponse)
def get_ui():
    """Serves the main Web UI dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h2>Auto Recap Video API is running. static/index.html not found.</h2>")


@app.get("/api/voices")
def list_voices():
    """Returns available TTS voices."""
    return DEFAULT_VOICES


@app.get("/api/voices/custom")
def list_saved_custom_voices():
    """Returns a list of all saved reference audio files in downloads/voices/ directory."""
    voices = []
    if VOICES_DIR.exists():
        for voice_file in VOICES_DIR.glob("*"):
            if voice_file.is_file() and voice_file.suffix.lower() in [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"]:
                clean_name = voice_file.stem
                if clean_name.startswith("voice_"):
                    clean_name = clean_name[6:]
                # remove optional trailing uuid if present
                if "_" in clean_name and len(clean_name.split("_")[-1]) == 8:
                    clean_name = "_".join(clean_name.split("_")[:-1])
                voices.append({
                    "voice_name": clean_name or voice_file.stem,
                    "voice_file_path": voice_file.as_posix(),
                    "file_name": voice_file.name
                })
    return voices


@app.get("/api/fonts")
def list_user_fonts():
    """Returns a list of all custom font files placed in downloads/fonts/ directory."""
    fonts = []
    if FONTS_DIR.exists():
        for font_file in FONTS_DIR.glob("*"):
            if font_file.is_file() and font_file.suffix.lower() in [".ttf", ".otf", ".woff", ".woff2"]:
                font_name = extract_ttf_font_family_name(str(font_file))
                fonts.append({
                    "font_name": font_name,
                    "font_file_path": font_file.as_posix(),
                    "file_name": font_file.name
                })
    return fonts


@app.post("/api/fonts")
async def upload_custom_font(file: UploadFile = File(...)):
    """Uploads a custom TTF/OTF font file and saves it to downloads/fonts/ directory."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Font file is required.")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".ttf", ".otf", ".woff", ".woff2"]:
        raise HTTPException(status_code=400, detail="Only .ttf, .otf, .woff, and .woff2 font files are supported.")

    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = FONTS_DIR / file.filename

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    font_name = extract_ttf_font_family_name(str(dest_path))

    return {
        "status": "success",
        "message": f"Font '{font_name}' uploaded successfully.",
        "font_name": font_name,
        "font_file_path": dest_path.as_posix(),
        "file_name": file.filename
    }


@app.post("/api/upload", status_code=status.HTTP_200_OK)
async def upload_video_file(file: UploadFile = File(...)):
    """Uploads local video file to server for processing."""
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".mp4"
    saved_path = VIDEO_DIR / f"upload_{file_id}{ext}"
    
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {
        "status": "success",
        "file_name": file.filename,
        "video_path": saved_path.as_posix()
    }


def extract_ttf_font_family_name(font_file_path: str) -> str:
    """Extracts the exact internal Font Family Name (Name ID 1 or 4) from a TTF/OTF file."""
    try:
        from fontTools.ttLib import TTFont
        tt = TTFont(font_file_path)
        for name_id in (1, 4):
            for rec in tt['name'].names:
                if rec.nameID == name_id:
                    try:
                        if rec.platformID == 3:
                            font_name = rec.string.decode('utf-16-be', errors='ignore').strip()
                        else:
                            font_name = rec.string.decode('utf-8', errors='ignore').strip()
                        if font_name:
                            return font_name
                    except Exception:
                        continue
    except Exception as e:
        print(f"Error extracting TTF font family name: {e}")
    return Path(font_file_path).stem


@app.post("/api/upload-font", status_code=status.HTTP_200_OK)
async def upload_font_file(file: UploadFile = File(...)):
    """Uploads custom .ttf or .otf font file to server for video subtitle rendering."""
    ext = Path(file.filename).suffix.lower()
    if ext not in [".ttf", ".otf", ".woff", ".woff2"]:
        raise HTTPException(status_code=400, detail="Only .ttf or .otf font files are supported.")
    
    font_id = str(uuid.uuid4())[:8]
    clean_stem = Path(file.filename).stem
    saved_path = FONTS_DIR / f"{clean_stem}_{font_id}{ext}"
    
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    extracted_font_name = extract_ttf_font_family_name(str(saved_path))

    return {
        "status": "success",
        "font_name": extracted_font_name,
        "font_file_path": saved_path.as_posix()
    }


@app.post("/api/upload-voice", status_code=status.HTTP_200_OK)
async def upload_voice_reference_file(file: UploadFile = File(...)):
    """Uploads custom reference audio file (.mp3, .wav, .m4a, .flac) for VoxCPM2 voice cloning."""
    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format. Please upload MP3, WAV, M4A, or FLAC.")
    
    voice_id = str(uuid.uuid4())[:8]
    clean_stem = Path(file.filename).stem
    saved_path = VOICES_DIR / f"voice_{clean_stem}_{voice_id}{ext}"
    
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {
        "status": "success",
        "voice_name": clean_stem,
        "voice_file_path": saved_path.as_posix()
    }



# Initial startup purge to clear any existing temp artifacts
purge_entire_temp_directory()


@app.post("/api/process-recap", response_model=RecapProcessResponse, status_code=status.HTTP_200_OK)
async def process_recap_video(req: RecapProcessRequest):
    """
    Master pipeline endpoint with Auto Clean System:
    1. Downloads or validates Video Input.
    2. Extracts audio file.
    3. Transcribes using Groq API (Whisper-v3).
    4. Translates/Recaps using Gemini or OpenAI API into target language (Burmese).
    5. Synthesizes voice via edge-tts.
    6. Merges video, audio & burns subtitles using RTX GPU acceleration.
    7. Auto cleans all temporary audio/video files, keeping ONLY final recap output.
    """
    task_id = str(uuid.uuid4())[:8]
    work_dir = TEMP_DIR / task_id
    work_dir.mkdir(parents=True, exist_ok=True)
    video_path = None

    try:
        # Step 1: Input Video
        if req.video_url and req.video_url.strip():
            video_path = process_video_download(req.video_url.strip(), f"dl_{task_id}")
        elif req.video_file_path and os.path.exists(req.video_file_path):
            video_path = req.video_file_path
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please provide a valid video_url or uploaded video_file_path."
            )

        # Step 2: Audio Extraction
        extracted_audio = str(work_dir / "extracted_audio.mp3")
        try:
            extract_audio_from_video(video_path, extracted_audio)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Audio extraction failed: {str(e)}")

        # Step 3: Groq Transcription
        try:
            transcription_result = transcribe_audio(
                extracted_audio,
                api_key=req.groq_api_key
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Groq Transcription error: {str(e)}")

        segments = transcription_result.get("segments", [])
        if not segments:
            raise HTTPException(status_code=400, detail="No speech segments detected in video audio.")

        # Step 4: Gemini / OpenAI Translation
        try:
            translated_segments = translate_transcript(
                segments=segments,
                target_language=req.target_language,
                mode=req.mode,
                provider=req.ai_provider,
                api_key=req.ai_api_key
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"AI Translation error: {str(e)}")

        # Step 5: Text-to-Speech (VoxCPM2 GPU Voice Clone)
        voice_prompt_audio = None
        if req.custom_voice_file and os.path.exists(req.custom_voice_file):
            voice_prompt_audio = req.custom_voice_file
        elif req.voice == "voxcpm2-clone" or "clone" in req.voice:
            voice_prompt_audio = extracted_audio

        tts_dir = work_dir / "tts_segments"
        seg_with_audio = await generate_segment_tts_tracks(
            segments=translated_segments,
            output_dir=tts_dir,
            voice=req.voice,
            prompt_wav_path=voice_prompt_audio
        )

        master_tts_audio = work_dir / "master_tts.mp3"
        create_master_audio_track(seg_with_audio, master_tts_audio)
        full_orig_text = "\n".join([s.get("text", "") for s in segments])
        full_trans_text = "\n".join([s.get("text", "") for s in translated_segments])

        # Step 6: SRT Subtitles & GPU Video Render
        srt_path = RECAP_DIR / f"recap_{task_id}.srt"
        create_srt_file(seg_with_audio, srt_path)

        output_video_file = RECAP_DIR / f"recap_{task_id}.mp4"
        try:
            final_video_path = render_recap_video_gpu(
                video_path=video_path,
                audio_path=str(master_tts_audio),
                srt_path=str(srt_path),
                output_path=str(output_video_file),
                keep_original_audio=req.keep_original_audio,
                original_audio_volume=req.original_audio_volume,
                burn_subtitles=req.burn_subtitles,
                sub_font_size=req.sub_font_size,
                sub_font_color=req.sub_font_color,
                sub_font_name=req.sub_font_name,
                sub_font_file=req.sub_font_file,
                sub_position=req.sub_position,
                sub_margin_v=req.sub_margin_v,
                sub_outline=req.sub_outline,
                sub_outline_color=req.sub_outline_color,
                sub_outline_width=req.sub_outline_width,
                sub_shadow=req.sub_shadow,
                sub_shadow_strength=req.sub_shadow_strength,
                sub_alignment=req.sub_alignment,
                sub_x=req.sub_x,
                sub_y=req.sub_y
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Video rendering error: {str(e)}")

        return RecapProcessResponse(
            status="success",
            message="Video recap & translation completed successfully using GPU NVENC.",
            output_video_path=final_video_path,
            srt_path=str(srt_path.as_posix()),
            transcript_text=full_orig_text,
            translated_text=full_trans_text
        )
    finally:
        # Step 7: Auto Clean System - Remove all temp files & source videos, keeping ONLY output video & srt
        cleanup_recap_temp_files(work_dir, video_path)


@app.post("/api/process-recap-stream")
async def process_recap_stream(req: RecapProcessRequest):
    """
    Streaming pipeline endpoint for real-time, accurate step-by-step progress tracking on the UI.
    Includes smooth ASGI streaming flush and graceful disconnect handling.
    """
    async def event_generator():
        task_id = str(uuid.uuid4())[:8]
        work_dir = TEMP_DIR / task_id
        work_dir.mkdir(parents=True, exist_ok=True)
        video_path = None

        def send_evt(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            # Step 1: Input Video
            yield send_evt({"step": 1, "status": "active", "message": "📥 Downloading & Preparing Video File..."})
            await asyncio.sleep(0.05)
            if req.video_url and req.video_url.strip():
                video_path = process_video_download(req.video_url.strip(), f"dl_{task_id}")
            elif req.video_file_path and os.path.exists(req.video_file_path):
                video_path = req.video_file_path
            else:
                yield send_evt({"type": "error", "detail": "Please provide a valid video_url or uploaded video_file_path."})
                await asyncio.sleep(0.05)
                return
            yield send_evt({"step": 1, "status": "done"})
            await asyncio.sleep(0.05)

            # Step 2: Audio Extraction
            yield send_evt({"step": 2, "status": "active", "message": "🎵 Extracting 16kHz Audio Track..."})
            await asyncio.sleep(0.05)
            extracted_audio = str(work_dir / "extracted_audio.mp3")
            try:
                extract_audio_from_video(video_path, extracted_audio)
            except Exception as e:
                yield send_evt({"type": "error", "detail": f"Audio extraction failed: {str(e)}"})
                await asyncio.sleep(0.05)
                return
            yield send_evt({"step": 2, "status": "done"})
            await asyncio.sleep(0.05)

            # Step 3: Groq Transcription
            yield send_evt({"step": 3, "status": "active", "message": "🎙️ Transcribing with Groq API (Whisper v3)..."})
            await asyncio.sleep(0.05)
            try:
                transcription_result = transcribe_audio(
                    extracted_audio,
                    api_key=req.groq_api_key
                )
            except Exception as e:
                yield send_evt({"type": "error", "detail": f"Groq Transcription error: {str(e)}"})
                await asyncio.sleep(0.05)
                return

            segments = transcription_result.get("segments", [])
            if not segments:
                yield send_evt({"type": "error", "detail": "No speech segments detected in video audio."})
                await asyncio.sleep(0.05)
                return
            yield send_evt({"step": 3, "status": "done"})
            await asyncio.sleep(0.05)

            # Step 4: AI Translation
            yield send_evt({"step": 4, "status": "active", "message": f"🤖 AI Translating to {req.target_language} ({req.mode} mode)..."})
            await asyncio.sleep(0.05)
            try:
                translated_segments = translate_transcript(
                    segments=segments,
                    target_language=req.target_language,
                    mode=req.mode,
                    provider=req.ai_provider,
                    api_key=req.ai_api_key
                )
            except Exception as e:
                yield send_evt({"type": "error", "detail": f"AI Translation error: {str(e)}"})
                await asyncio.sleep(0.05)
                return
            yield send_evt({"step": 4, "status": "done"})
            await asyncio.sleep(0.05)

            # Step 5: Text-to-Speech (VoxCPM2 GPU Voice Clone)
            yield send_evt({"step": 5, "status": "active", "message": "⚡ Synthesizing VoxCPM2 GPU Voice Clone Audio..."})
            await asyncio.sleep(0.05)
            
            voice_prompt_audio = None
            if req.custom_voice_file and os.path.exists(req.custom_voice_file):
                voice_prompt_audio = req.custom_voice_file
            elif req.voice == "voxcpm2-custom":
                voice_files = sorted([f for f in VOICES_DIR.glob("*") if f.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
                if voice_files:
                    voice_prompt_audio = voice_files[0].as_posix()
            elif req.voice == "voxcpm2-clone" or "clone" in req.voice:
                voice_prompt_audio = extracted_audio

            tts_dir = work_dir / "tts_segments"
            seg_with_audio = await generate_segment_tts_tracks(
                segments=translated_segments,
                output_dir=tts_dir,
                voice=req.voice,
                prompt_wav_path=voice_prompt_audio
            )
            master_tts_audio = work_dir / "master_tts.mp3"
            create_master_audio_track(seg_with_audio, master_tts_audio)
            yield send_evt({"step": 5, "status": "done"})
            await asyncio.sleep(0.05)

            # Step 6: SRT Subtitles & GPU Render
            yield send_evt({"step": 6, "status": "active", "message": "🎬 RTX GPU NVENC Video Render & Speed Sync..."})
            await asyncio.sleep(0.05)
            srt_path = RECAP_DIR / f"recap_{task_id}.srt"
            create_srt_file(seg_with_audio, srt_path)

            sub_font_file = req.sub_font_file
            sub_font_name = req.sub_font_name
            if (not sub_font_file or not os.path.exists(sub_font_file)):
                font_files = sorted([f for f in FONTS_DIR.glob("*") if f.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
                if font_files:
                    sub_font_file = font_files[0].as_posix()
                    sub_font_name = extract_ttf_font_family_name(sub_font_file)

            output_video_file = RECAP_DIR / f"recap_{task_id}.mp4"
            try:
                final_video_path = render_recap_video_gpu(
                    video_path=video_path,
                    audio_path=str(master_tts_audio),
                    srt_path=str(srt_path),
                    output_path=str(output_video_file),
                    keep_original_audio=req.keep_original_audio,
                    original_audio_volume=req.original_audio_volume,
                    burn_subtitles=req.burn_subtitles,
                    sub_font_size=req.sub_font_size,
                    sub_font_color=req.sub_font_color,
                    sub_font_name=sub_font_name,
                    sub_font_file=sub_font_file,
                    sub_position=req.sub_position,
                    sub_margin_v=req.sub_margin_v,
                    sub_outline=req.sub_outline,
                    sub_outline_color=req.sub_outline_color,
                    sub_outline_width=req.sub_outline_width,
                    sub_shadow=req.sub_shadow,
                    sub_shadow_strength=req.sub_shadow_strength,
                    sub_alignment=req.sub_alignment,
                    sub_x=req.sub_x,
                    sub_y=req.sub_y
                )
            except Exception as e:
                yield send_evt({"type": "error", "detail": f"Video rendering error: {str(e)}"})
                await asyncio.sleep(0.05)
                return
            yield send_evt({"step": 6, "status": "done"})
            await asyncio.sleep(0.05)

            full_orig_text = "\n".join([s.get("text", "") for s in segments])
            full_trans_text = "\n".join([s.get("text", "") for s in translated_segments])

            yield send_evt({
                "type": "complete",
                "output_video_path": final_video_path,
                "srt_path": str(srt_path.as_posix()),
                "transcript_text": full_orig_text,
                "translated_text": full_trans_text
            })
            await asyncio.sleep(0.05)

        except (asyncio.CancelledError, Exception) as err:
            print(f"Stream exception/disconnect handled: {err}")
        finally:
            cleanup_recap_temp_files(work_dir, video_path)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/history", response_model=List[HistoryItem])
def get_history():
    """Returns a list of all downloaded, uploaded, and generated recap video/audio files."""
    history: List[HistoryItem] = []
    
    for folder, item_type in [(RECAP_DIR, "recap"), (VIDEO_DIR, "video"), (AUDIO_DIR, "audio")]:
        if folder.exists():
            for f in folder.glob("*"):
                if f.is_file():
                    try:
                        mtime = os.path.getmtime(f)
                    except OSError:
                        mtime = 0.0
                    history.append(HistoryItem(
                        name=f.name,
                        type=item_type,
                        path=f.as_posix(),
                        time=mtime
                    ))

    history.sort(key=lambda x: x.time, reverse=True)
    return history


# Step-by-Step Interactive Wizard Models & Endpoints

class StepVideoPreviewRequest(BaseModel):
    video_url: Optional[str] = None
    video_file_path: Optional[str] = None

class StepTranscribeRequest(BaseModel):
    video_path: str
    groq_api_key: Optional[str] = None

class StepTranslateOptionsRequest(BaseModel):
    segments: List[Dict[str, Any]]
    target_language: str = "Burmese"
    translation_style: str = "documentary" # "comedy", "sad", "emotional", "documentary"
    ai_provider: str = "gemini"
    ai_api_key: Optional[str] = None

class StepVoicePreviewRequest(BaseModel):
    voice: str = "voxcpm2-clone"
    engine: str = "voxcpm" # "voxcpm" or "edge"
    custom_voice_file: Optional[str] = None
    prompt_text: Optional[str] = None
    text: str = "မင်္ဂလာပါ၊ ဒါကတော့ အသံ စမ်းသပ်မှု ဖြစ်ပါတယ်။"

class StepRenderRequest(BaseModel):
    video_path: str
    segments: List[Dict[str, Any]]
    voice: str = "voxcpm2-clone"
    engine: str = "voxcpm"
    custom_voice_file: Optional[str] = None
    prompt_text: Optional[str] = None
    keep_original_audio: bool = False
    original_audio_volume: float = 0.1
    burn_subtitles: bool = True
    sub_font_size: int = 16
    sub_font_color: str = "#FFFFFF"
    sub_font_name: Optional[str] = None
    sub_font_file: Optional[str] = None
    sub_position: str = "bottom"
    sub_margin_v: int = 25


@app.post("/api/step/preview-video")
async def step_preview_video(req: StepVideoPreviewRequest):
    """Step 1: Downloads/prepares video file and returns preview URL."""
    task_id = str(uuid.uuid4())[:8]
    if req.video_url and req.video_url.strip():
        video_path = process_video_download(req.video_url.strip(), f"preview_{task_id}")
    elif req.video_file_path and os.path.exists(req.video_file_path):
        video_path = req.video_file_path
    else:
        raise HTTPException(status_code=400, detail="Please provide a valid video URL or video file path.")

    rel_path = Path(video_path).as_posix()
    return {
        "status": "success",
        "video_path": rel_path,
        "preview_url": f"/{rel_path}",
        "file_name": Path(video_path).name
    }


@app.post("/api/step/transcribe")
async def step_transcribe_audio(req: StepTranscribeRequest):
    """Step 2a: Extracts audio and transcribes using Groq Whisper v3."""
    if not os.path.exists(req.video_path):
        raise HTTPException(status_code=404, detail="Video file not found.")

    task_id = str(uuid.uuid4())[:8]
    temp_audio = str(TEMP_DIR / f"temp_audio_{task_id}.mp3")
    try:
        extract_audio_from_video(req.video_path, temp_audio)
        result = transcribe_audio(temp_audio, api_key=req.groq_api_key)
        segments = result.get("segments", [])
        if not segments:
            raise HTTPException(status_code=400, detail="No speech detected in video audio.")
        
        full_text = "\n".join([s.get("text", "") for s in segments])
        return {
            "status": "success",
            "segments": segments,
            "transcript_text": full_text
        }
    finally:
        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except Exception:
                pass


@app.post("/api/step/translate-options")
async def step_translate_options(req: StepTranslateOptionsRequest):
    """Step 2b: Generates 3 translation candidates with Viral Scores & requested style."""
    if not req.segments:
        raise HTTPException(status_code=400, detail="Segments list cannot be empty.")

    try:
        options = generate_viral_translation_options(
            segments=req.segments,
            target_language=req.target_language,
            translation_style=req.translation_style,
            provider=req.ai_provider,
            api_key=req.ai_api_key
        )
        return {
            "status": "success",
            "options": options
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Translation error: {str(e)}")


VOICE_SAMPLE_CACHE_DIR = TEMP_DIR / "sample_cache"
VOICE_SAMPLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/api/step/preview-voice")
async def step_preview_voice(req: StepVoicePreviewRequest):
    """Step 3: Serves cached voice audio sample preview for testing selected voice without re-cloning every time."""
    clean_voice_id = req.voice.replace(":", "_").replace("/", "_").replace("\\", "_")
    sample_filename = f"sample_{req.engine}_{clean_voice_id}.mp3" if req.engine == "edge" else f"sample_{req.engine}_{clean_voice_id}.wav"
    sample_file = VOICE_SAMPLE_CACHE_DIR / sample_filename

    # If static cached sample already exists, return instantly!
    if sample_file.exists() and sample_file.stat().st_size > 0:
        print(f"[Voice Preview] Returning existing static audio sample for voice: {req.voice}")
        return {
            "status": "success",
            "audio_url": f"/{sample_file.as_posix()}"
        }

    prompt_wav = None
    if req.custom_voice_file and os.path.exists(req.custom_voice_file):
        prompt_wav = req.custom_voice_file

    try:
        audio_path = await generate_tts_audio_file(
            text=req.text or "မင်္ဂလာပါ၊ ဒါကတော့ အသံ စမ်းသပ်မှု ဖြစ်ပါတယ်။",
            output_path=str(sample_file),
            voice=req.voice,
            prompt_wav_path=prompt_wav,
            prompt_text=req.prompt_text,
            engine=req.engine
        )
        return {
            "status": "success",
            "audio_url": f"/{Path(audio_path).as_posix()}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice preview failed: {str(e)}")


@app.post("/api/step/render")
async def step_render_video(req: StepRenderRequest):
    """Step 5: Final render using selected segments, voice, engine, font, position, and GPU acceleration."""
    if not os.path.exists(req.video_path):
        raise HTTPException(status_code=404, detail="Input video file not found.")

    task_id = str(uuid.uuid4())[:8]
    work_dir = TEMP_DIR / task_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        extracted_audio = str(work_dir / "extracted_audio.mp3")
        extract_audio_from_video(req.video_path, extracted_audio)

        voice_prompt_audio = None
        if req.custom_voice_file and os.path.exists(req.custom_voice_file):
            voice_prompt_audio = req.custom_voice_file
        elif req.voice == "voxcpm2-clone" or "clone" in req.voice:
            voice_prompt_audio = extracted_audio

        tts_dir = work_dir / "tts_segments"
        seg_with_audio = await generate_segment_tts_tracks(
            segments=req.segments,
            output_dir=tts_dir,
            voice=req.voice,
            prompt_wav_path=voice_prompt_audio,
            prompt_text=req.prompt_text,
            engine=req.engine
        )

        master_tts_audio = work_dir / "master_tts.mp3"
        create_master_audio_track(seg_with_audio, master_tts_audio)

        srt_path = RECAP_DIR / f"recap_{task_id}.srt"
        create_srt_file(seg_with_audio, srt_path)

        output_video_file = RECAP_DIR / f"recap_{task_id}.mp4"

        sub_font_file = req.sub_font_file
        sub_font_name = req.sub_font_name
        if (not sub_font_file or not os.path.exists(sub_font_file)):
            font_files = sorted([f for f in FONTS_DIR.glob("*") if f.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
            if font_files:
                sub_font_file = font_files[0].as_posix()
                sub_font_name = extract_ttf_font_family_name(sub_font_file)

        final_video_path = render_recap_video_gpu(
            video_path=req.video_path,
            audio_path=str(master_tts_audio),
            srt_path=str(srt_path),
            output_path=str(output_video_file),
            keep_original_audio=req.keep_original_audio,
            original_audio_volume=req.original_audio_volume,
            burn_subtitles=req.burn_subtitles,
            sub_font_size=req.sub_font_size,
            sub_font_color=req.sub_font_color,
            sub_font_name=sub_font_name,
            sub_font_file=sub_font_file,
            sub_position=req.sub_position,
            sub_margin_v=req.sub_margin_v,
            sub_outline=req.sub_outline,
            sub_outline_color=req.sub_outline_color,
            sub_outline_width=req.sub_outline_width,
            sub_shadow=req.sub_shadow,
            sub_shadow_strength=req.sub_shadow_strength,
            sub_alignment=req.sub_alignment,
            sub_x=req.sub_x,
            sub_y=req.sub_y
        )

        return {
            "status": "success",
            "message": "Video rendered successfully with GPU acceleration.",
            "output_video_path": final_video_path,
            "output_video_url": f"/{Path(final_video_path).as_posix()}",
            "srt_path": str(srt_path.as_posix()),
            "srt_url": f"/{Path(srt_path).as_posix()}"
        }
    finally:
        cleanup_recap_temp_files(work_dir, req.video_path)


# ==============================================================================
# Persistent Settings API Endpoints
# ==============================================================================

@app.get("/api/settings")
def get_settings():
    """Returns saved settings with masked secret API keys."""
    return SettingsManager.get_instance().get_masked_settings_dict()


@app.post("/api/settings")
def update_settings(payload: Dict[str, Any]):
    """Updates and saves settings permanently to downloads/settings.json."""
    try:
        updated = SettingsManager.get_instance().update_settings(payload)
        return {
            "status": "success",
            "message": "Settings saved successfully.",
            "settings": SettingsManager.get_instance().get_masked_settings_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to update settings: {str(e)}")


from services.groq_key_manager import GroqKeyManager
from services.settings import is_masked


@app.get("/api/settings/groq-keys")
def get_groq_keys():
    """Returns safe UI status info for all configured Groq API keys."""
    return GroqKeyManager.get_instance().get_keys_status_for_ui()


@app.post("/api/settings/groq-keys")
def save_groq_keys(keys_payload: List[Dict[str, Any]]):
    """Saves or updates configured Groq API keys list."""
    try:
        updated = SettingsManager.get_instance().update_settings({"groq_api_keys": keys_payload})
        return {
            "status": "success",
            "message": "Groq API keys saved successfully.",
            "keys": GroqKeyManager.get_instance().get_keys_status_for_ui()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save Groq keys: {str(e)}")


@app.delete("/api/settings/groq-keys/{key_id}")
def delete_groq_key(key_id: str):
    """Removes a configured Groq API key by ID."""
    settings = SettingsManager.get_instance().load_settings()
    current_keys = [k.model_dump() for k in settings.groq_api_keys]
    new_keys = [k for k in current_keys if k.get("id") != key_id]
    SettingsManager.get_instance().update_settings({"groq_api_keys": new_keys})
    return {
        "status": "success",
        "message": f"Groq key '{key_id}' removed.",
        "keys": GroqKeyManager.get_instance().get_keys_status_for_ui()
    }


@app.post("/api/settings/groq-keys/{key_id}/test")
def test_specific_groq_key(key_id: str, payload: Optional[Dict[str, Any]] = None):
    """Tests a specific configured or unsaved Groq API key using lightweight models.list()."""
    test_key = None
    if payload and payload.get("key"):
        raw = payload.get("key").strip()
        if raw and not is_masked(raw):
            test_key = raw

    if not test_key:
        settings = SettingsManager.get_instance().load_settings()
        entry = next((k for k in settings.groq_api_keys if k.id == key_id), None)
        if entry and entry.key and not is_masked(entry.key):
            test_key = entry.key

    if not test_key or is_masked(test_key):
        raise HTTPException(
            status_code=400,
            detail="Groq API key is not configured. Open Settings → AI & Prompts and add a valid Groq API key."
        )

    try:
        from groq import Groq
        client = Groq(api_key=test_key)
        _ = client.models.list()
        return {"status": "success", "message": "✓ Groq API connection successful."}
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "authentication" in err_msg.lower() or "Invalid API Key" in err_msg:
            raise HTTPException(status_code=401, detail="✕ Groq API authentication failed. Please verify your Groq API Key in Settings → AI & Prompts.")
        elif "429" in err_msg or "rate limit" in err_msg.lower():
            raise HTTPException(status_code=429, detail="⚠ Groq API key is currently rate-limited.")
        else:
            raise HTTPException(status_code=400, detail=f"✕ Groq API test failed: {err_msg}")


@app.post("/api/settings/test-groq")
def test_groq_api_connection(payload: Optional[Dict[str, Any]] = None):
    """Tests connection to Groq API using provided unsaved key or active configured key."""
    test_key = None
    if payload and payload.get("groq_api_key"):
        raw = payload.get("groq_api_key").strip()
        if raw and not is_masked(raw):
            test_key = raw

    if not test_key:
        active_info = GroqKeyManager.get_instance().get_active_key()
        if active_info:
            test_key = active_info[1]

    if not test_key or is_masked(test_key):
        test_key = os.getenv("GROQ_API_KEY")

    if not test_key:
        raise HTTPException(
            status_code=400,
            detail="Groq API key is not configured. Open Settings → AI & Prompts and add a valid Groq API key."
        )

    try:
        from groq import Groq
        client = Groq(api_key=test_key)
        _ = client.models.list()
        return {"status": "success", "message": "✓ Groq API connection successful."}
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "authentication" in err_msg.lower() or "Invalid API Key" in err_msg:
            raise HTTPException(status_code=401, detail="✕ Groq API authentication failed. Please verify your Groq API Key in Settings → AI & Prompts.")
        elif "429" in err_msg or "rate limit" in err_msg.lower():
            raise HTTPException(status_code=429, detail="⚠ Groq API key is currently rate-limited.")
        else:
            raise HTTPException(status_code=400, detail=f"✕ Groq API test failed: {err_msg}")



# ==============================================================================
# Voice & TTS Metadata API Endpoints
# ==============================================================================

@app.get("/api/voices/edge")
def get_edge_voices():
    """Returns available Microsoft Edge TTS voices metadata."""
    from services.tts import get_edge_tts_voices
    return get_edge_tts_voices()


VOICES_METADATA_FILE = VOICES_DIR / "metadata.json"


def load_custom_voices_metadata() -> List[Dict[str, Any]]:
    if VOICES_METADATA_FILE.exists():
        try:
            with open(VOICES_METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_custom_voices_metadata(voices_list: List[Dict[str, Any]]):
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    with open(VOICES_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(voices_list, f, ensure_ascii=False, indent=2)


@app.get("/api/voices/custom")
def list_custom_voice_references():
    """Returns list of saved VoxCPM custom voice references."""
    return load_custom_voices_metadata()


@app.post("/api/voices/custom")
async def add_custom_voice_reference(
    name: Optional[str] = Form(None),
    voice_name: Optional[str] = Form(None),
    reference_text: Optional[str] = Form(""),
    file: UploadFile = File(...)
):
    """Uploads a new VoxCPM voice reference audio file + spoken reference text and saves metadata."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Audio reference file is required.")
    
    clean_name = (voice_name or name or "Custom Voice").strip()
    clean_text = (reference_text or "").strip()
    voice_id = f"vref_{uuid.uuid4().hex[:8]}"
    ext = Path(file.filename).suffix or ".wav"
    dest_path = VOICES_DIR / f"{voice_id}{ext}"

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    voices = load_custom_voices_metadata()
    new_entry = {
        "id": voice_id,
        "name": clean_name,
        "audio_path": dest_path.as_posix(),
        "text": clean_text,
        "file_name": file.filename
    }
    voices.append(new_entry)
    save_custom_voices_metadata(voices)

    return {
        "status": "success",
        "message": "Custom voice reference saved successfully.",
        "id": voice_id,
        "voice": new_entry
    }


# ==============================================================================
# Persistent Job Queue API Endpoints
# ==============================================================================

class AddQueueRequest(BaseModel):
    url: str
    mode: Optional[str] = "recap"
    client_id: Optional[str] = None


@app.post("/api/queue/add")
def add_video_to_queue(req: AddQueueRequest, request: Request):
    """Adds a new video URL to the automated processing queue tagged with mode and client session ID."""
    qm = QueueManager.get_instance()
    client_id = req.client_id or request.headers.get("x-client-id")
    try:
        job = qm.add_job(req.url, mode=req.mode, client_id=client_id)
        return {
            "status": "success",
            "message": "Video added to background queue.",
            "job": job.model_dump()
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add video to queue: {str(e)}")


@app.get("/api/queue")
def get_queue_jobs(request: Request, client_id: Optional[str] = Query(None)):
    """Returns all queued, processing, completed, and failed jobs filtered by client session ID."""
    header_client_id = request.headers.get("x-client-id")
    effective_client_id = client_id or header_client_id
    qm = QueueManager.get_instance()
    jobs = qm.get_all_jobs(client_id=effective_client_id)
    response = JSONResponse(content=[job.model_dump() for job in jobs])
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.post("/api/queue/cancel/{job_id}")
def cancel_queue_job(job_id: str):
    """Cancels a processing or queued video job."""
    qm = QueueManager.get_instance()
    success = qm.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job not found or already finished.")
    return {"status": "success", "message": f"Job {job_id} cancellation requested."}


@app.post("/api/queue/retry/{job_id}")
def retry_queue_job(job_id: str):
    """Retries a failed or cancelled video job."""
    qm = QueueManager.get_instance()
    success = qm.retry_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job not found or not in failed/cancelled state.")
    return {"status": "success", "message": f"Job {job_id} queued for retry."}


@app.delete("/api/queue/completed")
def clear_completed_jobs():
    """Clears completed and cancelled jobs from the queue view."""
    qm = QueueManager.get_instance()
    qm.clear_completed()
    return {"status": "success", "message": "Completed jobs cleared."}


# ==============================================================================
# Secure Output File Download Endpoints
# ==============================================================================

@app.get("/api/jobs/{job_id}/download/mp4")
def download_job_mp4(job_id: str):
    """Safely serves rendered MP4 output video for a completed job."""
    qm = QueueManager.get_instance()
    job = qm.get_job(job_id)
    if not job or not job.output_video_path or not os.path.exists(job.output_video_path):
        raise HTTPException(status_code=404, detail="Rendered MP4 file not found for this job.")
    return FileResponse(path=job.output_video_path, media_type="video/mp4", filename=f"recap_{job_id}.mp4")


@app.get("/api/jobs/{job_id}/download/srt")
def download_job_srt(job_id: str):
    """Safely serves generated SRT subtitles for a completed job."""
    qm = QueueManager.get_instance()
    job = qm.get_job(job_id)
    if not job or not job.output_srt_path or not os.path.exists(job.output_srt_path):
        raise HTTPException(status_code=404, detail="SRT subtitle file not found for this job.")
    return FileResponse(path=job.output_srt_path, media_type="application/x-subrip", filename=f"recap_{job_id}.srt")


@app.get("/api/jobs/{job_id}/stream")
def stream_job_video(job_id: str):
    """Streams completed MP4 output video for inline HTML5 video player with Range Request support."""
    qm = QueueManager.get_instance()
    job = qm.get_job(job_id)
    if not job or not job.output_video_path or not os.path.exists(job.output_video_path):
        raise HTTPException(status_code=404, detail="Rendered MP4 file not found for this job.")
    return FileResponse(
        path=job.output_video_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"}
    )


@app.get("/api/outputs/{filename}")
def stream_output_file(filename: str):
    """Safely serves output video files from downloads/recap/ directory."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename parameter.")
    file_path = RECAP_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found.")
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"}
    )


@app.delete("/api/queue/job/{job_id}")
def delete_single_job(job_id: str):
    """Removes a single job record from queue history."""
    qm = QueueManager.get_instance()
    success = qm.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"status": "success", "message": f"Job {job_id} removed from history."}



# ==============================================================================
# Live Real-Time SSE Stream Endpoint
# ==============================================================================

@app.get("/api/queue/stream")
async def queue_event_stream():
    """Server-Sent Events stream delivering real-time progress and job state updates."""
    async def event_generator():
        q = asyncio.Queue()
        qm = QueueManager.get_instance()
        qm.listeners.append(q)
        try:
            initial_jobs = [j.model_dump() for j in qm.get_all_jobs()]
            yield f"data: {json.dumps({'type': 'init', 'jobs': initial_jobs}, ensure_ascii=False)}\n\n"

            while True:
                data = await q.get()
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            if q in qm.listeners:
                qm.listeners.remove(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

