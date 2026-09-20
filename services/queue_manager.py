import os
import time
import uuid
import json
import shutil
import asyncio
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from services.settings import SettingsManager, Settings, is_masked

QUEUE_FILE = Path("downloads") / "queue.json"
JOBS_DIR = Path("downloads") / "jobs"
RECAP_DIR = Path("downloads") / "recap"

QUEUE_LOCK = threading.RLock()


class Job(BaseModel):
    job_id: str
    url: str
    title: str = "Video Job"
    status: str = "QUEUED"  # QUEUED, DOWNLOADING, EXTRACTING_AUDIO, TRANSCRIBING, TRANSLATING, GENERATING_VOICE, GENERATING_SUBTITLES, RENDERING, COMPLETED, FAILED, CANCELLED
    progress: int = 0
    stage_message: str = "Queued in processing line"
    error_message: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    output_video_path: Optional[str] = None
    output_srt_path: Optional[str] = None
    settings_snapshot: Optional[Dict[str, Any]] = None


def normalize_url(url: str) -> str:
    """Normalizes video URL for duplicate detection."""
    clean = url.strip().lower()
    if clean.endswith("/"):
        clean = clean[:-1]
    return clean


class QueueManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.listeners: List[asyncio.Queue] = []
        self.cancel_requested: set = set()
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = False

        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        RECAP_DIR.mkdir(parents=True, exist_ok=True)

        self.load_queue()
        self.recover_interrupted_jobs()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def load_queue(self):
        with QUEUE_LOCK:
            if QUEUE_FILE.exists():
                try:
                    with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.jobs = {item["job_id"]: Job(**item) for item in data}
                except Exception as e:
                    print(f"[Queue] Error loading {QUEUE_FILE}: {e}")
                    self.jobs = {}
            else:
                self.save_queue()

    def save_queue(self):
        with QUEUE_LOCK:
            QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = [job.model_dump() for job in self.jobs.values()]
                with open(QUEUE_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[Queue] Error saving queue: {e}")

    def recover_interrupted_jobs(self):
        """On startup, safely recovers any active jobs that were interrupted by crash/restart."""
        active_states = {"DOWNLOADING", "EXTRACTING_AUDIO", "TRANSCRIBING", "TRANSLATING", "GENERATING_VOICE", "GENERATING_SUBTITLES", "RENDERING"}
        recovered_count = 0
        for job in self.jobs.values():
            if job.status in active_states:
                print(f"[Queue Recovery] Interrupted job detected ({job.job_id}). Recovering state to QUEUED.")
                job.status = "QUEUED"
                job.stage_message = "Recovered after restart - Queued"
                job.progress = 0
                recovered_count += 1
        if recovered_count > 0:
            self.save_queue()

    def get_all_jobs(self) -> List[Job]:
        with QUEUE_LOCK:
            sorted_jobs = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
            return sorted_jobs

    def get_job(self, job_id: str) -> Optional[Job]:
        with QUEUE_LOCK:
            return self.jobs.get(job_id)

    def is_duplicate_url(self, url: str) -> bool:
        """Checks if URL is currently QUEUED or actively PROCESSING."""
        target = normalize_url(url)
        with QUEUE_LOCK:
            for job in self.jobs.values():
                if job.status in {"QUEUED", "DOWNLOADING", "EXTRACTING_AUDIO", "TRANSCRIBING", "TRANSLATING", "GENERATING_VOICE", "GENERATING_SUBTITLES", "RENDERING"}:
                    if normalize_url(job.url) == target:
                        return True
        return False

    def add_job(self, url: str) -> Job:
        clean_url = url.strip()
        if not clean_url:
            raise ValueError("Video URL cannot be empty.")

        if self.is_duplicate_url(clean_url):
            raise ValueError("This Video URL is already queued or currently processing.")

        job_id = f"{time.strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
        settings = SettingsManager.get_instance().load_settings()

        job = Job(
            job_id=job_id,
            url=clean_url,
            title=f"Video ({job_id})",
            status="QUEUED",
            progress=0,
            stage_message="Queued in processing line",
            max_retries=settings.max_retries,
            settings_snapshot=settings.model_dump()
        )

        with QUEUE_LOCK:
            self.jobs[job_id] = job
            self.save_queue()

        self.broadcast_event({
            "type": "job_added",
            "job_id": job_id,
            "status": "QUEUED",
            "progress": 0,
            "stage_message": "Queued in processing line"
        })

        return job

    def cancel_job(self, job_id: str) -> bool:
        with QUEUE_LOCK:
            job = self.jobs.get(job_id)
            if not job:
                return False

            if job.status in {"COMPLETED", "FAILED", "CANCELLED"}:
                return False

            self.cancel_requested.add(job_id)

            if job.status == "QUEUED":
                job.status = "CANCELLED"
                job.stage_message = "Cancelled by user"
                self.save_queue()
                self.broadcast_event({
                    "type": "job_update",
                    "job_id": job_id,
                    "status": "CANCELLED",
                    "progress": 0,
                    "stage_message": "Cancelled by user"
                })
                return True

            job.stage_message = "Cancelling processing..."
            self.save_queue()
            return True

    def retry_job(self, job_id: str) -> bool:
        with QUEUE_LOCK:
            job = self.jobs.get(job_id)
            if not job:
                return False

            if job.status not in {"FAILED", "CANCELLED"}:
                return False

            job.status = "QUEUED"
            job.progress = 0
            job.stage_message = "Retrying job..."
            job.error_message = None
            job.retry_count = 0
            if job_id in self.cancel_requested:
                self.cancel_requested.remove(job_id)

            self.save_queue()

        self.broadcast_event({
            "type": "job_update",
            "job_id": job_id,
            "status": "QUEUED",
            "progress": 0,
            "stage_message": "Retrying job..."
        })
        return True

    def clear_completed(self):
        with QUEUE_LOCK:
            to_remove = [jid for jid, j in self.jobs.items() if j.status in {"COMPLETED", "CANCELLED"}]
            for jid in to_remove:
                del self.jobs[jid]
            self.save_queue()

    def delete_job(self, job_id: str) -> bool:
        with QUEUE_LOCK:
            if job_id in self.jobs:
                del self.jobs[job_id]
                self.save_queue()
                return True
            return False

    def update_job_progress(self, job_id: str, status: str, progress: int, stage_message: str, error_message: Optional[str] = None):
        with QUEUE_LOCK:
            job = self.jobs.get(job_id)
            if not job:
                return

            job.status = status
            job.progress = progress
            job.stage_message = stage_message
            if error_message:
                job.error_message = error_message

            if status == "COMPLETED":
                job.completed_at = time.time()
                job.progress = 100

            self.save_queue()

        self.broadcast_event({
            "type": "job_update",
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "stage_message": stage_message,
            "error_message": error_message
        })

    def broadcast_event(self, event_data: dict):
        """Broadcasts event payload to all connected SSE clients."""
        active_listeners = []
        for queue in list(self.listeners):
            try:
                queue.put_nowait(event_data)
                active_listeners.append(queue)
            except Exception:
                pass
        self.listeners = active_listeners

    def start_worker(self):
        if not self.is_running:
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="AutoVideoQueueWorker")
            self.worker_thread.start()
            print("[QueueWorker] Background processing worker thread started successfully.")

    def _worker_loop(self):
        """Sequential background processing loop running in dedicated worker thread."""
        while self.is_running:
            try:
                next_job = self._pop_next_queued_job()
                if next_job:
                    if "_process_single_job" in self.__dict__:
                        self.__dict__["_process_single_job"](next_job)
                    else:
                        self._process_single_job(next_job)
                else:
                    time.sleep(0.5)
            except Exception as e:
                print(f"[QueueWorker] Worker loop error: {e}")
                time.sleep(1.0)

    def _pop_next_queued_job(self) -> Optional[Job]:
        with QUEUE_LOCK:
            queued_jobs = [j for j in self.jobs.values() if j.status == "QUEUED"]
            if not queued_jobs:
                return None
            queued_jobs.sort(key=lambda j: j.created_at)
            return queued_jobs[0]

    def _log_job_message(self, log_file: Path, message: str):
        """Logs sanitized, timestamped processing message to job log file (excluding secrets)."""
        log_file.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def _process_single_job(self, job: Job):
        job_id = job.job_id
        job_dir = JOBS_DIR / job_id

        source_dir = job_dir / "source"
        audio_dir = job_dir / "audio"
        transcript_dir = job_dir / "transcript"
        voice_dir = job_dir / "voice"
        sub_dir = job_dir / "subtitles"
        render_dir = job_dir / "render"
        output_dir = job_dir / "output"
        logs_dir = job_dir / "logs"

        for d in [source_dir, audio_dir, transcript_dir, voice_dir, sub_dir, render_dir, output_dir, logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / "process.log"
        self._log_job_message(log_file, f"Starting processing for Job ID: {job_id} | URL: {job.url}")

        settings_dict = job.settings_snapshot or SettingsManager.get_instance().load_settings().model_dump()
        settings = Settings(**settings_dict)

        # Helper to check if job was cancelled by user
        def is_cancelled() -> bool:
            if job_id in self.cancel_requested:
                self._log_job_message(log_file, "Job cancellation requested by user.")
                self.update_job_progress(job_id, "CANCELLED", job.progress, "Cancelled by user")
                return True
            return False

        try:
            job.started_at = time.time()

            # ----------------------------------------------------
            # Stage 1: DOWNLOADING (10%)
            # ----------------------------------------------------
            if is_cancelled(): return
            video_path = str(source_dir / f"vid_{job_id}.mp4")
            if not (os.path.exists(video_path) and os.path.getsize(video_path) > 1000):
                from main import VIDEO_DIR
                matching_files = list(VIDEO_DIR.glob(f"vid_{job_id}.*"))
                if matching_files and os.path.exists(matching_files[0]) and os.path.getsize(matching_files[0]) > 1000:
                    video_path = str(matching_files[0])
                else:
                    video_path = None

            if video_path and os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
                self._log_job_message(log_file, f"Stage 1 Checkpoint: Using existing downloaded video file: {video_path}")
                self.update_job_progress(job_id, "DOWNLOADING", 10, "📥 Using Existing Downloaded Video File...")
                job.title = Path(video_path).stem.replace(f"vid_{job_id}", "Video Task")
            else:
                self.update_job_progress(job_id, "DOWNLOADING", 10, "📥 Downloading Video File (yt-dlp)...")
                self._log_job_message(log_file, "Stage 1: Downloading video via yt-dlp...")
                from main import process_video_download
                video_path = process_video_download(job.url, f"vid_{job_id}", resolution=settings.resolution.split("x")[0] if "x" in settings.resolution else "4k")
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found after download: {video_path}")
                job.title = Path(video_path).stem.replace(f"vid_{job_id}", "Video Task")
                self._log_job_message(log_file, f"Video downloaded successfully: {video_path}")

            # ----------------------------------------------------
            # Stage 2: EXTRACTING_AUDIO (20%)
            # ----------------------------------------------------
            if is_cancelled(): return
            extracted_audio = str(audio_dir / "extracted_audio.mp3")
            if os.path.exists(extracted_audio) and os.path.getsize(extracted_audio) > 1000:
                self._log_job_message(log_file, f"Stage 2 Checkpoint: Using existing extracted audio file: {extracted_audio}")
                self.update_job_progress(job_id, "EXTRACTING_AUDIO", 20, "🎵 Using Existing Extracted Audio Track...")
            else:
                self.update_job_progress(job_id, "EXTRACTING_AUDIO", 20, "🎵 Extracting 16kHz Audio Track (FFmpeg)...")
                self._log_job_message(log_file, "Stage 2: Extracting audio...")
                from services.processor import extract_audio_from_video
                extract_audio_from_video(video_path, extracted_audio)
                self._log_job_message(log_file, f"Audio extracted successfully: {extracted_audio}")

            # ----------------------------------------------------
            # Stage 3: TRANSCRIBING (35%)
            # ----------------------------------------------------
            if is_cancelled(): return
            transcript_json_path = transcript_dir / "transcript.json"
            segments = None
            if transcript_json_path.exists() and os.path.getsize(transcript_json_path) > 10:
                try:
                    with open(transcript_json_path, "r", encoding="utf-8") as f:
                        segments = json.load(f)
                    self._log_job_message(log_file, f"Stage 3 Checkpoint: Loaded existing transcription ({len(segments)} segments).")
                    self.update_job_progress(job_id, "TRANSCRIBING", 35, f"🎙️ Using Existing Transcript ({len(segments)} segments)...")
                except Exception as e:
                    self._log_job_message(log_file, f"Could not load transcript checkpoint: {e}")
                    segments = None

            if not segments:
                self.update_job_progress(job_id, "TRANSCRIBING", 35, "🎙️ Transcribing Audio Speech (Groq Whisper v3)...")
                self._log_job_message(log_file, "Stage 3: Groq Whisper Audio Transcription...")
                from services.transcriber import transcribe_audio
                trans_res = transcribe_audio(extracted_audio)
                segments = trans_res.get("segments", [])
                if not segments:
                    raise ValueError("No speech segments detected in video audio.")
                with open(transcript_json_path, "w", encoding="utf-8") as f:
                    json.dump(segments, f, ensure_ascii=False, indent=2)
                self._log_job_message(log_file, f"Transcription completed & saved checkpoint ({len(segments)} segments).")

            # ----------------------------------------------------
            # Stage 4: TRANSLATING / SCRIPT REWRITING (50%)
            # ----------------------------------------------------
            if is_cancelled(): return
            translated_json_path = transcript_dir / "translated_segments.json"
            translated_segments = None
            if translated_json_path.exists() and os.path.getsize(translated_json_path) > 10:
                try:
                    with open(translated_json_path, "r", encoding="utf-8") as f:
                        translated_segments = json.load(f)
                    self._log_job_message(log_file, "Stage 4 Checkpoint: Loaded existing AI translation script.")
                    self.update_job_progress(job_id, "TRANSLATING", 50, "🤖 Using Existing AI Translated Script...")
                except Exception as e:
                    self._log_job_message(log_file, f"Could not load translation checkpoint: {e}")
                    translated_segments = None

            if not translated_segments:
                self.update_job_progress(job_id, "TRANSLATING", 50, f"🤖 AI Script Rewriting & Translating ({settings.translation_style} style)...")
                self._log_job_message(log_file, f"Stage 4: AI Translation to {settings.target_language} ({settings.translation_style} style)...")
                from services.translator import translate_transcript
                live_settings = SettingsManager.get_instance().load_settings()
                ai_key = live_settings.gemini_api_key if settings.ai_provider == "gemini" else live_settings.openai_api_key
                if not ai_key or is_masked(ai_key):
                    ai_key = settings.gemini_api_key if settings.ai_provider == "gemini" else settings.openai_api_key
                translated_segments = translate_transcript(
                    segments=segments,
                    target_language=settings.target_language,
                    mode="full" if settings.script_rewriting_enabled else "full",
                    translation_style=settings.translation_style,
                    provider=settings.ai_provider,
                    api_key=ai_key
                )
                with open(translated_json_path, "w", encoding="utf-8") as f:
                    json.dump(translated_segments, f, ensure_ascii=False, indent=2)
                self._log_job_message(log_file, "AI Translation completed & saved checkpoint successfully.")

            # ----------------------------------------------------
            # Stage 5: GENERATING_VOICE (65%)
            # ----------------------------------------------------
            if is_cancelled(): return
            self.update_job_progress(job_id, "GENERATING_VOICE", 65, f"⚡ Synthesizing Voice TTS ({settings.voice})...")
            self._log_job_message(log_file, f"Stage 5: Voice Synthesis using {settings.voice} ({settings.voice_provider})...")

            from services.tts import generate_segment_tts_tracks
            from services.processor import create_master_audio_track

            # Determine engine, voice, pitch, and reference audio/text
            active_engine = getattr(settings, "voice_engine", None) or settings.voice_provider or "edge_tts"
            is_edge = active_engine in ["edge_tts", "edge"] or settings.voice.startswith("edge-") or settings.voice.startswith("edge_")

            target_sr = 48000 if getattr(settings, "audio_quality", "high") == "high" else 44100
            master_tts_audio = audio_dir / "master_tts.wav"

            seg_audio_json_path = voice_dir / "seg_with_audio.json"
            seg_with_audio = None

            if master_tts_audio.exists() and os.path.getsize(master_tts_audio) > 1000 and seg_audio_json_path.exists() and os.path.getsize(seg_audio_json_path) > 10:
                try:
                    with open(seg_audio_json_path, "r", encoding="utf-8") as f:
                        seg_with_audio = json.load(f)
                    self._log_job_message(log_file, "Stage 5 Checkpoint: Loaded existing master TTS audio & segment tracks.")
                    self.update_job_progress(job_id, "GENERATING_VOICE", 65, "⚡ Using Existing Master Voice Audio Track...")
                except Exception as e:
                    self._log_job_message(log_file, f"Could not load voice checkpoint: {e}")
                    seg_with_audio = None

            if not seg_with_audio:
                if is_edge:
                    # ----------------------------------------------------
                    # Microsoft Edge TTS Engine (Original Workflow)
                    # ----------------------------------------------------
                    active_voice = settings.edge_tts_voice or settings.voice or "my-MM-NilarNeural"
                    pitch_param = settings.edge_tts_pitch or "+1.2"
                    speed_param = getattr(settings, "edge_tts_speed", 1.0) or 1.0

                    from services.tts import generate_segment_tts_tracks
                    from services.processor import create_master_audio_track

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        seg_with_audio = loop.run_until_complete(
                            generate_segment_tts_tracks(
                                segments=translated_segments,
                                output_dir=voice_dir,
                                voice=active_voice,
                                prompt_wav_path=None,
                                prompt_text=None,
                                engine="edge_tts",
                                pitch=pitch_param,
                                speed=speed_param,
                                retry_count=settings.max_retries
                            )
                        )
                    finally:
                        loop.close()

                    create_master_audio_track(
                        segments=seg_with_audio,
                        output_audio_path=master_tts_audio,
                        target_sample_rate=target_sr,
                        audio_normalization=getattr(settings, "audio_normalization", True),
                        smooth_segment_join=getattr(settings, "smooth_segment_join", True),
                        silence_cleanup=getattr(settings, "silence_cleanup", True),
                        auto_audio_validation=getattr(settings, "auto_audio_validation", True)
                    )
                else:
                    # ----------------------------------------------------
                    # VoxCPM2 Voice Clone Long-Text Pipeline (~20s Chunks)
                    # ----------------------------------------------------
                    self._log_job_message(log_file, "VoxCPM2 Voice Clone selected. Executing long-text chunking & merge pipeline...")

                    # Resolve reference audio file (Settings reference audio or video audio)
                    ref_audio = getattr(settings, "voxcpm_reference_audio", None) or getattr(settings, "custom_voice_file", None)
                    voice_prompt_text = getattr(settings, "voxcpm_reference_text", None) or getattr(settings, "prompt_text", None)
                    ref_id = getattr(settings, "voxcpm_reference_id", None) or getattr(settings, "voice_clone_name", None)

                    # Fallback lookup from custom voices metadata if reference audio or text are empty or missing
                    if (not ref_audio or not os.path.exists(ref_audio) or not voice_prompt_text) and ref_id:
                        try:
                            from main import load_custom_voices_metadata
                            custom_voices = load_custom_voices_metadata()
                            v_match = next((v for v in custom_voices if v.get("id") == ref_id or v.get("name") == ref_id), None)
                            if v_match:
                                if not ref_audio or not os.path.exists(ref_audio):
                                    ref_audio = v_match.get("audio_path") or v_match.get("voice_file_path")
                                if not voice_prompt_text:
                                    voice_prompt_text = v_match.get("text") or v_match.get("reference_text")
                                print(f"[QueueWorker] Auto-resolved custom voice metadata for '{ref_id}': audio={ref_audio}, text='{voice_prompt_text[:30]}...'")
                        except Exception as ex:
                            print(f"[QueueWorker Warning] Could not resolve custom voice metadata for '{ref_id}': {ex}")

                    if ref_audio and os.path.exists(ref_audio):
                        voice_prompt_audio = ref_audio
                    else:
                        voice_prompt_audio = extracted_audio

                    if not voice_prompt_text and 'segments' in locals() and segments:
                        # Automatically populate prompt text from initial transcript segments of reference audio
                        first_few = [s.get("text", "") for s in segments[:3] if s.get("text")]
                        voice_prompt_text = " ".join(first_few).strip()

                    from services.voxcpm_pipeline import process_voxcpm_long_text_pipeline

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        seg_with_audio = loop.run_until_complete(
                            process_voxcpm_long_text_pipeline(
                                translated_segments=translated_segments,
                                output_voice_dir=voice_dir,
                                output_master_audio_path=master_tts_audio,
                                prompt_wav_path=voice_prompt_audio,
                                prompt_text=voice_prompt_text,
                                target_sample_rate=target_sr,
                                retry_count=settings.max_retries
                            )
                        )
                    finally:
                        loop.close()

                with open(seg_audio_json_path, "w", encoding="utf-8") as f:
                    json.dump(seg_with_audio, f, ensure_ascii=False, indent=2)

            self._log_job_message(log_file, "Master TTS audio track created & validated successfully.")

            # ----------------------------------------------------
            # Stage 6: GENERATING_SUBTITLES (78%)
            # ----------------------------------------------------
            if is_cancelled(): return
            self.update_job_progress(job_id, "GENERATING_SUBTITLES", 78, "📄 Generating Subtitles & Formatting Cues...")
            self._log_job_message(log_file, "Stage 6: SRT Subtitles Generation...")

            from services.processor import create_srt_file
            srt_path = sub_dir / f"recap_{job_id}.srt"
            create_srt_file(seg_with_audio, srt_path)

            # Copy SRT to public recap dir
            final_srt_public = RECAP_DIR / f"recap_{job_id}.srt"
            shutil.copyfile(srt_path, final_srt_public)
            self._log_job_message(log_file, f"SRT subtitle file generated: {srt_path}")

            # ----------------------------------------------------
            # Stage 7: RENDERING (90%)
            # ----------------------------------------------------
            if is_cancelled(): return
            self.update_job_progress(job_id, "RENDERING", 90, "🎬 RTX GPU NVENC Video Rendering & Speed Sync...")
            self._log_job_message(log_file, "Stage 7: GPU NVENC Video Rendering...")

            from services.processor import render_recap_video_gpu
            output_video_file = output_dir / f"recap_{job_id}.mp4"
            final_video_public = RECAP_DIR / f"recap_{job_id}.mp4"

            final_video_path = render_recap_video_gpu(
                video_path=video_path,
                audio_path=str(master_tts_audio),
                srt_path=str(srt_path),
                output_path=str(output_video_file),
                keep_original_audio=False,
                original_audio_volume=0.1,
                burn_subtitles=settings.burn_subtitles,
                sub_font_size=settings.sub_font_size,
                sub_font_color=settings.sub_font_color,
                sub_font_name=settings.sub_font_name,
                sub_font_file=settings.sub_font_file,
                sub_position=settings.sub_position,
                sub_margin_v=settings.sub_margin_v,
                sub_outline=getattr(settings, "sub_outline", True),
                sub_outline_color=getattr(settings, "sub_outline_color", "#000000"),
                sub_outline_width=getattr(settings, "sub_outline_width", 2),
                sub_shadow=getattr(settings, "sub_shadow", True),
                sub_shadow_strength=getattr(settings, "sub_shadow_strength", 1),
                sub_alignment=getattr(settings, "sub_alignment", "center"),
                sub_x=getattr(settings, "sub_x", None),
                sub_y=getattr(settings, "sub_y", None)
            )

            shutil.copyfile(output_video_file, final_video_public)
            self._log_job_message(log_file, f"GPU Video Rendering completed successfully: {final_video_path}")

            # Update job state to COMPLETED
            job.output_video_path = final_video_public.as_posix()
            job.output_srt_path = final_srt_public.as_posix()
            self.update_job_progress(job_id, "COMPLETED", 100, "✓ Processing Completed Successfully")
            self._log_job_message(log_file, "Job pipeline completed successfully!")

            # Auto clean temp files if enabled in settings
            if settings.auto_clean_temp:
                try:
                    from services.processor import cleanup_recap_temp_files
                    cleanup_recap_temp_files(job_dir, video_path)
                except Exception as e:
                    print(f"Error during auto clean temp files for job {job_id}: {e}")

        except Exception as e:
            err_msg = str(e)
            self._log_job_message(log_file, f"ERROR: Job execution failed: {err_msg}")
            print(f"[QueueWorker] Error processing job {job_id}: {err_msg}")

            if settings.auto_retry_failed and job.retry_count < settings.max_retries:
                job.retry_count += 1
                self.update_job_progress(job_id, "QUEUED", 0, f"Retrying attempt ({job.retry_count}/{settings.max_retries})...", error_message=err_msg)
                self._log_job_message(log_file, f"Scheduled retry attempt {job.retry_count}/{settings.max_retries}")
            else:
                self.update_job_progress(job_id, "FAILED", job.progress, "✕ Processing Failed", error_message=err_msg)
