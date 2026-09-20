import os
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import soundfile as sf
import edge_tts

# Add VoxCPM repository path to sys.path if needed
possible_voxcpm_paths = [
    os.environ.get("VOXCPM_SRC_PATH", ""),
    r"C:\Users\Zimmimoo\VoxCPM\src",
    os.path.join(os.getcwd(), "VoxCPM", "src"),
    os.path.join(os.getcwd(), "VoxCPM"),
    "/kaggle/working/VoxCPM/src",
    "/kaggle/working/VoxCPM",
    "/content/VoxCPM/src",
    "/content/VoxCPM"
]
for p in possible_voxcpm_paths:
    if p and p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

# Supported TTS Engines & Voices
DEFAULT_VOICES = {
    "voxcpm2-clone": "VoxCPM2 GPU Voice Clone (Video Speaker Timbre)",
    "voxcpm2-neural": "VoxCPM2 GPU Burmese Neural Studio Voice",
    "edge-my-MM-NilarNeural": "Microsoft Edge TTS - Burmese Female (Nilar)",
    "edge-my-MM-ThihaNeural": "Microsoft Edge TTS - Burmese Male (Thiha)",
    "edge-en-US-AvaNeural": "Microsoft Edge TTS - English Female (Ava)",
    "edge-en-US-AndrewNeural": "Microsoft Edge TTS - English Male (Andrew)"
}


def get_edge_tts_voices() -> List[Dict[str, str]]:
    """Returns structured list of available Microsoft Edge TTS voices with locale, language, and gender."""
    return [
        {"id": "my-MM-NilarNeural", "name": "မြန်မာ အမျိုးသမီး (Nilar)", "locale": "my-MM", "language": "Burmese", "gender": "Female"},
        {"id": "my-MM-ThihaNeural", "name": "မြန်မာ အမျိုးသား (Thiha)", "locale": "my-MM", "language": "Burmese", "gender": "Male"},
        {"id": "en-US-AvaNeural", "name": "English Female (Ava)", "locale": "en-US", "language": "English (US)", "gender": "Female"},
        {"id": "en-US-AndrewNeural", "name": "English Male (Andrew)", "locale": "en-US", "language": "English (US)", "gender": "Male"},
        {"id": "en-US-EmmaMultilingualNeural", "name": "English Multilingual (Emma)", "locale": "en-US", "language": "English (Multilingual)", "gender": "Female"},
        {"id": "en-GB-SoniaNeural", "name": "English Female (Sonia - UK)", "locale": "en-GB", "language": "English (UK)", "gender": "Female"}
    ]

# Global singleton model reference for GPU acceleration
_voxcpm_model = None

def get_voxcpm_model():
    """
    Loads and returns the VoxCPM2 model singleton on NVIDIA RTX GPU.
    Reuses model in VRAM across all requests for maximum GPU throughput.
    """
    global _voxcpm_model
    if _voxcpm_model is None:
        print("[VoxCPM2] Loading VoxCPM2 Voice Clone Model onto GPU VRAM...")
        try:
            from voxcpm import VoxCPM
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _voxcpm_model = VoxCPM.from_pretrained(
                "openbmb/VoxCPM2",
                device=device,
                load_denoiser=False
            )
            print(f"[VoxCPM2] VoxCPM2 GPU Voice Clone Model loaded successfully on ({device})!")
        except Exception as e:
            print(f"[VoxCPM2] Error loading VoxCPM2 GPU model: {e}")
            raise RuntimeError(f"VoxCPM2 GPU Model load failed: {str(e)}")
    return _voxcpm_model


def generate_voxcpm_audio_sync(
    text: str,
    output_path: str,
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    inference_timesteps: int = 10,
    cfg_value: float = 2.0,
    seed: int = 42
) -> str:
    """
    Synchronous VoxCPM2 TTS & Voice Clone generator running on GPU.
    Uses constant random seed (42) for deterministic, identical speaker voice timbre across chunks.
    """
    if not text.strip():
        return output_path

    model = get_voxcpm_model()

    kwargs = {
        "text": text.strip(),
        "inference_timesteps": inference_timesteps,
        "cfg_value": cfg_value,
        "seed": seed
    }

    has_wav = bool(prompt_wav_path and os.path.exists(prompt_wav_path))
    has_text = bool(prompt_text and prompt_text.strip())

    if has_wav and not has_text:
        try:
            from services.transcriber import transcribe_audio
            res = transcribe_audio(prompt_wav_path)
            auto_text = res.get("text", "").strip()
            if auto_text:
                prompt_text = auto_text[:150]
                has_text = True
                print(f"[VoxCPM2] Auto-transcribed prompt text for reference audio: '{prompt_text[:30]}...'")
        except Exception as e:
            print(f"[VoxCPM2 Warning] Could not auto-transcribe reference WAV: {e}")

    if has_wav and has_text:
        kwargs["prompt_wav_path"] = prompt_wav_path
        kwargs["reference_wav_path"] = prompt_wav_path
        kwargs["prompt_text"] = prompt_text.strip()
    else:
        print("[VoxCPM2] Prompt audio/text missing or incomplete. Using neural voice mode (both set to None).")

    try:
        wav = model.generate(**kwargs)
        
        # Sample rate from VoxCPM model (48000 Hz studio quality)
        sr = 48000
        if hasattr(model, 'tts_model') and hasattr(model.tts_model, 'sample_rate'):
            sr = model.tts_model.sample_rate
        elif hasattr(model, 'sample_rate'):
            sr = model.sample_rate

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, wav, sr)
    except Exception as e:
        print(f"VoxCPM2 generation error for text '{text[:20]}...': {e}")
        raise e

    return output_path


def format_edge_pitch(pitch_val: Any) -> str:
    """Formats pitch value into valid edge-tts pitch string matching ^[+-]\\d+Hz$."""
    if isinstance(pitch_val, str) and (pitch_val.endswith("Hz") or pitch_val.endswith("%")):
        return pitch_val
    try:
        val = float(pitch_val)
        val_int = int(round(val))
        return f"{val_int:+d}Hz"
    except Exception:
        return "+1Hz"


def format_edge_rate(speed_val: Any) -> str:
    """Formats speed value into valid edge-tts rate string matching ^[+-]\\d+%$."""
    if isinstance(speed_val, str) and speed_val.endswith("%"):
        return speed_val
    try:
        val = float(speed_val)
        pct = int(round((val - 1.0) * 100))
        return f"{pct:+d}%"
    except Exception:
        return "+0%"


async def generate_edge_tts_audio_file(
    text: str,
    output_path: str,
    voice: str = "my-MM-NilarNeural",
    pitch: Any = "+1.2",
    speed: Any = 1.0
) -> str:
    """
    Generates audio using Microsoft Edge TTS async engine with configurable pitch and speed.
    Default Pitch: +1.2 (+1Hz for edge_tts)
    Default Speed: 1.0 (+0% for edge_tts)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    clean_voice = voice.replace("edge-", "").replace("edge_", "")
    if clean_voice not in ["my-MM-NilarNeural", "my-MM-ThihaNeural", "en-US-AvaNeural", "en-US-AndrewNeural"]:
        clean_voice = "my-MM-NilarNeural"
    
    pitch_str = format_edge_pitch(pitch)
    rate_str = format_edge_rate(speed)

    communicate = edge_tts.Communicate(text.strip(), clean_voice, pitch=pitch_str, rate=rate_str)
    await communicate.save(output_path)

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Edge TTS generated empty or missing file at {output_path}")

    return output_path


async def generate_tts_audio_file(
    text: str,
    output_path: str,
    voice: str = "voxcpm2-clone",
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    engine: str = "voxcpm",
    pitch: Any = "+1.2",
    speed: Any = 1.0
) -> str:
    """
    Unified async TTS audio generator supporting VoxCPM2 Voice Clone and Microsoft Edge TTS.
    - Edge TTS: Applies configured pitch (default +1.2) and speed.
    - Voice Clone: DO NOT modify pitch. Keeps natural original pitch of reference voice.
    """
    if engine in ["edge", "edge_tts"] or voice.startswith("edge-") or voice.startswith("edge_"):
        edge_voice_name = voice.replace("edge-", "").replace("edge_", "")
        return await generate_edge_tts_audio_file(text, output_path, edge_voice_name, pitch=pitch, speed=speed)

    # Voice Clone (VoxCPM2) - Preserves original natural pitch (pitch modification disabled/off)
    loop = asyncio.get_event_loop()
    ref_path = prompt_wav_path if (prompt_wav_path and os.path.exists(prompt_wav_path)) else None
    return await loop.run_in_executor(
        None,
        generate_voxcpm_audio_sync,
        text,
        output_path,
        ref_path,
        prompt_text
    )


def generate_tts_sync(
    text: str,
    output_path: str,
    voice: str = "voxcpm2-clone",
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    engine: str = "voxcpm",
    pitch: Any = "+1.2",
    speed: Any = 1.0
) -> str:
    """
    Synchronous wrapper for generate_tts_audio_file.
    """
    return asyncio.run(generate_tts_audio_file(text, output_path, voice, prompt_wav_path, prompt_text, engine, pitch, speed))


async def generate_segment_tts_tracks(
    segments: List[Dict[str, Any]],
    output_dir: Path,
    voice: str = "voxcpm2-clone",
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    engine: str = "voxcpm",
    pitch: Any = "+1.2",
    speed: Any = 1.0,
    retry_count: int = 3
) -> List[Dict[str, Any]]:
    """
    Generates TTS audio files for each individual segment using selected engine (VoxCPM2 or Edge TTS).
    Validates output audio files and retries if generation fails or audio duration is zero.
    Returns segments list with added 'audio_path'.
    """
    from services.processor import split_long_segments

    output_dir.mkdir(parents=True, exist_ok=True)
    split_segs = split_long_segments(segments, max_chars=48)
    updated_segments = []
    loop = asyncio.get_event_loop()

    ref_path = prompt_wav_path if (prompt_wav_path and os.path.exists(prompt_wav_path)) else None
    is_edge = engine in ["edge", "edge_tts"] or voice.startswith("edge-") or voice.startswith("edge_")

    for idx, seg in enumerate(split_segs):
        text = seg.get("text", "").strip()
        if not text:
            continue

        ext = ".mp3" if is_edge else ".wav"
        seg_audio_path = str(output_dir / f"seg_{idx:04d}{ext}")

        success = False
        last_err = None

        for attempt in range(1, retry_count + 1):
            try:
                if is_edge:
                    edge_voice_name = voice.replace("edge-", "").replace("edge_", "")
                    await generate_edge_tts_audio_file(text, seg_audio_path, edge_voice_name, pitch=pitch, speed=speed)
                else:
                    # Voice Clone: Original Pitch (NO pitch modification)
                    await loop.run_in_executor(
                        None,
                        generate_voxcpm_audio_sync,
                        text,
                        seg_audio_path,
                        ref_path,
                        prompt_text
                    )

                # Validation: file must exist and have non-zero size
                if os.path.exists(seg_audio_path) and os.path.getsize(seg_audio_path) > 100:
                    success = True
                    break
                else:
                    raise ValueError(f"Generated segment audio file is invalid or empty: {seg_audio_path}")
            except Exception as e:
                last_err = e
                print(f"[TTS Segment {idx}] Attempt {attempt}/{retry_count} failed: {e}")
                await asyncio.sleep(0.5)

        if success:
            seg_copy = dict(seg)
            seg_copy["audio_path"] = seg_audio_path
            updated_segments.append(seg_copy)
        else:
            print(f"[TTS Segment {idx}] CRITICAL: All {retry_count} attempts failed for text: '{text[:20]}...'. Error: {last_err}")
            raise RuntimeError(f"Failed to generate TTS audio for segment {idx}: {last_err}")

    return updated_segments



