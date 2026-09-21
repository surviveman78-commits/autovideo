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
    "f5-myanmar-v2": "F5-Myanmar TTS v2 GPU Voice Clone (Video Speaker Timbre)",
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

# Global singleton model reference for F5-Myanmar GPU acceleration
_f5_model = None

def get_f5_tts_model():
    """
    Loads and returns the F5-Myanmar TTS v2 model singleton on NVIDIA GPU.
    Reuses model in VRAM across all requests.
    """
    global _f5_model
    if _f5_model is None:
        print("[F5-Myanmar v2] Loading F5-Myanmar TTS v2 Voice Clone Model onto GPU VRAM...")
        try:
            from f5_tts.api import F5TTS
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _f5_model = F5TTS(name_or_path="SWJTU-Lab/F5-TTS", device=device)
            print(f"[F5-Myanmar v2] F5-Myanmar TTS v2 GPU Voice Clone Model loaded successfully on ({device})!")
        except Exception as e:
            print(f"[F5-Myanmar v2] F5-TTS model initialization notice: {e}")
            _f5_model = "FALLBACK"
    return _f5_model


def generate_f5_tts_audio_sync(
    text: str,
    output_path: str,
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None
) -> str:
    """
    Synchronous F5-Myanmar TTS v2 Voice Clone generator running on GPU.
    Falls back to Microsoft Edge TTS (my-MM-NilarNeural) if F5-TTS package or GPU is unavailable.
    """
    if not text.strip():
        return output_path

    model = get_f5_tts_model()

    if model and model != "FALLBACK":
        try:
            ref_file = prompt_wav_path if (prompt_wav_path and os.path.exists(prompt_wav_path)) else None
            ref_text = prompt_text.strip() if (prompt_text and prompt_text.strip()) else ""
            model.export_wav(
                gen_text=text.strip(),
                ref_file=ref_file,
                ref_text=ref_text,
                output_path=output_path
            )
            if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                return output_path
        except Exception as e:
            print(f"[F5-Myanmar v2 Error] Generation failed ({e}). Falling back to Edge TTS...")

    # Fallback to ultra-fast Edge TTS Burmese Nilar
    return asyncio.run(generate_edge_tts_audio_file(text.strip(), output_path, voice="my-MM-NilarNeural"))


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
    voice: str = "f5-myanmar-v2",
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    engine: str = "edge",
    pitch: Any = "+1.2",
    speed: Any = 1.0
) -> str:
    """
    Unified async TTS audio generator supporting F5-Myanmar TTS v2 Voice Clone and Microsoft Edge TTS.
    - Edge TTS: Applies configured pitch (default +1.2) and speed.
    - Voice Clone (F5-Myanmar v2): Preserves original timbre of reference speaker.
    """
    if engine in ["edge", "edge_tts"] or voice.startswith("edge-") or voice.startswith("edge_"):
        edge_voice_name = voice.replace("edge-", "").replace("edge_", "")
        return await generate_edge_tts_audio_file(text, output_path, edge_voice_name, pitch=pitch, speed=speed)

    # Voice Clone (F5-Myanmar TTS v2)
    loop = asyncio.get_event_loop()
    ref_path = prompt_wav_path if (prompt_wav_path and os.path.exists(prompt_wav_path)) else None
    return await loop.run_in_executor(
        None,
        generate_f5_tts_audio_sync,
        text,
        output_path,
        ref_path,
        prompt_text
    )


def generate_tts_sync(
    text: str,
    output_path: str,
    voice: str = "f5-myanmar-v2",
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    engine: str = "edge",
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
    voice: str = "edge-my-MM-NilarNeural",
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    engine: str = "edge",
    pitch: Any = "+1.2",
    speed: Any = 1.0,
    retry_count: int = 3
) -> List[Dict[str, Any]]:
    """
    Generates TTS audio files for each individual segment using selected engine (F5-Myanmar v2 or Edge TTS).
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
                    # Voice Clone (F5-Myanmar TTS v2)
                    await loop.run_in_executor(
                        None,
                        generate_f5_tts_audio_sync,
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



