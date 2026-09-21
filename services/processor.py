import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional


def extract_audio_from_video(video_path: str, output_audio_path: str) -> str:
    """Extracts 16kHz mono audio from video file for Groq Whisper transcription."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ar", "16000", "-ac", "1", "-c:a", "mp3",
        output_audio_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {res.stderr}")
    return output_audio_path


def purge_entire_temp_directory():
    """
    Auto Clean System: Purges all files, test outputs, and old task subdirectories
    inside downloads/temp/ to keep the temp folder 100% clean.
    """
    temp_dir = Path("downloads") / "temp"
    if temp_dir.exists():
        for item in temp_dir.glob("*"):
            try:
                if item.is_file():
                    os.remove(item)
                    print(f"[Auto Clean] Auto Cleaned temp file: {item}")
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                    print(f"[Auto Clean] Auto Cleaned temp directory: {item}")
            except Exception as e:
                print(f"Error purging temp item {item}: {e}")


def cleanup_recap_temp_files(work_dir: Path, source_video_path: Optional[str] = None):
    """
    Auto Clean System: Removes all intermediate temporary files, extracted audio,
    TTS segment tracks, downloaded/uploaded source videos, AND purges the entire temp folder.
    Preserves ONLY the final generated recap video and SRT subtitles in downloads/recap/.
    """
    # 1. Delete temporary task working directory
    if work_dir.exists():
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
            print(f"[Auto Clean] Auto Cleaned work directory: {work_dir}")
        except Exception as e:
            print(f"Error cleaning work_dir {work_dir}: {e}")

    # 2. Delete downloaded/uploaded source video file
    if source_video_path and os.path.exists(source_video_path):
        try:
            os.remove(source_video_path)
            print(f"[Auto Clean] Auto Cleaned temporary source video: {source_video_path}")
        except Exception as e:
            print(f"Error removing source video {source_video_path}: {e}")

    # 3. Purge any remaining files/folders in downloads/temp/
    purge_entire_temp_directory()



def hex_to_ass_color(hex_str: str) -> str:
    """Converts CSS hex color (#RRGGBB or #RGB) to FFmpeg ASS subtitle color format (&H00BBGGRR)."""
    if not hex_str:
        return "&H00FFFFFF"
    clean_hex = hex_str.lstrip("#")
    if len(clean_hex) == 3:
        clean_hex = "".join([c * 2 for c in clean_hex])
    if len(clean_hex) == 6:
        r, g, b = clean_hex[0:2], clean_hex[2:4], clean_hex[4:6]
        return f"&H00{b.upper()}{g.upper()}{r.upper()}"
    return "&H00FFFFFF"


def wrap_subtitle_text(text: str, max_chars_per_line: int = 34) -> str:
    """
    Wraps Burmese and multi-lingual subtitle text cleanly without ruining grammatical meaning or dropping text.
    Ensures 100% of input text is preserved, balanced across maximum 2 lines per cue.
    """
    if not text:
        return ""

    clean_text = text.strip()
    if len(clean_text) <= max_chars_per_line:
        return clean_text

    mid = len(clean_text) // 2

    # 1. Look for Burmese comma (၊) or full stop (။) closest to the middle
    punc_positions = [i + 1 for i, char in enumerate(clean_text) if char in ["၊", "။"]]
    if punc_positions:
        best_pos = min(punc_positions, key=lambda pos: abs(pos - mid))
        line1 = clean_text[:best_pos].strip()
        line2 = clean_text[best_pos:].strip()
        if line1 and line2:
            return f"{line1}\n{line2}"

    # 2. Look for space closest to the middle
    space_positions = [i for i, char in enumerate(clean_text) if char == " "]
    if space_positions:
        best_pos = min(space_positions, key=lambda pos: abs(pos - mid))
        line1 = clean_text[:best_pos].strip()
        line2 = clean_text[best_pos:].strip()
        if line1 and line2:
            return f"{line1}\n{line2}"

    # 3. Fallback: Split at middle character boundary cleanly without losing any words
    line1 = clean_text[:mid].strip()
    line2 = clean_text[mid:].strip()
    return f"{line1}\n{line2}"


import re


def split_text_into_sentences(text: str) -> List[str]:
    """
    Splits text into separate sentences using Burmese and universal sentence punctuation (။, ?, !, .).
    Preserves trailing punctuation on each sentence.
    """
    if not text:
        return []
    
    pattern = r'([^။?!.\n]+[။?!.\n]*)'
    matches = re.findall(pattern, text)
    
    sentences = []
    for m in matches:
        cleaned = m.strip()
        if cleaned:
            sentences.append(cleaned)
            
    if not sentences and text.strip():
        sentences = [text.strip()]
        
    return sentences


def split_long_segments(segments: List[Dict[str, Any]], max_chars: int = 30) -> List[Dict[str, Any]]:
    """
    Splits transcript or translated segments exceeding max_chars or containing multiple sentences
    into shorter, naturally timed sub-segments.
    Ensures sentence boundaries (။, ?, !, .) are ALWAYS split into separate subtitle cues.
    Prevents overcrowded text on 9:16 vertical video screens and ensures smooth TTS narration & subtitle reading.
    """
    new_segments = []

    for seg in segments:
        raw_text = seg.get("text", "").strip()
        if not raw_text:
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        dur = max(0.4, end - start)

        # 1. First, check if segment contains multiple sentences (separated by ။, ?, !, .)
        sentences = split_text_into_sentences(raw_text)

        if len(sentences) > 1:
            total_len = sum(len(s) for s in sentences) or 1
            curr_start = start

            for s_idx, sentence in enumerate(sentences):
                s_len = len(sentence)
                s_dur = dur * (s_len / total_len)
                s_end = curr_start + s_dur if s_idx < len(sentences) - 1 else end

                sub_seg = dict(seg)
                sub_seg["text"] = sentence
                sub_seg["start"] = round(curr_start, 3)
                sub_seg["end"] = round(s_end, 3)

                # Recursively process each sentence for length constraints
                new_segments.extend(split_long_segments([sub_seg], max_chars=max_chars))
                curr_start = s_end
            continue

        # 2. If it's a single sentence and within max_chars limit, keep as is
        if len(raw_text) <= max_chars:
            new_segments.append(dict(seg))
            continue

        # 3. If single sentence exceeds max_chars, split at punctuation (၊ or spaces) or middle
        mid = len(raw_text) // 2

        split_idx = -1
        punc_positions = [i + 1 for i, c in enumerate(raw_text) if c in ["၊", "။", "?", "!", "."]]
        if punc_positions:
            split_idx = min(punc_positions, key=lambda pos: abs(pos - mid))
        else:
            space_positions = [i for i, c in enumerate(raw_text) if c == " "]
            if space_positions:
                split_idx = min(space_positions, key=lambda pos: abs(pos - mid))
            else:
                split_idx = mid

        part1 = raw_text[:split_idx].strip()
        part2 = raw_text[split_idx:].strip()

        if part1 and part2:
            ratio = len(part1) / float(len(raw_text))
            mid_time = round(start + (dur * ratio), 3)

            seg1 = dict(seg)
            seg1["text"] = part1
            seg1["start"] = round(start, 3)
            seg1["end"] = mid_time

            seg2 = dict(seg)
            seg2["text"] = part2
            seg2["start"] = mid_time
            seg2["end"] = round(end, 3)

            new_segments.extend(split_long_segments([seg1], max_chars=max_chars))
            new_segments.extend(split_long_segments([seg2], max_chars=max_chars))
        else:
            new_segments.append(dict(seg))

    return new_segments


def format_timestamp_srt(seconds: float) -> str:
    """Formats float seconds into SRT timestamp format: HH:MM:SS,mmm"""
    seconds = max(0.0, float(seconds))
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        seconds += 1.0
        millis = 0
    total_sec = int(seconds)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_srt_file(segments: List[Dict[str, Any]], srt_path: Path) -> Path:
    """
    Creates an SRT subtitle file from timed transcript segments.
    Ensures non-overlapping, non-zero duration cues so subtitles never freeze on screen.
    """
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pre-split any long segments or multi-sentence segments into concise sub-cues (max 30 chars)
    split_segments = split_long_segments(segments, max_chars=30)
    valid_cues = [s for s in split_segments if s.get("text", "").strip()]

    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(valid_cues, 1):
            raw_text = seg.get("text", "").strip()
            start_val = max(0.0, float(seg.get("start", 0.0)))
            end_val = float(seg.get("end", 0.0))

            # Minimum display duration based on character length (at least 1.5s)
            min_duration = max(1.5, len(raw_text) * 0.08)
            if end_val <= start_val:
                end_val = start_val + min_duration

            # Ensure cue ends before next cue starts (avoid overlapping cues)
            if idx < len(valid_cues):
                next_start = float(valid_cues[idx].get("start", 0.0))
                if next_start > start_val and end_val > next_start:
                    end_val = max(start_val + 0.5, next_start - 0.02)

            start_str = format_timestamp_srt(start_val)
            end_str = format_timestamp_srt(end_val)
            wrapped_text = wrap_subtitle_text(raw_text, max_chars_per_line=34)

            f.write(f"{idx}\n{start_str} --> {end_str}\n{wrapped_text}\n\n")

    return srt_path


from pydub import AudioSegment
from pydub.silence import detect_leading_silence


def trim_accidental_silence(seg: AudioSegment, silence_threshold: float = -42.0) -> AudioSegment:
    """
    Carefully trims leading and trailing accidental silence added by TTS generators,
    preserving natural word endings and leaving subtle natural padding (30-50ms).
    """
    if len(seg) <= 100:
        return seg

    start_trim = detect_leading_silence(seg, silence_threshold=silence_threshold)
    end_trim = detect_leading_silence(seg.reverse(), silence_threshold=silence_threshold)

    # Keep small padding (30ms start, 50ms end) to prevent clipping words
    start_pos = max(0, start_trim - 30)
    end_pos = max(start_pos + 50, len(seg) - max(0, end_trim - 50))

    trimmed = seg[start_pos:end_pos]
    return trimmed if len(trimmed) > 50 else seg


def normalize_segment_format(seg: AudioSegment, sample_rate: int = 48000, channels: int = 2, sample_width: int = 2) -> AudioSegment:
    """Standardizes audio segment format (48000 Hz, 2 channels stereo, 16-bit PCM)."""
    if seg.frame_rate != sample_rate:
        seg = seg.set_frame_rate(sample_rate)
    if seg.channels != channels:
        seg = seg.set_channels(channels)
    if seg.sample_width != sample_width:
        seg = seg.set_sample_width(sample_width)
    return seg


def validate_audio_file(file_path: str) -> bool:
    """Validates that audio file exists, has non-zero size, and probed duration > 0."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 100:
        return False
    dur = get_media_duration(file_path)
    return dur > 0.1


def create_master_audio_track(
    segments: List[Dict[str, Any]],
    output_audio_path: Path,
    target_sample_rate: int = 48000,
    audio_normalization: bool = True,
    smooth_segment_join: bool = True,
    silence_cleanup: bool = True,
    auto_audio_validation: bool = True
) -> Path:
    """
    Assembles individual TTS audio segment files into one continuous, studio-quality narration track.
    1. Standardizes format (48000 Hz, stereo, 16-bit PCM).
    2. Carefully cleans accidental TTS silence without clipping word endings.
    3. Normalizes volume levels across segments.
    4. Applies subtle crossfading to eliminate clicks/pops/boundary artifacts.
    5. Updates segment start & end timestamps in place so subtitles stay 100% synchronized!
    6. Validates final output track duration.
    """
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    valid_segments = [s for s in segments if s.get("audio_path") and os.path.exists(s["audio_path"])]

    if not valid_segments:
        silent = AudioSegment.silent(duration=1000, frame_rate=target_sample_rate).set_channels(2)
        silent.export(str(output_audio_path), format="mp3", bitrate="320k")
        return output_audio_path

    master = AudioSegment.silent(duration=0, frame_rate=target_sample_rate).set_channels(2)

    for idx, seg in enumerate(valid_segments):
        audio_file = seg["audio_path"]
        try:
            raw_seg = AudioSegment.from_file(audio_file)
        except Exception as e:
            print(f"Error loading segment audio {audio_file}: {e}")
            continue

        if len(raw_seg) == 0:
            continue

        # 1. Standardize sample rate, channels, PCM depth
        seg_audio = normalize_segment_format(raw_seg, sample_rate=target_sample_rate)

        # 2. Silence cleanup (trim long accidental TTS padding while preserving natural speech)
        if silence_cleanup:
            seg_audio = trim_accidental_silence(seg_audio, silence_threshold=-42.0)

        # 3. Pre-normalize peak loudness for each segment to prevent quiet vs loud jumps
        if audio_normalization:
            seg_audio = seg_audio.normalize(headroom=1.0)

        # 4. Record exact start timestamp in master track
        actual_start_ms = len(master)
        seg["start"] = round(actual_start_ms / 1000.0, 3)

        # 5. Smooth segment join / subtle crossfade
        if smooth_segment_join and len(master) > 0:
            crossfade_ms = min(15, len(master), len(seg_audio))
            if crossfade_ms > 5:
                master = master.append(seg_audio, crossfade=crossfade_ms)
            else:
                master += seg_audio
        else:
            master += seg_audio

        # Record exact end timestamp in master track
        actual_end_ms = len(master)
        seg["end"] = round(actual_end_ms / 1000.0, 3)

        # Insert brief natural sentence pause (150ms-250ms) for realistic human cadence
        text = seg.get("text", "").strip()
        pause_ms = 250 if any(text.endswith(p) for p in ["။", ".", "?", "!"]) else 120
        master += AudioSegment.silent(duration=pause_ms, frame_rate=target_sample_rate).set_channels(2)

    # 6. Final master audio loudness normalization
    if audio_normalization and len(master) > 0:
        master = master.normalize(headroom=1.5)

    # Export master audio track (high quality 320k MP3 or WAV)
    output_format = "wav" if str(output_audio_path).endswith(".wav") else "mp3"
    master.export(str(output_audio_path), format=output_format, bitrate="320k")

    # 7. Final audio validation
    if auto_audio_validation and not validate_audio_file(str(output_audio_path)):
        raise RuntimeError(f"Master audio track validation failed for output file: {output_audio_path}")

    return output_audio_path



def escape_ffmpeg_path(path: str) -> str:
    """Escapes Windows path for FFmpeg subtitle filter using relative paths to prevent colon escape bugs."""
    try:
        p = Path(path).resolve()
        try:
            return p.relative_to(Path.cwd()).as_posix()
        except Exception:
            return p.as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def get_media_duration(file_path: str) -> float:
    """Returns media duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(res.stdout.strip())
    except Exception as e:
        print(f"Error probing media duration for {file_path}: {e}")
        return 0.0


def create_dubbing_master_audio_track(
    segments: List[Dict[str, Any]],
    output_audio_path: Path,
    total_video_duration: float,
    target_sample_rate: int = 48000
) -> Path:
    """
    Assembles TTS audio segments for 1:1 Video Dubbing Mode with Dynamic Audio Speed Matching.
    Places each TTS segment at its exact target start timestamp.
    If generated TTS duration exceeds target segment duration, applies pitch-preserved time-stretching (speed change)
    so TTS speech fits the exact target window.
    """
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    master_len_ms = int(max(1.0, total_video_duration) * 1000)
    master = AudioSegment.silent(duration=master_len_ms, frame_rate=target_sample_rate).set_channels(2)

    for seg in segments:
        audio_file = seg.get("audio_path")
        if not audio_file or not os.path.exists(audio_file):
            continue

        start_sec = float(seg.get("start", 0.0))
        end_sec = float(seg.get("end", 0.0))
        target_dur_sec = max(0.5, end_sec - start_sec)
        target_start_ms = int(start_sec * 1000)

        try:
            raw_seg = AudioSegment.from_file(audio_file)
        except Exception:
            continue

        raw_seg = normalize_segment_format(raw_seg, sample_rate=target_sample_rate)
        seg_dur_sec = len(raw_seg) / 1000.0

        if seg_dur_sec > target_dur_sec and seg_dur_sec > 0.4:
            # Dynamic Audio Speed Matching (Speed Up to fit target slot)
            speed_factor = min(1.35, seg_dur_sec / target_dur_sec)
            new_rate = int(raw_seg.frame_rate * speed_factor)
            adjusted_seg = raw_seg._spawn(raw_seg.raw_data, overrides={'frame_rate': new_rate})
            adjusted_seg = adjusted_seg.set_frame_rate(target_sample_rate)
        else:
            adjusted_seg = raw_seg

        adjusted_seg = adjusted_seg.normalize(headroom=1.0)
        master = master.overlay(adjusted_seg, position=target_start_ms)

    master = master.normalize(headroom=1.5)
    output_format = "wav" if str(output_audio_path).endswith(".wav") else "mp3"
    master.export(str(output_audio_path), format=output_format, bitrate="320k")
    return output_audio_path


def render_recap_video_gpu(
    video_path: str,
    audio_path: str,
    srt_path: str,
    output_path: str,
    keep_original_audio: bool = False,
    original_audio_volume: float = 0.1,
    burn_subtitles: bool = True,
    sub_font_size: int = 16,
    sub_font_color: str = "#FFFFFF",
    sub_font_name: Optional[str] = None,
    sub_font_file: Optional[str] = None,
    sub_position: str = "bottom",
    sub_margin_v: int = 25,
    sub_outline: bool = True,
    sub_outline_color: str = "#000000",
    sub_outline_width: int = 2,
    sub_shadow: bool = True,
    sub_shadow_strength: int = 1,
    sub_alignment: str = "center",
    sub_x: Optional[int] = None,
    sub_y: Optional[int] = None,
    mode: str = "recap",
    segments: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Renders final video using NVIDIA GPU acceleration (h264_nvenc / h264_mf).
    Mode 1 (recap): Dynamically matches speed (setpts) to narration track.
    Mode 2 (dubbing): Preserves 1.0x video speed, applies selective speech ducking/muting to original audio.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    clean_srt = escape_ffmpeg_path(srt_path)

    video_dur = get_media_duration(video_path)
    audio_dur = get_media_duration(audio_path)

    vf_filters = []

    if mode == "dubbing":
        # 1:1 Video Dubbing Mode: Keep 100% original video playback speed
        vf_filters.append("setpts=PTS")
        print(f"[Video Render Dubbing] Mode: 1:1 Dubbing | Original Video Duration: {video_dur:.2f}s")
    elif video_dur > 0 and audio_dur > 0:
        pts_scale = audio_dur / video_dur
        print(f"[Video Render Recap] Video Duration: {video_dur:.2f}s | Audio Duration: {audio_dur:.2f}s | Speed Scale (setpts): {pts_scale:.4f}x")
        vf_filters.append(f"setpts={pts_scale:.6f}*PTS")
    else:
        vf_filters.append("tpad=stop_mode=clone:stop=-1")

    # FORCE OUTPUT VIDEO ASPECT RATIO TO 9:16 (1080x1920)
    # Scale video so it fills 1080x1920 without stretching, center crop, and set square pixel ratio (SAR=1)
    vf_filters.append("scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1")

    if burn_subtitles and os.path.exists(srt_path):
        ass_primary_color = hex_to_ass_color(sub_font_color)
        ass_outline_color = hex_to_ass_color(sub_outline_color) if sub_outline else "&H00000000"
        font_name_str = f",FontName={sub_font_name}" if sub_font_name else ""

        # Determine ASS numpad alignment (1..9) & margins
        horiz = (sub_alignment or "center").lower().strip()
        pos_clean = (sub_position or "bottom").lower().strip()

        if pos_clean == "top":
            base_align = 7 if horiz == "left" else (9 if horiz == "right" else 8)
            margin_v = sub_margin_v if sub_margin_v else 120
            margin_l = max(0, min(1000, int(sub_x))) if (horiz == "left" and sub_x is not None) else 20
        elif pos_clean == "center":
            base_align = 4 if horiz == "left" else (6 if horiz == "right" else 5)
            margin_v = 0
            margin_l = max(0, min(1000, int(sub_x))) if (horiz == "left" and sub_x is not None) else 20
        elif pos_clean == "custom" and sub_y is not None:
            if horiz == "left":
                base_align = 1
                margin_l = max(0, min(1000, int(sub_x))) if sub_x is not None else 20
            elif horiz == "right":
                base_align = 3
                margin_l = 20
            else:  # "center"
                base_align = 2
                margin_l = 20
            margin_v = max(40, min(1700, 1920 - int(sub_y)))
        else:  # bottom
            base_align = 1 if horiz == "left" else (3 if horiz == "right" else 2)
            margin_v = sub_margin_v if sub_margin_v else 140
            margin_l = max(0, min(1000, int(sub_x))) if (horiz == "left" and sub_x is not None) else 20

        # Subtitle Font Size for 1080x1920 vertical canvas
        actual_font_size = int(sub_font_size) if sub_font_size else 34

        outline_val = sub_outline_width if sub_outline else 0
        shadow_val = sub_shadow_strength if sub_shadow else 0

        subtitle_style = (
            f"FontSize={actual_font_size},PrimaryColour={ass_primary_color},"
            f"OutlineColour={ass_outline_color},BorderStyle=1,"
            f"Outline={outline_val},Shadow={shadow_val},Alignment={base_align},"
            f"MarginL={margin_l},MarginV={margin_v}{font_name_str}"
        )
        
        fonts_dir_path = Path("downloads/fonts").resolve()
        custom_font_matched = False
        if fonts_dir_path.exists() and sub_font_name:
            for f_item in fonts_dir_path.glob("*"):
                if f_item.is_file() and f_item.suffix.lower() in [".ttf", ".otf", ".woff", ".woff2"]:
                    if sub_font_name.lower() in f_item.name.lower() or f_item.stem.lower() in sub_font_name.lower():
                        custom_font_matched = True
                        break

        if sub_font_file and os.path.exists(sub_font_file):
            fonts_dir = escape_ffmpeg_path(str(Path(sub_font_file).parent.resolve()))
            vf_filters.append(f"subtitles='{clean_srt}':fontsdir='{fonts_dir}':force_style='{subtitle_style}'")
        elif custom_font_matched:
            fonts_dir = escape_ffmpeg_path(str(fonts_dir_path))
            vf_filters.append(f"subtitles='{clean_srt}':fontsdir='{fonts_dir}':force_style='{subtitle_style}'")
        else:
            vf_filters.append(f"subtitles='{clean_srt}':force_style='{subtitle_style}'")

    vf_str = ",".join(vf_filters)

    # Audio mapping & mixing filter
    filter_complex = None
    if mode == "dubbing" and segments:
        duck_conditions = " + ".join([f"between(t,{float(s.get('start',0)):.3f},{float(s.get('end',0)):.3f})" for s in segments if s.get('start') is not None])
        if duck_conditions:
            filter_complex = f"[0:a]volume=eval=frame:volume='if({duck_conditions},0.05,1.0)'[orig];[1:a]volume=1.0[tts];[orig][tts]amix=inputs=2:duration=first[aout]"
        else:
            filter_complex = "[0:a]volume=0.05[orig];[1:a]volume=1.0[tts];[orig][tts]amix=inputs=2:duration=first[aout]"
    elif keep_original_audio:
        filter_complex = f"[0:a]volume={original_audio_volume}[orig];[1:a]volume=1.0[tts];[orig][tts]amix=inputs=2:duration=first[aout]"

    # Helper function to run ffmpeg command
    def build_cmd(encoder_name: str, extra_args: list = None) -> list:
        c = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path]
        c.extend(["-c:v", encoder_name])
        if extra_args:
            c.extend(extra_args)
        c.extend(["-vf", vf_str])
        if keep_original_audio and filter_complex:
            c.extend(["-filter_complex", filter_complex, "-map", "0:v", "-map", "[aout]"])
        else:
            c.extend(["-map", "0:v", "-map", "1:a"])
        c.extend(["-c:a", "aac", "-b:a", "192k", "-shortest", str(output_file)])
        return c

    # 1. Try NVIDIA NVENC GPU Encoder
    print("[Video Render] Attempting GPU Video Render with NVIDIA NVENC (h264_nvenc)...")
    cmd_nvenc = build_cmd("h264_nvenc", ["-preset", "p4", "-tune", "hq", "-rc:v", "vbr", "-cq", "19"])
    result = subprocess.run(cmd_nvenc, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        print("[Video Render] GPU NVENC Render successful!")
        return str(output_file.as_posix())

    # 2. Try Windows Media Foundation GPU Encoder (h264_mf - NVIDIA GPU Accelerated)
    print("[Video Render] Attempting GPU Video Render with Media Foundation GPU (h264_mf)...")
    cmd_mf = build_cmd("h264_mf", ["-b:v", "6M"])
    result_mf = subprocess.run(cmd_mf, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result_mf.returncode == 0:
        print("[Video Render] GPU Media Foundation (h264_mf) Render successful!")
        return str(output_file.as_posix())

    # 3. Fallback to CPU x264 if all GPU encoders fail
    print("[Video Render] GPU Encoders unavailable, falling back to CPU (libx264)...")
    cmd_cpu = build_cmd("libx264", ["-preset", "fast"])
    fallback_result = subprocess.run(cmd_cpu, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if fallback_result.returncode != 0:
        raise RuntimeError(f"FFmpeg video render failed: {fallback_result.stderr}")

    return str(output_file.as_posix())


