import os
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

# Burmese sentence punctuation & grammar markers
BURMESE_FULL_STOPS = ["။", "?", "!", "\n"]
BURMESE_COMMAS = ["၊", ","]
GRAMMAR_ENDINGS = [
    "တယ်", "ပါတယ်", "ပါ", "မယ်", "ဘူး", "လား", "မလား", 
    "နိုင်ပါတယ်", "ဖြစ်ပါတယ်", "လုပ်ပါတယ်", "လုပ်မယ်", "မဖြစ်ဘူး"
]


def estimate_burmese_speech_duration(text: str) -> float:
    """
    Estimates spoken speech duration in seconds for Burmese and multi-lingual text.
    Based on average Burmese speech rate (~13 Burmese chars/sec) + punctuation pauses.
    Target: ~20 seconds of speech per chunk (approx 230-270 Burmese characters).
    """
    if not text:
        return 0.0

    clean_text = text.strip()
    char_count = len(clean_text)

    # Base duration from character count (~13 chars/sec)
    base_duration = char_count / 13.0

    # Add realistic pauses for punctuation
    full_stop_count = sum(clean_text.count(p) for p in ["။", "?", "!"])
    comma_count = sum(clean_text.count(p) for p in ["၊", ","])

    total_duration = base_duration + (full_stop_count * 0.4) + (comma_count * 0.2)
    return max(0.5, total_duration)


def split_text_into_grammatical_sentences(text: str) -> List[str]:
    """
    Splits continuous text into sentence units using Burmese punctuation (။, ?, !, \n).
    Preserves trailing punctuation on each sentence.
    """
    if not text:
        return []

    pattern = r'([^။?!.\n]+[။?!.\n]*)'
    matches = re.findall(pattern, text)

    sentences = [m.strip() for m in matches if m.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]

    return sentences


def is_safe_burmese_split_point(text: str, idx: int) -> bool:
    """
    Checks if splitting text at character index `idx` is safe in Burmese.
    Returns False if `idx` splits inside a Burmese word, subjoined consonant (\u1039),
    or diacritic tone marker.
    """
    if idx <= 0 or idx >= len(text):
        return True

    prev_char = text[idx - 1]
    next_char = text[idx]

    # 1. Do NOT split after Virama (subjoined consonant marker \u1039)
    if prev_char == '\u1039':
        return False

    # 2. Do NOT split before diacritics / tone marks (\u102B-\u103E)
    if '\u102B' <= next_char <= '\u103E':
        return False

    # 3. Do NOT split between base consonant and tone markers
    if '\u1000' <= prev_char <= '\u1020' and '\u1000' <= next_char <= '\u103E':
        return False

    return True


def split_long_sentence_into_clauses(sentence: str, max_est_seconds: float = 23.0) -> List[str]:
    """
    If a single sentence is extremely long (>23s estimated duration),
    splits it at internal clause boundaries (၊, ,, natural pauses)
    while preserving complete words and grammatical units.
    """
    if estimate_burmese_speech_duration(sentence) <= max_est_seconds:
        return [sentence]

    clauses = []
    # Split by Burmese comma (၊) or English comma (,)
    parts = re.split(r'([၊,])', sentence)

    current_clause = ""
    for i in range(0, len(parts), 2):
        part = parts[i]
        punc = parts[i+1] if i + 1 < len(parts) else ""
        piece = part + punc

        if estimate_burmese_speech_duration(current_clause + piece) <= max_est_seconds:
            current_clause += piece
        else:
            if current_clause.strip():
                clauses.append(current_clause.strip())
            current_clause = piece

    if current_clause.strip():
        clauses.append(current_clause.strip())

    return clauses if clauses else [sentence]


def chunk_text_for_voxcpm(
    segments: List[Dict[str, Any]],
    target_duration_sec: float = 12.0,
    min_duration_sec: float = 8.0,
    max_duration_sec: float = 14.0
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Intelligently chunks translated text into ~10-14 second natural Burmese speech blocks
    for VoxCPM2 voice cloning.

    Prioritizes natural sentence integrity (။, ?, !) and grammatical boundaries while keeping
    chunk sizes optimal for VoxCPM2 to synthesize 100% of text without dropping leading sentences.
    Target: ~10-14 seconds per chunk.
    """
    # 1. Flatten all translated segments into ordered sentence units
    all_units = []
    for seg in segments:
        raw_text = seg.get("text", "").strip()
        if not raw_text:
            continue

        sentences = split_text_into_grammatical_sentences(raw_text)
        for s in sentences:
            sub_clauses = split_long_sentence_into_clauses(s, max_est_seconds=max_duration_sec)
            for clause in sub_clauses:
                all_units.append({
                    "text": clause,
                    "orig_start": seg.get("start", 0.0),
                    "orig_end": seg.get("end", 0.0)
                })

    if not all_units:
        return [], []

    # 2. Check if total estimated duration <= 12.0s. If so, keep as single chunk!
    total_est_duration = sum(estimate_burmese_speech_duration(u["text"]) for u in all_units)
    if total_est_duration <= 12.0:
        chunk_text = " ".join(u["text"] for u in all_units).strip()
        single_chunk = [{
            "chunk_id": 0,
            "text": chunk_text,
            "est_duration": round(total_est_duration, 2),
            "units": all_units
        }]
        validate_voxcpm_chunks(segments, single_chunk)
        return single_chunk, all_units

    # 3. Build progressive ~20s-25s speech chunks for long narrations (>30s)
    chunks = []
    current_chunk_units = []
    current_est = 0.0

    for unit in all_units:
        unit_text = unit["text"]
        unit_est = estimate_burmese_speech_duration(unit_text)

        # Check if adding this unit stays within natural ~20s range (up to max_duration_sec)
        if current_est + unit_est <= max_duration_sec:
            current_chunk_units.append(unit)
            current_est += unit_est

            # If we've reached near target (~16-20s) and unit ends with sentence punctuation (။, ?, !), close chunk!
            if current_est >= min_duration_sec and any(unit_text.endswith(p) for p in ["။", "?", "!"]):
                chunk_text = " ".join(u["text"] for u in current_chunk_units).strip()
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": chunk_text,
                    "est_duration": round(current_est, 2),
                    "units": list(current_chunk_units)
                })
                current_chunk_units = []
                current_est = 0.0
        else:
            # Adding unit exceeds max_duration_sec. Close current chunk first if non-empty.
            if current_chunk_units:
                chunk_text = " ".join(u["text"] for u in current_chunk_units).strip()
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": chunk_text,
                    "est_duration": round(current_est, 2),
                    "units": list(current_chunk_units)
                })
                current_chunk_units = [unit]
                current_est = unit_est
            else:
                # Unit itself is longer than max_duration_sec
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": unit_text,
                    "est_duration": round(unit_est, 2),
                    "units": [unit]
                })
                current_chunk_units = []
                current_est = 0.0

    if current_chunk_units:
        chunk_text = " ".join(u["text"] for u in current_chunk_units).strip()
        chunks.append({
            "chunk_id": len(chunks),
            "text": chunk_text,
            "est_duration": round(current_est, 2),
            "units": list(current_chunk_units)
        })

    # 3. Validation: Ensure no text loss
    validate_voxcpm_chunks(segments, chunks)

    return chunks, all_units


def validate_voxcpm_chunks(original_segments: List[Dict[str, Any]], chunks: List[Dict[str, Any]]):
    """
    Validates that concatenating all chunks recovers 100% of the original translated text
    without missing words, duplications, or skips.
    """
    orig_text = "".join(s.get("text", "").strip() for s in original_segments if s.get("text"))
    chunk_combined = "".join(c.get("text", "").strip() for c in chunks)

    norm_orig = re.sub(r'\s+', '', orig_text)
    norm_chunk = re.sub(r'\s+', '', chunk_combined)

    if norm_orig != norm_chunk:
        print(f"[VoxCPM Validation Warning] Character mismatch! Original ({len(norm_orig)} chars) vs Chunks ({len(norm_chunk)} chars)")
    else:
        print(f"[VoxCPM Validation] 100% Text Integrity Verified! ({len(chunks)} chunks created from {len(original_segments)} segments)")


async def process_voxcpm_long_text_pipeline(
    translated_segments: List[Dict[str, Any]],
    output_voice_dir: Path,
    output_master_audio_path: Path,
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    target_sample_rate: int = 48000,
    retry_count: int = 3
) -> List[Dict[str, Any]]:
    """
    Full VoxCPM2 Voice Clone long-text pipeline:
    1. Divide translated text into ~20s Burmese speech chunks.
    2. Generate each chunk sequentially using the SAME VoxCPM2 reference voice & model parameters.
    3. Merge generated audio smoothly without boundary clicks/pops or pitch changes.
    4. Calculate actual chunk durations and update segment timestamps for exact subtitle sync.
    Returns: List[Dict[str, Any]] updated segments with accurate timestamps & 'audio_path'.
    """
    from services.tts import generate_voxcpm_audio_sync

    output_voice_dir.mkdir(parents=True, exist_ok=True)
    output_master_audio_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Chunk text into ~20s natural Burmese speech blocks
    chunks, _ = chunk_text_for_voxcpm(translated_segments)
    if not chunks:
        raise ValueError("No text available for VoxCPM2 voice generation.")

    loop = asyncio.get_event_loop()

    # Step 2: Sequential generation of each chunk with SAME voice parameters
    generated_chunk_files = []

    for idx, chunk in enumerate(chunks):
        chunk_text = chunk["text"]
        chunk_wav_path = str(output_voice_dir / f"voxcpm_chunk_{idx:03d}.wav")

        # Resume Checkpoint: Skip re-generating chunk if valid wav file already exists from previous attempt
        if os.path.exists(chunk_wav_path) and os.path.getsize(chunk_wav_path) > 1000:
            print(f"[VoxCPM2 Pipeline] Found existing chunk audio for Chunk {idx+1}/{len(chunks)} ({chunk_wav_path}). Skipping re-generation!")
            generated_chunk_files.append((chunk, chunk_wav_path))
            continue

        try:
            print(f"[VoxCPM2 Pipeline] Generating Chunk {idx+1}/{len(chunks)} (~{chunk['est_duration']:.1f}s est)...")
        except Exception:
            pass

        success = False
        last_error = None

        for attempt in range(1, retry_count + 1):
            try:
                await loop.run_in_executor(
                    None,
                    generate_voxcpm_audio_sync,
                    chunk_text,
                    chunk_wav_path,
                    prompt_wav_path,
                    prompt_text
                )

                if os.path.exists(chunk_wav_path) and os.path.getsize(chunk_wav_path) > 100:
                    success = True
                    break
                else:
                    raise ValueError(f"Generated VoxCPM2 chunk audio is empty or invalid: {chunk_wav_path}")
            except Exception as e:
                last_error = e
                print(f"[VoxCPM2 Chunk {idx+1}/{len(chunks)}] Attempt {attempt}/{retry_count} failed: {e}")
                await asyncio.sleep(1.0)

        if not success:
            error_msg = f"VoxCPM2 failed while generating narration segment {idx+1}/{len(chunks)}: {last_error}"
            print(f"[VoxCPM2 Pipeline Error] {error_msg}")
            raise RuntimeError(error_msg)

        generated_chunk_files.append((chunk, chunk_wav_path))

    # Step 3: Seamless audio merging
    master_audio = AudioSegment.silent(duration=0, frame_rate=target_sample_rate).set_channels(2)
    timed_segments = []

    for idx, (chunk, wav_file) in enumerate(generated_chunk_files):
        try:
            raw_seg = AudioSegment.from_file(wav_file)
        except Exception as e:
            raise RuntimeError(f"Failed to read VoxCPM2 generated audio chunk {wav_file}: {e}")

        # Normalize format to target sample rate, stereo, 16-bit PCM
        if raw_seg.frame_rate != target_sample_rate:
            raw_seg = raw_seg.set_frame_rate(target_sample_rate)
        if raw_seg.channels != 2:
            raw_seg = raw_seg.set_channels(2)
        if raw_seg.sample_width != 2:
            raw_seg = raw_seg.set_sample_width(2)

        # Trim subtle accidental leading/trailing silence (> -42dB) leaving 30ms padding
        start_trim = detect_leading_silence(raw_seg, silence_threshold=-42.0)
        end_trim = detect_leading_silence(raw_seg.reverse(), silence_threshold=-42.0)
        start_pos = max(0, start_trim - 30)
        end_pos = max(start_pos + 50, len(raw_seg) - max(0, end_trim - 50))
        clean_seg = raw_seg[start_pos:end_pos] if len(raw_seg[start_pos:end_pos]) > 50 else raw_seg

        # Record exact start time in master track
        chunk_start_sec = len(master_audio) / 1000.0

        # Append with micro-crossfade (15ms) to eliminate boundary pops/clicks without artificial silence
        if len(master_audio) > 0:
            master_audio = master_audio.append(clean_seg, crossfade=min(15, len(master_audio), len(clean_seg)))
        else:
            master_audio += clean_seg

        # Record exact end time in master track
        chunk_end_sec = len(master_audio) / 1000.0

        # Create timed segment cues for subtitles corresponding to units in this chunk
        chunk_units = chunk.get("units", [])
        if chunk_units:
            chunk_duration = max(0.5, chunk_end_sec - chunk_start_sec)
            total_chars = sum(len(u["text"]) for u in chunk_units) or 1

            curr_unit_start = chunk_start_sec
            for u_idx, u in enumerate(chunk_units):
                u_ratio = len(u["text"]) / total_chars
                u_dur = chunk_duration * u_ratio
                u_end = curr_unit_start + u_dur if u_idx < len(chunk_units) - 1 else chunk_end_sec

                timed_segments.append({
                    "text": u["text"],
                    "start": round(curr_unit_start, 3),
                    "end": round(u_end, 3),
                    "audio_path": wav_file
                })
                curr_unit_start = u_end
        else:
            timed_segments.append({
                "text": chunk["text"],
                "start": round(chunk_start_sec, 3),
                "end": round(chunk_end_sec, 3),
                "audio_path": wav_file
            })

    # Final master audio normalization
    if len(master_audio) > 0:
        master_audio = master_audio.normalize(headroom=1.5)

    # Export master audio file
    out_format = "wav" if str(output_master_audio_path).endswith(".wav") else "mp3"
    master_audio.export(str(output_master_audio_path), format=out_format, bitrate="320k")

    print(f"[VoxCPM2 Pipeline] Master audio generated successfully! Duration: {len(master_audio)/1000.0:.2f}s | Path: {output_master_audio_path}")

    return timed_segments
