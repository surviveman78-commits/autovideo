import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from groq import Groq
from services.settings import is_masked


class GroqRateLimitException(Exception):
    """Raised when a Groq API Key encounters an HTTP 429 rate limit or quota condition."""
    def __init__(self, key_id: str, retry_after: float = 60.0, message: str = ""):
        self.key_id = key_id
        self.retry_after = retry_after
        self.message = message
        super().__init__(message)


class GroqAuthenticationException(Exception):
    """Exception raised when a Groq API Key fails authentication (401 / Invalid Key)."""
    def __init__(self, key_id: str, message: str = "Authentication failed"):
        self.key_id = key_id
        self.message = message
        super().__init__(self.message)


def parse_retry_after(error_obj: Exception) -> float:
    """Parses Retry-After seconds from Groq RateLimitError response headers or message."""
    default_retry = 60.0
    if hasattr(error_obj, "response") and getattr(error_obj, "response", None) is not None:
        resp = getattr(error_obj, "response")
        if hasattr(resp, "headers") and resp.headers:
            ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
            if ra:
                try:
                    return float(ra)
                except ValueError:
                    pass

    err_str = str(error_obj).lower()
    match = re.search(r"try again in ([\d\.]+)s", err_str) or re.search(r"retry after ([\d\.]+)s", err_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return default_retry


def execute_transcription(
    audio_path: str,
    raw_key: str,
    model: str = "whisper-large-v3",
    key_id: str = "unknown"
) -> Dict[str, Any]:
    """
    Executes Groq API Whisper transcription with a specific raw API key.
    Differentiates 429 rate limits from 401 authentication errors and general failures.
    """
    file_path = Path(audio_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        client = Groq(api_key=raw_key.strip())
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(file_path.name, file.read()),
                model=model,
                response_format="verbose_json",
                temperature=0.0
            )
    except Exception as e:
        err_str = str(e).lower()
        is_429 = (
            "429" in err_str or
            "rate limit" in err_str or
            "ratelimit" in err_str or
            "quota" in err_str or
            "tpm" in err_str or
            "rpm" in err_str or
            "rpd" in err_str or
            "rate_limit_exceeded" in err_str or
            type(e).__name__ == "RateLimitError"
        )
        if is_429:
            retry_seconds = parse_retry_after(e)
            raise GroqRateLimitException(
                key_id=key_id,
                retry_after=retry_seconds,
                message=f"Groq API Key '{key_id}' rate limited (429): {str(e)}"
            ) from e
        elif "401" in err_str or "authentication" in err_str or "invalid api key" in err_str:
            raise GroqAuthenticationException(
                key_id=key_id,
                message=f"Groq API Key '{key_id}' authentication failed: {str(e)}"
            ) from e
        else:
            raise e

    # Parse verbose_json response
    raw_data = transcription.model_dump() if hasattr(transcription, "model_dump") else dict(transcription)

    full_text = raw_data.get("text", "")
    segments = []

    for seg in raw_data.get("segments", []):
        segments.append({
            "id": seg.get("id"),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": seg.get("text", "").strip()
        })

    return {
        "text": full_text.strip(),
        "segments": segments,
        "language": raw_data.get("language", "en")
    }


def transcribe_audio(
    audio_path: str,
    api_key: Optional[str] = None,
    model: str = "whisper-large-v3"
) -> Dict[str, Any]:
    """
    Transcribes audio using Groq API Whisper model.
    Includes sticky active key resolution and multi-key failover loop across ready keys.
    """
    from services.groq_key_manager import GroqKeyManager
    manager = GroqKeyManager.get_instance()

    # If caller explicitly provided an unmasked raw key (e.g., custom test override), execute directly
    if api_key and not is_masked(api_key):
        return execute_transcription(audio_path, raw_key=api_key, model=model, key_id="explicit_override")

    attempted_ids: Set[str] = set()

    while True:
        # 1. Try active key first if available and not yet attempted
        key_info = None
        active_candidate = manager.get_active_key()
        if active_candidate and active_candidate[0] not in attempted_ids:
            key_info = active_candidate

        # 2. If active key is unavailable or already attempted, get next ready key
        if key_info is None:
            key_info = manager.get_next_available_key(attempted_ids)

        if key_info is None:
            # No available un-attempted keys remaining
            break

        key_id, raw_key = key_info
        attempted_ids.add(key_id)

        try:
            result = execute_transcription(audio_path, raw_key=raw_key, model=model, key_id=key_id)
            # Promote successful key to become sticky active key
            manager.set_active_key(key_id)
            return result
        except GroqRateLimitException as e:
            # Mark key as rate-limited with cooldown and continue failover loop
            manager.mark_key_rate_limited(e.key_id, e.retry_after)
            try:
                from services.queue_manager import QueueManager
                QueueManager.get_instance().broadcast_event({"type": "groq_failover", "key_id": e.key_id})
            except Exception:
                pass
            continue
        except GroqAuthenticationException as e:
            # Mark key as temporarily invalid/cooldown so active key switches to next ready key
            manager.mark_key_rate_limited(e.key_id, retry_after=300.0)
            print(f"[Transcriber] Groq key '{e.key_id}' authentication failed. Attempting next ready key...")
            try:
                from services.queue_manager import QueueManager
                QueueManager.get_instance().broadcast_event({"type": "groq_failover", "key_id": e.key_id})
            except Exception:
                pass
            continue

    # If all configured keys fail due to rate limiting, authentication error, or none exist
    try:
        from services.queue_manager import QueueManager
        QueueManager.get_instance().broadcast_event({"type": "groq_all_failed"})
    except Exception:
        pass
    raise RuntimeError("All configured Groq API keys are currently rate-limited, invalid, or unavailable.")
