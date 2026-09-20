import os
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

SETTINGS_FILE = Path("downloads") / "settings.json"
SETTINGS_LOCK = threading.RLock()

DEFAULT_SYSTEM_PROMPT = """You are a professional video script translator, narrator, and scriptwriter.
Rewrite and translate transcript content naturally for a Burmese audience while maintaining core accuracy.
Keep video subtitle segments concise, engaging, and structured cleanly."""

DEFAULT_TRANSLATION_PROMPT = """Translate each transcript segment into natural, natural Burmese.
Ensure segments are short (max 25-30 characters per cue) and end with proper punctuation (။ or ?).
Do NOT combine multiple sentences into a single long segment."""

DEFAULT_SCRIPT_REWRITE_PROMPT = """Rewrite the video narration script into an engaging viral commentary style.
Maintain curiosity, clear storytelling flow, and concise speech units."""

DEFAULT_SUBTITLE_PROMPT = """Format subtitles into short, readable cues suitable for vertical 9:16 video overlay."""


class GroqKeyEntry(BaseModel):
    id: str
    name: str
    key: str
    enabled: bool = True


class Settings(BaseModel):
    # AI Settings
    ai_provider: str = "gemini"  # "gemini" or "openai"
    groq_api_key: Optional[str] = None
    groq_api_keys: List[GroqKeyEntry] = Field(default_factory=list)
    active_groq_key_id: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    model_selection: str = "gemini-2.5-flash"
    translation_model: str = "gemini-2.5-flash"
    script_rewriting_model: str = "gemini-2.5-flash"
    viral_analysis_model: str = "gemini-2.5-flash"
    api_timeout: int = 60
    retry_count: int = 3

    # Translation Settings
    source_language: str = "Auto Detect"
    target_language: str = "Burmese"
    translation_style: str = "documentary"  # "documentary", "comedy", "sad", "emotional"
    preserve_original_meaning: bool = True
    natural_burmese_translation: bool = True
    script_rewriting_enabled: bool = True
    viral_hook_enabled: bool = True
    script_length_control: str = "concise"

    # Voice Settings
    voice_engine: str = "edge_tts"  # "edge_tts" or "voice_clone"
    voice_provider: str = "edge"    # "edge" or "voxcpm"
    voice: str = "edge-my-MM-NilarNeural"
    edge_tts_voice: str = "my-MM-NilarNeural"
    edge_tts_pitch: str = "+1.2"    # Default pitch +1.2 for Edge TTS
    edge_tts_speed: float = 1.0     # Default speed 1.0 for Edge TTS
    voice_clone_name: str = "voxcpm2-clone"
    voice_clone_pitch: str = "0"    # Default 0 / Original (NO pitch modification for Voice Clone)
    voxcpm_reference_id: Optional[str] = None
    voxcpm_reference_audio: Optional[str] = None
    voxcpm_reference_text: Optional[str] = None
    voice_language: str = "Burmese"
    custom_voice_file: Optional[str] = None
    prompt_text: Optional[str] = None
    speaking_speed: float = 1.0
    pitch: float = 1.0
    auto_voice_generation: bool = True
    audio_quality: str = "high"     # "high" (48000Hz) or "standard" (44100Hz)
    audio_normalization: bool = True
    smooth_segment_join: bool = True
    silence_cleanup: bool = True
    auto_audio_validation: bool = True

    # Subtitle Settings
    sub_font_name: Optional[str] = "Noto Sans Myanmar"
    sub_font_file: Optional[str] = None
    sub_font_size: int = 32
    sub_font_weight: str = "bold"
    sub_font_color: str = "#FFD700"  # Default TikTok educational subtitle yellow
    sub_outline: bool = True
    sub_outline_color: str = "#000000"
    sub_outline_width: int = 2
    sub_shadow: bool = True
    sub_shadow_strength: int = 1
    sub_position: str = "bottom"  # "bottom", "center", "top", "custom"
    sub_margin_v: int = 140
    sub_alignment: str = "center" # "center", "left", "right"
    sub_x: Optional[int] = None
    sub_y: Optional[int] = None
    max_chars_per_line: int = 30
    srt_generation: bool = True
    burn_subtitles: bool = True
    show_safe_area: bool = True
    sample_preview_text: str = "သစ်သားကို မြေတွင်းထဲထည့် မီးရှို့လိုက်တဲ့အခါ ဘာဖြစ်လာသလဲ။"

    # Video Settings
    aspect_ratio: str = "9:16"
    resolution: str = "1080x1920"
    fps: int = 30
    video_quality: str = "high"
    output_format: str = "mp4"
    output_folder: str = "downloads/recap"

    # GPU Settings
    gpu_acceleration: bool = True
    gpu_selection: str = "auto"
    ffmpeg_nvenc_settings: str = "-preset p4 -tune hq -rc:v vbr -cq 19"
    gpu_render_quality: str = "high"
    hardware_encoding: bool = True

    # Automation Settings
    auto_process_queue: bool = True
    auto_start_next: bool = True
    auto_retry_failed: bool = True
    max_retries: int = 3
    continue_after_error: bool = True
    auto_save_completed: bool = True
    auto_generate_srt: bool = True
    auto_clean_temp: bool = True

    # AI System Prompts
    ai_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    translation_prompt: str = DEFAULT_TRANSLATION_PROMPT
    script_rewrite_prompt: str = DEFAULT_SCRIPT_REWRITE_PROMPT
    subtitle_prompt: str = DEFAULT_SUBTITLE_PROMPT


def is_masked(value: Optional[str]) -> bool:
    """Checks if a secret value is masked (contains * or • or is blank/invalid)."""
    if not value or not isinstance(value, str):
        return False
    return "*" in value or "•" in value


def mask_secret(value: Optional[str]) -> Optional[str]:
    """Masks secret API keys for safe JSON API responses."""
    if not value or not value.strip():
        return ""
    clean = value.strip()
    if len(clean) <= 8:
        return "********"
    return "••••••••••••••••" + clean[-4:]


class SettingsManager:
    """Thread-safe persistent settings manager."""
    _instance = None
    _lock = threading.RLock()

    def __init__(self):
        self.settings = Settings()
        self.load_settings()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def load_settings(self) -> Settings:
        with SETTINGS_LOCK:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            if SETTINGS_FILE.exists():
                try:
                    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)

                        # Auto-migrate single legacy groq_api_key or GROQ_API_KEY if groq_api_keys is empty
                        if not data.get("groq_api_keys"):
                            legacy_key = data.get("groq_api_key") or os.getenv("GROQ_API_KEY")
                            if legacy_key and not is_masked(legacy_key):
                                data["groq_api_keys"] = [{
                                    "id": "key_1",
                                    "name": "Groq Key 1",
                                    "key": legacy_key,
                                    "enabled": True
                                }]

                        # Fallback for environment variables if API keys missing
                        if not data.get("groq_api_key") and os.getenv("GROQ_API_KEY"):
                            data["groq_api_key"] = os.getenv("GROQ_API_KEY")
                        if not data.get("gemini_api_key") and os.getenv("GEMINI_API_KEY"):
                            data["gemini_api_key"] = os.getenv("GEMINI_API_KEY")
                        if not data.get("openai_api_key") and os.getenv("OPENAI_API_KEY"):
                            data["openai_api_key"] = os.getenv("OPENAI_API_KEY")

                        self.settings = Settings(**data)
                except Exception as e:
                    print(f"[Settings] Warning loading {SETTINGS_FILE}: {e}. Using default settings.")
                    self.settings = Settings()
            else:
                # Load environment variable defaults if available
                env_groq = os.getenv("GROQ_API_KEY")
                if env_groq:
                    self.settings.groq_api_key = env_groq
                    self.settings.groq_api_keys = [
                        GroqKeyEntry(id="key_1", name="Groq Key 1", key=env_groq, enabled=True)
                    ]
                if os.getenv("GEMINI_API_KEY"):
                    self.settings.gemini_api_key = os.getenv("GEMINI_API_KEY")
                if os.getenv("OPENAI_API_KEY"):
                    self.settings.openai_api_key = os.getenv("OPENAI_API_KEY")
                self.save_settings(self.settings)

            return self.settings

    def save_settings(self, new_settings: Settings) -> Settings:
        with SETTINGS_LOCK:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.settings = new_settings
            try:
                data = self.settings.model_dump()
                with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"[Settings] Saved settings to {SETTINGS_FILE}")
            except Exception as e:
                print(f"[Settings] Error saving settings: {e}")
            return self.settings

    def update_settings(self, update_data: Dict[str, Any]) -> Settings:
        current_data = self.settings.model_dump()

        # Handle single masked API key updates
        for key in ["groq_api_key", "gemini_api_key", "openai_api_key"]:
            if key in update_data:
                val = update_data[key]
                if is_masked(val):
                    # Don't overwrite actual key with masked characters
                    if current_data.get(key):
                        update_data[key] = current_data.get(key)
                    else:
                        update_data.pop(key, None)

        # Handle groq_api_keys list updates preserving masked keys
        if "groq_api_keys" in update_data and isinstance(update_data["groq_api_keys"], list):
            existing_keys = {e.id: e.key for e in self.settings.groq_api_keys}
            new_entries = []
            for item in update_data["groq_api_keys"]:
                item_dict = dict(item) if hasattr(item, "model_dump") or isinstance(item, dict) else item
                if hasattr(item_dict, "model_dump"):
                    item_dict = item_dict.model_dump()
                k_id = item_dict.get("id")
                k_val = item_dict.get("key", "")
                if is_masked(k_val) and k_id in existing_keys:
                    item_dict["key"] = existing_keys[k_id]
                new_entries.append(item_dict)
            update_data["groq_api_keys"] = new_entries

        current_data.update(update_data)
        updated = Settings(**current_data)
        return self.save_settings(updated)

    def get_masked_settings_dict(self) -> Dict[str, Any]:
        data = self.settings.model_dump()
        data["groq_api_key"] = mask_secret(self.settings.groq_api_key)
        data["gemini_api_key"] = mask_secret(self.settings.gemini_api_key)
        data["openai_api_key"] = mask_secret(self.settings.openai_api_key)

        if "groq_api_keys" in data and isinstance(data["groq_api_keys"], list):
            for k in data["groq_api_keys"]:
                if isinstance(k, dict) and "key" in k:
                    k["key"] = mask_secret(k["key"])
        return data

