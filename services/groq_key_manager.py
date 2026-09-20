import time
import threading
from typing import List, Dict, Any, Optional, Set, Tuple
from services.settings import SettingsManager, GroqKeyEntry, mask_secret, is_masked


class GroqKeyManager:
    """
    Thread-safe Singleton Groq API Key Manager.
    Manages sticky active key, rate-limit cooldown tracking, failover rotation,
    and status reporting for the Multi-Groq-API-Key system.
    """
    _instance = None
    _lock = threading.RLock()

    def __init__(self):
        self._active_key_id: Optional[str] = None
        self._cooldowns: Dict[str, float] = {}  # key_id -> cooldown_until timestamp

    @classmethod
    def get_instance(cls) -> "GroqKeyManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _get_configured_entries(self) -> List[GroqKeyEntry]:
        settings = SettingsManager.get_instance().load_settings()
        return settings.groq_api_keys or []

    def get_active_key(self) -> Optional[Tuple[str, str]]:
        """
        Returns (key_id, raw_api_key) for the current sticky active key.
        If current active key is missing, disabled, or on cooldown, selects the first ready key.
        """
        with self._lock:
            entries = self._get_configured_entries()
            now = time.time()

            # If active_key_id not in memory, restore from saved settings
            if not self._active_key_id:
                saved_active = SettingsManager.get_instance().load_settings().active_groq_key_id
                if saved_active:
                    self._active_key_id = saved_active

            # Check existing sticky active key if set
            if self._active_key_id:
                active_entry = next((e for e in entries if e.id == self._active_key_id), None)
                if active_entry and active_entry.enabled and active_entry.key and not is_masked(active_entry.key):
                    cooldown_until = self._cooldowns.get(active_entry.id, 0)
                    if now >= cooldown_until:
                        return (active_entry.id, active_entry.key)

            # Active key unavailable or not set -> find first ready key
            for entry in entries:
                if entry.enabled and entry.key and not is_masked(entry.key):
                    cooldown_until = self._cooldowns.get(entry.id, 0)
                    if now >= cooldown_until:
                        self.set_active_key(entry.id)
                        return (entry.id, entry.key)

            # Fallback 1: check legacy single key in settings if available
            settings = SettingsManager.get_instance().load_settings()
            if settings.groq_api_key and not is_masked(settings.groq_api_key):
                cooldown_until = self._cooldowns.get("legacy_key", 0)
                if now >= cooldown_until:
                    return ("legacy_key", settings.groq_api_key)

            return None

    def get_next_available_key(self, attempted_ids: Set[str]) -> Optional[Tuple[str, str]]:
        """
        Returns (key_id, raw_api_key) for the next ready key NOT in attempted_ids.
        Does NOT alter active_key_id until explicit set_active_key call upon success.
        """
        with self._lock:
            entries = self._get_configured_entries()
            now = time.time()

            for entry in entries:
                if entry.id in attempted_ids:
                    continue
                if not entry.enabled or not entry.key or is_masked(entry.key):
                    continue
                cooldown_until = self._cooldowns.get(entry.id, 0)
                if now < cooldown_until:
                    continue
                return (entry.id, entry.key)

            # Check legacy single key fallback if not attempted
            if "legacy_key" not in attempted_ids:
                settings = SettingsManager.get_instance().load_settings()
                if settings.groq_api_key and not is_masked(settings.groq_api_key):
                    cooldown_until = self._cooldowns.get("legacy_key", 0)
                    if now >= cooldown_until:
                        return ("legacy_key", settings.groq_api_key)

            return None

    def mark_key_rate_limited(self, key_id: str, retry_after: float = 60.0):
        """
        Marks a key as RATE_LIMITED with a cooldown timestamp.
        """
        with self._lock:
            cooldown_seconds = max(1.0, float(retry_after) if retry_after else 60.0)
            self._cooldowns[key_id] = time.time() + cooldown_seconds
            print(f"[GroqKeyManager] Key '{key_id}' rate limited. Cooldown for {cooldown_seconds:.1f}s.")

            # If the rate-limited key was active, reset active_key_id so next lookup selects a ready key
            if self._active_key_id == key_id:
                self._active_key_id = None

    def set_active_key(self, key_id: str):
        """
        Promotes successful key_id to become the new sticky ACTIVE key and persists it.
        """
        with self._lock:
            if self._active_key_id != key_id:
                self._active_key_id = key_id
                try:
                    SettingsManager.get_instance().update_settings({"active_groq_key_id": key_id})
                except Exception as e:
                    print(f"[GroqKeyManager] Warning persisting active key: {e}")

    def get_keys_status_for_ui(self) -> List[Dict[str, Any]]:
        """
        Returns detailed UI status list for all configured Groq keys.
        Status types: ACTIVE, READY, RATE_LIMITED, DISABLED.
        """
        with self._lock:
            entries = self._get_configured_entries()
            now = time.time()
            active_info = self.get_active_key()
            current_active_id = active_info[0] if active_info else None

            result = []
            for entry in entries:
                cooldown_until = self._cooldowns.get(entry.id, 0)
                cooldown_remaining = max(0, int(cooldown_until - now))

                if not entry.enabled:
                    status = "DISABLED"
                elif cooldown_remaining > 0:
                    status = "RATE_LIMITED"
                elif entry.id == current_active_id:
                    status = "ACTIVE"
                else:
                    status = "READY"

                result.append({
                    "id": entry.id,
                    "name": entry.name,
                    "masked_key": mask_secret(entry.key),
                    "enabled": entry.enabled,
                    "status": status,
                    "cooldown_seconds": cooldown_remaining
                })

            return result

    def reset_cooldowns(self):
        """Clears all key cooldowns."""
        with self._lock:
            self._cooldowns.clear()
