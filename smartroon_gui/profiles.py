"""Storage utilities for GUI profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from smartroon.logging_utils import get_logger


@dataclass
class ProfileData:
    """Serializable profile payload for the GUI."""

    filter_path: str | None
    sample_rate: int | None
    target_tp: float
    oversample: int
    gain_mode: str
    ear_left_db: float | None
    ear_right_db: float | None
    ear_offset_db: float | None
    output_path: str | None = None
    keep_structure: bool | None = None
    suffix: str | None = None
    copy_metadata: bool | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProfileData":
        """Create :class:`ProfileData` from a mapping with safe defaults."""

        def _int_or_none(value: Any) -> int | None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def _float_or_none(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        return cls(
            filter_path=payload.get("filter_path"),
            sample_rate=_int_or_none(payload.get("sample_rate")),
            target_tp=float(payload.get("target_tp", -0.1)),
            oversample=int(payload.get("oversample", 4)),
            gain_mode=str(payload.get("gain_mode", "down-only")),
            ear_left_db=_float_or_none(payload.get("ear_left_db")),
            ear_right_db=_float_or_none(payload.get("ear_right_db")),
            ear_offset_db=_float_or_none(payload.get("ear_offset_db")),
            output_path=payload.get("output_path"),
            keep_structure=payload.get("keep_structure"),
            suffix=payload.get("suffix"),
            copy_metadata=payload.get("copy_metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the profile data to a JSON-serializable mapping."""

        return {
            "filter_path": self.filter_path,
            "sample_rate": self.sample_rate,
            "target_tp": self.target_tp,
            "oversample": self.oversample,
            "gain_mode": self.gain_mode,
            "ear_left_db": self.ear_left_db,
            "ear_right_db": self.ear_right_db,
            "ear_offset_db": self.ear_offset_db,
            "output_path": self.output_path,
            "keep_structure": self.keep_structure,
            "suffix": self.suffix,
            "copy_metadata": self.copy_metadata,
        }


class ProfilesStore:
    """Disk persistence helper for GUI profiles."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or Path.home() / ".rloc_gui"
        self._profiles_path = self._base_dir / "profiles.json"
        self._logger = get_logger(__name__)

    @property
    def path(self) -> Path:
        """Return the path to the profiles JSON file."""

        return self._profiles_path

    def load_profiles(self) -> tuple[str | None, dict[str, ProfileData]]:
        """Load stored profiles and the active profile name."""

        if not self._profiles_path.exists():
            return None, {}

        try:
            raw_content = self._profiles_path.read_text(encoding="utf-8")
            payload = json.loads(raw_content)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Не удалось загрузить профили: %s", exc)
            return None, {}

        profiles_payload = payload.get("profiles", {}) if isinstance(payload, dict) else {}
        profiles: dict[str, ProfileData] = {}
        for name, profile_payload in profiles_payload.items():
            try:
                profiles[name] = ProfileData.from_dict(profile_payload)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("Пропускаем профиль %s: %s", name, exc)

        active_profile = payload.get("active_profile") if isinstance(payload, dict) else None
        if active_profile not in profiles:
            active_profile = None
        return active_profile, profiles

    def persist_profiles(self, profiles: Mapping[str, ProfileData], active_profile: str | None) -> None:
        """Persist profiles and the active selection to disk."""

        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            active_name = active_profile if active_profile in profiles else None
            payload = {
                "active_profile": active_name,
                "profiles": {name: profile.to_dict() for name, profile in profiles.items()},
            }
            serialized = json.dumps(payload, ensure_ascii=False, indent=2)
            self._profiles_path.write_text(serialized, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Не удалось сохранить профили: %s", exc)
