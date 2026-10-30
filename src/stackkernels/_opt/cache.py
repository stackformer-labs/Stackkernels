"""shipped defaults plus user cache"""

import json
import os
from pathlib import Path
from typing import Any

CacheKey = tuple[str, str, str, str]


def normalize_bucket(bucket: Any) -> str:
    """Normalize bucket/shape into a canonical string key."""
    if isinstance(bucket, (list, tuple)):
        return "x".join(str(x) for x in bucket)
    return str(bucket)


def make_cache_key(op: str, arch: str, dtype: str, bucket: Any) -> CacheKey:
    """Build a normalized 4-tuple cache key."""
    return (
        str(op).strip().lower(),
        str(arch).strip().lower(),
        str(dtype).strip().lower(),
        normalize_bucket(bucket),
    )


def serialize_key(key: CacheKey) -> str:
    return ":".join(key)


def deserialize_key(raw: str) -> CacheKey | None:
    parts = raw.split(":")
    if len(parts) == 4:
        return (parts[0].lower(), parts[1].lower(), parts[2].lower(), parts[3])
    return None


class TuningCache:
    """Cache for autotuned kernel configurations.

    Maintains in-memory lookup over shipped defaults and user-tuned entries,
    keyed by (op, arch, dtype, bucket).
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        load_shipped: bool = True,
    ) -> None:
        if cache_dir is None:
            base_dir = os.environ.get("STACKKERNELS_CACHE_DIR")
            if base_dir:
                self.cache_dir = Path(base_dir)
            else:
                self.cache_dir = Path.home() / ".cache" / "stackkernels"
        else:
            self.cache_dir = Path(cache_dir)

        self._shipped_cache: dict[CacheKey, dict[str, Any]] = {}
        self._user_cache: dict[CacheKey, dict[str, Any]] = {}

        if load_shipped:
            self.load_shipped()

    @property
    def shipped_dir(self) -> Path:
        return Path(__file__).parent / "tuned"

    def load_shipped(self) -> None:
        """Load all shipped JSON configuration files from _opt/tuned/."""
        if not self.shipped_dir.exists():
            return
        for json_file in self.shipped_dir.glob("*.json"):
            self.load_file(json_file, target=self._shipped_cache)

    def load_file(
        self,
        path: Path | str,
        target: dict[CacheKey, dict[str, Any]] | None = None,
    ) -> None:
        """Load JSON configurations into the cache."""
        target_dict = self._user_cache if target is None else target
        file_path = Path(path)
        if not file_path.is_file():
            return

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return

        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            key = deserialize_key(k)
            if key is not None:
                target_dict[key] = dict(v)

    def save_user_cache(self, file_path: Path | str | None = None) -> Path:
        """Persist user cache to JSON."""
        out_path = Path(file_path) if file_path else (self.cache_dir / "user_cache.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {serialize_key(k): v for k, v in self._user_cache.items()}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out_path

    def get(self, op: str, arch: str, dtype: str, bucket: Any) -> dict[str, Any] | None:
        """Lookup config in user cache first, then shipped cache."""
        key = make_cache_key(op, arch, dtype, bucket)
        if key in self._user_cache:
            return dict(self._user_cache[key])
        if key in self._shipped_cache:
            return dict(self._shipped_cache[key])
        return None

    def set(
        self,
        op: str,
        arch: str,
        dtype: str,
        bucket: Any,
        config: dict[str, Any],
    ) -> None:
        """Set a configuration in the user cache."""
        key = make_cache_key(op, arch, dtype, bucket)
        self._user_cache[key] = dict(config)

    def clear(self) -> None:
        """Clear user cache entries."""
        self._user_cache.clear()

    def clear_all(self) -> None:
        """Clear both user and shipped cache entries."""
        self._user_cache.clear()
        self._shipped_cache.clear()


_GLOBAL_CACHE = TuningCache()


def get_global_cache() -> TuningCache:
    """Return the global TuningCache instance."""
    return _GLOBAL_CACHE


def get_cached_config(
    op: str,
    arch: str,
    dtype: str,
    bucket: Any,
) -> dict[str, Any] | None:
    """Look up a configuration from the global cache."""
    return _GLOBAL_CACHE.get(op, arch, dtype, bucket)


def set_cached_config(
    op: str,
    arch: str,
    dtype: str,
    bucket: Any,
    config: dict[str, Any],
) -> None:
    """Set a configuration in the global cache."""
    _GLOBAL_CACHE.set(op, arch, dtype, bucket, config)


def clear_cache() -> None:
    """Clear the global user cache."""
    _GLOBAL_CACHE.clear()
