"""Shipped defaults plus user cache keyed by (op, fingerprint, triton_ver, kernel_rev, dtype, bucket)."""

import json
import os
from pathlib import Path
from typing import Any

CacheKey = tuple[str, str, str, int, str, str]


def get_current_triton_version() -> str:
    """Retrieve active Triton version string or 'unknown'."""
    try:
        import triton

        return str(getattr(triton, "__version__", "unknown"))
    except Exception:
        return "unknown"


def get_current_kernel_rev(op: str) -> int:
    """Retrieve KERNEL_REV integer from the op's _core kernel module."""
    from stackkernels._opt.tune import get_kernel_rev

    return get_kernel_rev(op)


def normalize_bucket(bucket: Any) -> str:
    """Normalize bucket/shape into a canonical string key."""
    if isinstance(bucket, (list, tuple)):
        return "x".join(str(x) for x in bucket)
    return str(bucket)


def make_cache_key(
    op: str,
    arch: str,
    dtype: str,
    bucket: Any,
    triton_version: str | None = None,
    kernel_rev: int | None = None,
) -> CacheKey:
    """Build a normalized 6-tuple cache key:
    (op, device_fingerprint, triton_version, kernel_rev, dtype, shape_bucket)
    """
    op_clean = str(op).strip().lower()
    t_ver = triton_version if triton_version is not None else get_current_triton_version()
    k_rev = int(kernel_rev) if kernel_rev is not None else get_current_kernel_rev(op_clean)
    return (
        op_clean,
        str(arch).strip().lower(),
        str(t_ver).strip(),
        k_rev,
        str(dtype).strip().lower(),
        normalize_bucket(bucket),
    )


def serialize_key(key: CacheKey) -> str:
    return ":".join(str(x) for x in key)


def deserialize_key(raw: str) -> CacheKey | None:
    parts = raw.split(":")
    if len(parts) == 6:
        try:
            return (
                parts[0].lower(),
                parts[1].lower(),
                parts[2],
                int(parts[3]),
                parts[4].lower(),
                parts[5],
            )
        except ValueError:
            return None
    elif len(parts) == 4:
        # Backward compatibility for legacy 4-tuple: (op, arch, dtype, bucket)
        return (
            parts[0].lower(),
            parts[1].lower(),
            "unknown",
            1,
            parts[2].lower(),
            parts[3],
        )
    return None


class TuningCache:
    """Cache for autotuned kernel configurations.

    Maintains in-memory lookup over shipped defaults and user-tuned entries,
    keyed by (op, device_fingerprint, triton_version, kernel_rev, dtype, shape_bucket).
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
        self._metadata_cache: dict[CacheKey, dict[str, Any]] = {}

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
                if "config" in v and isinstance(v["config"], dict):
                    target_dict[key] = dict(v["config"])
                    if "metadata" in v and isinstance(v["metadata"], dict):
                        self._metadata_cache[key] = dict(v["metadata"])
                else:
                    target_dict[key] = dict(v)

    def save_user_cache(self, file_path: Path | str | None = None) -> Path:
        """Persist user cache to JSON."""
        out_path = Path(file_path) if file_path else (self.cache_dir / "user_cache.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {}
        for k, v in self._user_cache.items():
            meta = self._metadata_cache.get(k)
            if meta:
                payload[serialize_key(k)] = {"config": v, "metadata": meta}
            else:
                payload[serialize_key(k)] = v

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out_path

    def get(
        self,
        op: str,
        arch: str,
        dtype: str,
        bucket: Any,
        triton_version: str | None = None,
        kernel_rev: int | None = None,
    ) -> dict[str, Any] | None:
        """Lookup config in user cache first, then shipped cache."""
        key = make_cache_key(op, arch, dtype, bucket, triton_version, kernel_rev)
        if key in self._user_cache:
            return dict(self._user_cache[key])
        if key in self._shipped_cache:
            return dict(self._shipped_cache[key])
        return None

    def get_user(
        self,
        op: str,
        arch: str,
        dtype: str,
        bucket: Any,
        triton_version: str | None = None,
        kernel_rev: int | None = None,
    ) -> dict[str, Any] | None:
        """Lookup config exclusively in user cache."""
        key = make_cache_key(op, arch, dtype, bucket, triton_version, kernel_rev)
        if key in self._user_cache:
            return dict(self._user_cache[key])
        return None

    def get_shipped(
        self,
        op: str,
        arch: str,
        dtype: str,
        bucket: Any,
        triton_version: str | None = None,
        kernel_rev: int | None = None,
    ) -> dict[str, Any] | None:
        """Lookup config exclusively in shipped defaults."""
        key = make_cache_key(op, arch, dtype, bucket, triton_version, kernel_rev)
        if key in self._shipped_cache:
            return dict(self._shipped_cache[key])
        return None

    def get_metadata(
        self,
        op: str,
        arch: str,
        dtype: str,
        bucket: Any,
        triton_version: str | None = None,
        kernel_rev: int | None = None,
    ) -> dict[str, Any] | None:
        """Lookup metadata for a cached entry."""
        key = make_cache_key(op, arch, dtype, bucket, triton_version, kernel_rev)
        return self._metadata_cache.get(key)

    def set(
        self,
        op: str,
        arch: str,
        dtype: str,
        bucket: Any,
        config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        triton_version: str | None = None,
        kernel_rev: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Set a configuration in the user cache."""
        key = make_cache_key(op, arch, dtype, bucket, triton_version, kernel_rev)
        self._user_cache[key] = dict(config)
        if metadata:
            self._metadata_cache[key] = dict(metadata)

    def clear(self) -> None:
        """Clear user cache entries."""
        self._user_cache.clear()
        self._metadata_cache.clear()

    def clear_all(self) -> None:
        """Clear both user and shipped cache entries."""
        self._user_cache.clear()
        self._shipped_cache.clear()
        self._metadata_cache.clear()


_GLOBAL_CACHE = TuningCache()


def get_global_cache() -> TuningCache:
    """Return the global TuningCache instance."""
    return _GLOBAL_CACHE


def get_cached_config(
    op: str,
    arch: str,
    dtype: str,
    bucket: Any,
    triton_version: str | None = None,
    kernel_rev: int | None = None,
) -> dict[str, Any] | None:
    """Look up a configuration from the global cache."""
    return _GLOBAL_CACHE.get(op, arch, dtype, bucket, triton_version, kernel_rev)


def set_cached_config(
    op: str,
    arch: str,
    dtype: str,
    bucket: Any,
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    triton_version: str | None = None,
    kernel_rev: int | None = None,
) -> None:
    """Set a configuration in the global cache."""
    _GLOBAL_CACHE.set(
        op,
        arch,
        dtype,
        bucket,
        config,
        metadata=metadata,
        triton_version=triton_version,
        kernel_rev=kernel_rev,
    )


def clear_cache() -> None:
    """Clear the global user cache."""
    _GLOBAL_CACHE.clear()


__all__ = [
    "CacheKey",
    "TuningCache",
    "clear_cache",
    "deserialize_key",
    "get_cached_config",
    "get_current_kernel_rev",
    "get_current_triton_version",
    "get_global_cache",
    "make_cache_key",
    "normalize_bucket",
    "serialize_key",
    "set_cached_config",
]
