"""Single-call redesign engine: provider fallback chain plus a persistent disk cache.

Every generation costs credits or free-tier GPU quota, so the engine:
- makes exactly one edit call per provider attempt (no multi-pass, no automatic retries),
- caches every successful result on disk keyed by (image, prompt, tier, seed), so repeated
  runs on the same photo are free, even across backend restarts,
- supports REDESIGN_OFFLINE=1, which serves only cached results and never calls a provider.

Providers are tried in the order given by IMAGE_PROVIDERS (default "hf_space,fal"); providers
that are not configured (e.g. fal without FAL_KEY) are skipped.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image

from .base import ImageEditingProvider
from .gemini_provider import GeminiImageEditingProvider
from .kontext_provider import FalKontextProvider, HuggingFaceSpaceKontextProvider
from .schemas import ValidationReport
from .validation import validate_image_output

logger = logging.getLogger("resilicity.redesign_engine")

PROVIDER_REGISTRY = {
    "hf_space": HuggingFaceSpaceKontextProvider,
    "fal": FalKontextProvider,
    "gemini": GeminiImageEditingProvider,
}

DEFAULT_CACHE_DIR = r"D:\Resillicity\cache\redesign"


@dataclass
class GenerationResult:
    image: Optional[Image.Image]
    provider: str
    model: str
    error: Optional[str] = None
    cached: bool = False
    validation: Optional[ValidationReport] = None


def _model_label(provider: ImageEditingProvider, quality_tier: str) -> str:
    if isinstance(provider, GeminiImageEditingProvider):
        return provider.get_model_for_tier(quality_tier)
    if isinstance(provider, FalKontextProvider):
        return provider.model_for_tier(quality_tier)
    return getattr(provider, "model_name", type(provider).__name__)


def _is_configured(provider: ImageEditingProvider) -> bool:
    if isinstance(provider, GeminiImageEditingProvider):
        return bool(provider.api_key)
    return provider.is_configured()


class RedesignEngine:
    def __init__(self, provider_names: Optional[List[str]] = None, cache_dir: Optional[str] = None):
        names = provider_names or [
            n.strip().lower()
            for n in os.environ.get("IMAGE_PROVIDERS", "hf_space,fal").split(",")
            if n.strip()
        ]
        unknown = [n for n in names if n not in PROVIDER_REGISTRY]
        if unknown:
            logger.warning("Ignoring unknown image providers: %s", unknown)
        self.provider_names = [n for n in names if n in PROVIDER_REGISTRY]
        self.providers = {n: PROVIDER_REGISTRY[n]() for n in self.provider_names}
        self.offline = os.environ.get("REDESIGN_OFFLINE", "").strip().lower() in ("1", "true", "yes")

        self.cache_dir = Path(cache_dir or os.environ.get("REDESIGN_CACHE_DIR", DEFAULT_CACHE_DIR))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict:
        return {
            "providers": [
                {"name": n, "configured": _is_configured(p), "model": _model_label(p, "fast")}
                for n, p in self.providers.items()
            ],
            "offline_mode": self.offline,
            "cache_dir": str(self.cache_dir),
            "cached_results": len(list(self.cache_dir.glob("*.png"))),
        }

    def cache_key(self, image_bytes: bytes, prompt: str, quality_tier: str, seed: int) -> str:
        h = hashlib.sha256()
        h.update(hashlib.sha256(image_bytes).digest())
        h.update(prompt.encode("utf-8"))
        h.update(f"|{quality_tier}|{seed}".encode("utf-8"))
        return h.hexdigest()[:32]

    def _load_cached(self, key: str) -> Optional[GenerationResult]:
        img_path, meta_path = self.cache_dir / f"{key}.png", self.cache_dir / f"{key}.json"
        if not (img_path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        validation = ValidationReport(**meta["validation"]) if meta.get("validation") else None
        return GenerationResult(
            image=Image.open(img_path).convert("RGB"),
            provider=meta["provider"],
            model=meta["model"],
            cached=True,
            validation=validation,
        )

    def _save_cached(self, key: str, result: GenerationResult, prompt: str) -> None:
        try:
            result.image.save(self.cache_dir / f"{key}.png")
            meta = {
                "provider": result.provider,
                "model": result.model,
                "prompt": prompt,
                "validation": result.validation.dict() if result.validation else None,
            }
            (self.cache_dir / f"{key}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Could not write redesign cache entry %s: %s", key, e)

    async def generate(
        self,
        image: Image.Image,
        image_bytes: bytes,
        prompt: str,
        quality_tier: str = "fast",
        seed: int = 42,
        require_visible_change: bool = True,
    ) -> GenerationResult:
        """Run one edit, trying each configured provider in order until one succeeds.

        require_visible_change rejects outputs nearly identical to the input (a failed redesign);
        refinements disable it because a small requested tweak is a legitimate result.
        """
        key = self.cache_key(image_bytes, prompt, quality_tier, seed)
        cached = self._load_cached(key)
        if cached is not None:
            logger.info("Redesign cache hit %s (%s)", key, cached.provider)
            return cached

        first_name = self.provider_names[0] if self.provider_names else "none"
        if self.offline:
            return GenerationResult(
                None, "offline", "",
                error="Offline mode (REDESIGN_OFFLINE=1): no cached redesign exists for this image yet.",
            )

        errors: List[str] = []
        for name, provider in self.providers.items():
            if not _is_configured(provider):
                continue
            model = _model_label(provider, quality_tier)
            logger.info("Generating redesign with %s (%s)", name, model)
            if isinstance(provider, GeminiImageEditingProvider):
                out, err = await provider.edit(image, prompt=prompt, quality_tier=quality_tier)
            else:
                out, err = await provider.edit(image, prompt=prompt, quality_tier=quality_tier, seed=seed)
            if out is None:
                logger.warning("Provider %s failed: %s", name, err)
                errors.append(f"{name}: {err}")
                continue

            ok, report, normalized = validate_image_output(
                out, image.size, orig_img=image if require_visible_change else None
            )
            if not ok:
                msg = "; ".join(report.warnings) or "validation failed"
                logger.warning("Provider %s output rejected: %s", name, msg)
                errors.append(f"{name}: output rejected ({msg})")
                continue

            # Match the upload's exact size so the before/after comparison lines up.
            final = normalized.resize(image.size, Image.Resampling.LANCZOS)
            result = GenerationResult(final, name, model, validation=report)
            self._save_cached(key, result, prompt)
            return result

        if not errors:
            errors.append("no image provider is configured (check IMAGE_PROVIDERS / API keys)")
        return GenerationResult(None, first_name, "", error=" | ".join(errors))
