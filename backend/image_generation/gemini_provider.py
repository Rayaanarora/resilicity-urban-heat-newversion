"""Gemini multimodal image editing provider implementation."""

import asyncio
import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import Optional, Tuple
from PIL import Image

from .base import ImageEditingProvider
from .validation import validate_image_output


class GeminiImageEditingProvider(ImageEditingProvider):
    """Integrates Google Gemini 3.1 Flash Image and Gemini 3 Pro Image editing APIs."""

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.flash_model = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image").strip()
        self.pro_model = os.environ.get("GEMINI_FINAL_IMAGE_MODEL", "gemini-3-pro-image").strip()

        # Enforce prohibition of deprecated/unwanted models
        if "gemini-2.5-flash-image" in (self.flash_model, self.pro_model):
            raise ValueError("gemini-2.5-flash-image is prohibited by system policy. Use gemini-3.1-flash-image or gemini-3-pro-image.")

    def get_model_for_tier(self, quality_tier: str = "fast") -> str:
        """Select model based on user requested tier (Fast / Final)."""
        if quality_tier == "final":
            return self.pro_model
        return self.flash_model

    async def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        model_name: Optional[str] = None,
        quality_tier: str = "fast",
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        """Edit photograph via Gemini multimodal image generation API."""
        if not self.api_key:
            return None, "Gemini API key is not configured in backend environment (GEMINI_API_KEY)."

        selected_model = model_name or self.get_model_for_tier(quality_tier)

        # Disallow prohibited model
        if "gemini-2.5-flash-image" in selected_model:
            selected_model = self.flash_model

        # Prepare JPEG base64 payload
        # Resize temporarily if input image is massive (>2048px) to optimize network transfer
        max_dim = 1536 if quality_tier == "final" else 1024
        w, h = image.size
        orig_size = (w, h)
        work_img = image.copy()
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            work_img = work_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        work_img.convert("RGB").save(buf, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": img_b64,
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
            }
        }

        # Attempt call with up to 2 retries for transient issues
        max_attempts = 2
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                loop = asyncio.get_running_loop()
                data_bytes = json.dumps(payload).encode("utf-8")

                def _make_call():
                    req = urllib.request.Request(
                        endpoint_url,
                        data=data_bytes,
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=90) as resp:
                        return json.loads(resp.read().decode("utf-8"))

                resp_json = await loop.run_in_executor(None, _make_call)

                # Extract generated image bytes
                candidates = resp_json.get("candidates", [])
                if not candidates:
                    return None, "Gemini returned no candidates in response."

                parts = candidates[0].get("content", {}).get("parts", [])
                image_bytes = None
                for p in parts:
                    inline = p.get("inlineData")
                    if inline and inline.get("data"):
                        image_bytes = base64.b64decode(inline["data"])
                        break

                if not image_bytes:
                    # Check if text was returned instead
                    text_parts = [p.get("text", "") for p in parts if p.get("text")]
                    explanation = " ".join(text_parts).strip()
                    return None, f"Gemini returned text instead of image: {explanation[:200]}"

                gen_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Validate and normalize aspect ratio
                is_valid, report, normalized = validate_image_output(gen_image, orig_size)
                if is_valid:
                    return normalized, None
                else:
                    last_error = f"Validation failed: {', '.join(report.warnings)}"

            except urllib.error.HTTPError as e:
                try:
                    err_body = json.loads(e.read().decode("utf-8"))
                    msg = err_body.get("error", {}).get("message", str(e))
                except Exception:
                    msg = str(e)

                # Format user-friendly error without leaking internal URLs or keys
                if e.code == 429:
                    last_error = "Gemini API quota exceeded for image generation. Please check project quota or billing."
                elif e.code == 404:
                    last_error = f"Model '{selected_model}' not found or does not support image generation."
                elif e.code == 401 or e.code == 403:
                    last_error = "Gemini API authentication failed. Verify GEMINI_API_KEY."
                else:
                    last_error = f"Gemini API error ({e.code}): {msg[:160]}"

                # Non-retriable auth or quota errors
                if e.code in (401, 403, 429):
                    break

            except Exception as e:
                last_error = f"Image generation connection error: {str(e)[:150]}"

            if attempt < max_attempts:
                await asyncio.sleep(1.5)

        return None, last_error or "Image generation failed."
