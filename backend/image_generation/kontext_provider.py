"""FLUX.1 Kontext instruction-based image editing providers.

Kontext takes the original street photo plus a written design brief and returns the
same street redesigned, preserving buildings, camera angle and perspective. Two backends:

- HuggingFaceSpaceKontextProvider: the official black-forest-labs FLUX.1-Kontext-Dev Space
  (free ZeroGPU quota; an optional HF_TOKEN raises the quota). Plain HTTP against the
  Gradio REST API, so no extra Python dependency is needed.
- FalKontextProvider: fal.ai hosted Kontext (pay per image, needs FAL_KEY).
"""

import asyncio
import base64
import io
import json
import os
import urllib.error
import urllib.request
import uuid
from typing import Optional, Tuple

from PIL import Image

from .base import ImageEditingProvider

# Kontext works at ~1 megapixel; larger uploads only cost transfer time.
MAX_INPUT_DIM = 1024


def _prepare_input(image: Image.Image) -> Image.Image:
    work = image.convert("RGB")
    w, h = work.size
    if max(w, h) > MAX_INPUT_DIM:
        scale = MAX_INPUT_DIM / float(max(w, h))
        work = work.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    return work


def _to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _http_json(url: str, data: Optional[bytes], headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_bytes(url: str, headers: dict, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_error_message(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode("utf-8", errors="replace")[:200]
    except Exception:
        return str(e)


class HuggingFaceSpaceKontextProvider(ImageEditingProvider):
    """Calls the FLUX.1-Kontext-Dev Gradio Space on Hugging Face ZeroGPU."""

    name = "hf_space"

    def __init__(self):
        space = os.environ.get("KONTEXT_SPACE", "black-forest-labs/FLUX.1-Kontext-Dev").strip()
        self.space = space
        self.base_url = "https://" + space.lower().replace("/", "-").replace(".", "-") + ".hf.space/gradio_api"
        self.hf_token = os.environ.get("HF_TOKEN", "").strip()
        self.timeout = int(os.environ.get("KONTEXT_SPACE_TIMEOUT", "240"))
        self._fn_index: Optional[int] = None

    @property
    def model_name(self) -> str:
        return f"{self.space} (Hugging Face Space)"

    def is_configured(self) -> bool:
        return True

    def _headers(self, extra: Optional[dict] = None) -> dict:
        headers = {"User-Agent": "resilicity-backend"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        if extra:
            headers.update(extra)
        return headers

    def _upload(self, png: bytes) -> str:
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="files"; filename="street.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + png + f"\r\n--{boundary}--\r\n".encode("utf-8")
        paths = _http_json(
            f"{self.base_url}/upload",
            body,
            self._headers({"Content-Type": f"multipart/form-data; boundary={boundary}"}),
            timeout=60,
        )
        return paths[0]

    def _infer_fn_index(self) -> int:
        if self._fn_index is None:
            config = _http_json(f"{self.base_url[: -len('/gradio_api')]}/config", None, self._headers(), timeout=30)
            self._fn_index = next(
                d["id"] for d in config["dependencies"] if d.get("api_name") == "infer"
            )
        return self._fn_index

    def _run(self, png: bytes, prompt: str, seed: int, steps: int, guidance: float) -> bytes:
        # Uses the Gradio queue API (what gradio_client uses) rather than /call, because only
        # the queue stream carries readable error messages such as "ZeroGPU quota exceeded".
        server_path = self._upload(png)
        session = uuid.uuid4().hex
        payload = {
            "data": [
                {"path": server_path, "meta": {"_type": "gradio.FileData"}},
                prompt,
                seed,
                False,  # randomize_seed: keep results reproducible for caching
                guidance,
                steps,
            ],
            "fn_index": self._infer_fn_index(),
            "session_hash": session,
            "event_data": None,
            "trigger_id": None,
        }
        _http_json(
            f"{self.base_url}/queue/join",
            json.dumps(payload).encode("utf-8"),
            self._headers({"Content-Type": "application/json"}),
            timeout=60,
        )
        stream = _http_bytes(
            f"{self.base_url}/queue/data?session_hash={session}", self._headers(), timeout=self.timeout
        ).decode("utf-8", errors="replace")

        for line in stream.splitlines():
            if not line.startswith("data:"):
                continue
            msg = json.loads(line[len("data:"):])
            if msg.get("msg") != "process_completed":
                continue
            output = msg.get("output") or {}
            if not msg.get("success"):
                raise RuntimeError(output.get("error") or msg.get("title") or "Kontext Space job failed")
            result = output["data"][0]
            file_url = result.get("url") or f"{self.base_url}/file={result['path']}"
            return _http_bytes(file_url, self._headers(), timeout=60)

        raise RuntimeError("Kontext Space stream ended without a result")

    async def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        model_name: Optional[str] = None,
        quality_tier: str = "fast",
        mask_image: Optional[Image.Image] = None,
        seed: int = 42,
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        steps = 28 if quality_tier == "final" else 20
        png = _to_png_bytes(_prepare_input(image))
        try:
            loop = asyncio.get_running_loop()
            out = await loop.run_in_executor(None, self._run, png, prompt, seed, steps, 2.5)
            return Image.open(io.BytesIO(out)).convert("RGB"), None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                return None, "Hugging Face Space GPU quota reached. Add HF_TOKEN or try again later."
            return None, f"Hugging Face Space error ({e.code}): {_http_error_message(e)}"
        except Exception as e:
            msg = str(e)
            if "quota" in msg.lower() or "runs limit" in msg.lower():
                hint = "" if self.hf_token else " Set HF_TOKEN in backend/.env for more free quota."
                return None, f"Hugging Face Space GPU quota reached.{hint}"
            return None, f"Hugging Face Space request failed: {msg[:200]}"


class FalKontextProvider(ImageEditingProvider):
    """Calls fal.ai hosted FLUX.1 Kontext (dev for previews, pro for final renders)."""

    name = "fal"

    def __init__(self):
        self.api_key = os.environ.get("FAL_KEY", "").strip()
        self.fast_model = os.environ.get("FAL_KONTEXT_MODEL", "fal-ai/flux-kontext/dev").strip()
        self.final_model = os.environ.get("FAL_KONTEXT_FINAL_MODEL", "fal-ai/flux-pro/kontext").strip()

    @property
    def model_name(self) -> str:
        return self.fast_model

    def model_for_tier(self, quality_tier: str) -> str:
        return self.final_model if quality_tier == "final" else self.fast_model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _run(self, model: str, png: bytes, prompt: str, seed: int) -> bytes:
        data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        payload = {
            "prompt": prompt,
            "image_url": data_uri,
            "seed": seed,
            "num_images": 1,
            "output_format": "png",
            "sync_mode": True,
        }
        headers = {"Authorization": f"Key {self.api_key}", "Content-Type": "application/json"}
        resp = _http_json(f"https://fal.run/{model}", json.dumps(payload).encode("utf-8"), headers, timeout=180)
        url = resp["images"][0]["url"]
        if url.startswith("data:"):
            return base64.b64decode(url.split(",", 1)[1])
        return _http_bytes(url, {}, timeout=60)

    async def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        model_name: Optional[str] = None,
        quality_tier: str = "fast",
        mask_image: Optional[Image.Image] = None,
        seed: int = 42,
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        if not self.api_key:
            return None, "fal.ai is not configured (FAL_KEY missing)."
        model = model_name or self.model_for_tier(quality_tier)
        png = _to_png_bytes(_prepare_input(image))
        try:
            loop = asyncio.get_running_loop()
            out = await loop.run_in_executor(None, self._run, model, png, prompt, seed)
            return Image.open(io.BytesIO(out)).convert("RGB"), None
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return None, "fal.ai authentication failed. Verify FAL_KEY."
            if e.code in (402, 429):
                return None, "fal.ai credits or rate limit exhausted."
            return None, f"fal.ai error ({e.code}): {_http_error_message(e)}"
        except Exception as e:
            return None, f"fal.ai request failed: {str(e)[:200]}"
