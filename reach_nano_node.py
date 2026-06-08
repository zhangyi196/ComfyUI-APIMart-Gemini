import concurrent.futures
import io
import json
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

try:
    import comfy.model_management as comfy_model_management
except ImportError:
    comfy_model_management = None


class ReachNanoBananaGenerationNode:
    """ComfyUI node for ReachAPI nanobanana-pro async image generation."""

    API_URL = "https://api.reachapi.ai/v1/images/create"
    QUERY_URL = "https://api.reachapi.ai/v1/tasks"
    FILE_UPLOAD_URL = "https://file.reachapi.ai/file/uploads"
    MODEL_NAME = "nanobanana-pro"
    ASPECT_RATIOS = ("1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")
    MAX_REFERENCE_IMAGE_BYTES = 10 * 1024 * 1024

    def __init__(self):
        self.poll_interval = 4
        self.max_polls = 90
        self.first_poll_delay = 4
        self._session_lock = threading.Lock()
        self._active_session: Optional[requests.Session] = None

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mode": (["text-to-image", "image-to-image"], {"default": "text-to-image"}),
                "api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True}),
                "model": ([cls.MODEL_NAME], {"default": cls.MODEL_NAME}),
                "aspect_ratio": (list(cls.ASPECT_RATIOS), {"default": "1:1"}),
                "resolution": (["1k", "2k", "4k"], {"default": "2k"}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "enable_web_search": (["false", "true"], {"default": "false"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "callback_url": ("STRING", {"multiline": False, "default": ""}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
                "image_11": ("IMAGE",),
                "image_12": ("IMAGE",),
                "image_13": ("IMAGE",),
                "image_14": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "response")
    FUNCTION = "generate"
    CATEGORY = "image/generation"

    def open_http_session(self) -> requests.Session:
        session = requests.Session()
        with self._session_lock:
            if self._active_session is not None:
                self._active_session.close()
            self._active_session = session
        return session

    def close_http_session(self) -> None:
        with self._session_lock:
            session = self._active_session
            self._active_session = None
        if session is not None:
            session.close()

    def is_comfy_interrupt(self, exc: BaseException) -> bool:
        return exc.__class__.__name__ == "InterruptProcessingException"

    def check_interrupted(self) -> None:
        if comfy_model_management is None:
            return
        try:
            comfy_model_management.throw_exception_if_processing_interrupted()
        except Exception as exc:
            if self.is_comfy_interrupt(exc):
                print("[ReachNanoBananaNode] ComfyUI interrupt detected; closing active HTTP session")
                self.close_http_session()
            raise

    def interruptible_sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while True:
            self.check_interrupted()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.2))

    def request_with_interrupt(
        self,
        session: requests.Session,
        method: str,
        url: str,
        total_timeout: float,
        **kwargs: Any,
    ) -> requests.Response:
        deadline = time.monotonic() + total_timeout
        timeout = kwargs.pop("timeout", (10, total_timeout))
        if isinstance(timeout, tuple):
            connect_timeout, read_timeout = timeout
            timeout = (connect_timeout, min(read_timeout, total_timeout))
        else:
            timeout = min(float(timeout), total_timeout)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(session.request, method, url, timeout=timeout, **kwargs)
            while True:
                try:
                    self.check_interrupted()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        future.cancel()
                        raise requests.exceptions.ReadTimeout(
                            f"Request timed out after {total_timeout:g} seconds"
                        )
                    try:
                        return future.result(timeout=min(0.2, remaining))
                    except concurrent.futures.TimeoutError:
                        continue
                except BaseException as exc:
                    if self.is_comfy_interrupt(exc):
                        self.close_http_session()
                    future.cancel()
                    raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def tensor_to_png_bytes(self, tensor: Any) -> bytes:
        if torch is not None and isinstance(tensor, torch.Tensor):
            img_array = tensor.cpu().detach().numpy()
        elif isinstance(tensor, np.ndarray):
            img_array = tensor
        else:
            try:
                img_array = np.array(tensor)
            except Exception as exc:
                raise ValueError(f"Cannot convert input image to array: {exc}") from exc

        if img_array.ndim == 4:
            img_array = img_array[0]

        if img_array.ndim != 3:
            raise ValueError(f"Unexpected image tensor shape: {img_array.shape}")

        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

        if img_array.shape[2] == 3:
            image = Image.fromarray(img_array, mode="RGB")
        elif img_array.shape[2] == 4:
            image = Image.fromarray(img_array, mode="RGBA")
        else:
            raise ValueError(f"Unsupported channel count: {img_array.shape[2]}")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def pil_to_tensor(self, image: Image.Image) -> Any:
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        if torch is not None:
            return torch.from_numpy(img_array)
        return img_array

    def collect_images(self, **kwargs: Any) -> List[Any]:
        images = []
        for index in range(1, 15):
            key = f"image_{index}"
            if key in kwargs and kwargs[key] is not None:
                images.append(kwargs[key])
        return images

    def validate_inputs(
        self,
        mode: str,
        prompt: str,
        reference_images: List[Any],
        callback_url: str,
    ) -> None:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        if mode == "image-to-image" and not reference_images:
            raise ValueError("image-to-image mode requires at least one reference image")

        if mode == "text-to-image" and reference_images:
            raise ValueError("text-to-image mode should not include reference images")

        if len(reference_images) > 14:
            raise ValueError("nanobanana-pro supports at most 14 reference images")

        if callback_url and not callback_url.startswith("https://"):
            raise ValueError("callback_url only supports https URLs")

        if len(callback_url) > 2048:
            raise ValueError("callback_url cannot exceed 2048 characters")

    def upload_image(
        self,
        session: requests.Session,
        image_tensor: Any,
        api_key: str,
        index: int,
    ) -> str:
        image_bytes = self.tensor_to_png_bytes(image_tensor)
        if len(image_bytes) > self.MAX_REFERENCE_IMAGE_BYTES:
            raise ValueError("Each nanobanana-pro reference image must be no larger than 10MB")

        headers = {"Authorization": f"Bearer {api_key}"}
        files = {
            "file": (f"reach_nanobanana_{index}.png", image_bytes, "image/png"),
        }

        print(f"[ReachNanoBananaNode] Uploading reference image {index}: {self.FILE_UPLOAD_URL}")
        response = self.request_with_interrupt(
            session,
            "POST",
            self.FILE_UPLOAD_URL,
            total_timeout=60,
            headers=headers,
            files=files,
        )
        try:
            response.raise_for_status()
            response_data = response.json()
            print(f"[ReachNanoBananaNode] Upload response {index}: {json.dumps(response_data, ensure_ascii=False)}")

            if response_data.get("code") != 200:
                raise ValueError(f"Reference image upload failed: {response_data}")

            data = response_data.get("data", {})
            image_url = data.get("url")
            if not image_url:
                raise ValueError(f"Upload response missing data.url: {response_data}")
            if not image_url.startswith("https://"):
                raise ValueError(f"Uploaded image URL is not https: {image_url}")
            return image_url
        finally:
            response.close()

    def upload_reference_images(
        self,
        session: requests.Session,
        reference_images: List[Any],
        api_key: str,
    ) -> List[str]:
        image_urls = []
        for index, image_tensor in enumerate(reference_images, start=1):
            image_urls.append(self.upload_image(session, image_tensor, api_key, index))
        return image_urls

    def build_input_payload(
        self,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
        output_format: str,
        enable_web_search: str,
        seed: int,
        image_urls: List[str],
    ) -> Dict[str, Any]:
        if isinstance(enable_web_search, bool):
            web_search_enabled = enable_web_search
        else:
            web_search_enabled = str(enable_web_search).lower() == "true"

        input_payload: Dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "output_format": output_format,
            "enable_web_search": web_search_enabled,
        }

        if seed > 0:
            input_payload["seed"] = seed

        if image_urls:
            input_payload["image_urls"] = image_urls

        return input_payload

    def extract_task_id(self, response_data: Dict[str, Any]) -> str:
        if response_data.get("code") != 200:
            raise ValueError(f"Create task failed: {response_data}")

        task_id = response_data.get("task_id")
        if not task_id:
            raise ValueError(f"Create task response missing task_id: {response_data}")
        return task_id

    def extract_image_url(self, response_data: Dict[str, Any]) -> str:
        status = response_data.get("status")
        if status != "success":
            raise ValueError(f"Task is not successful, current status: {status}")

        data = response_data.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError(f"Task succeeded but returned no image data: {response_data}")

        first_image = data[0]
        if not isinstance(first_image, dict):
            raise ValueError(f"Task image data has unexpected format: {first_image}")

        image_url = first_image.get("url")
        if not image_url:
            raise ValueError(f"Cannot find result image URL: {first_image}")
        return image_url

    def poll_task_status(
        self,
        session: requests.Session,
        task_id: str,
        api_key: str,
    ) -> Tuple[str, Dict[str, Any]]:
        print(f"[ReachNanoBananaNode] Polling task: {task_id}")
        headers = {"Authorization": f"Bearer {api_key}"}

        self.interruptible_sleep(self.first_poll_delay)

        for poll_count in range(1, self.max_polls + 1):
            try:
                response = self.request_with_interrupt(
                    session,
                    "GET",
                    f"{self.QUERY_URL}/{task_id}",
                    total_timeout=15,
                    headers=headers,
                )
                response.raise_for_status()
                try:
                    response_data = response.json()
                    print(
                        f"[ReachNanoBananaNode] Poll {poll_count}/{self.max_polls}: "
                        f"{json.dumps(response_data, ensure_ascii=False)}"
                    )

                    status = response_data.get("status")
                    if status == "success":
                        return self.extract_image_url(response_data), response_data

                    if status == "failed":
                        raise ValueError(f"Task failed: {response_data.get('msg') or response_data}")

                    if status in {"queued", "generating"}:
                        self.interruptible_sleep(self.poll_interval)
                        continue

                    if response_data.get("code") != 200:
                        raise ValueError(f"Task query failed: {response_data}")

                    print(f"[ReachNanoBananaNode] Unknown task status {status}; polling again")
                    self.interruptible_sleep(self.poll_interval)
                finally:
                    response.close()
            except requests.exceptions.RequestException as exc:
                print(f"[ReachNanoBananaNode] Task query request failed: {exc}")
                if poll_count == self.max_polls:
                    raise RuntimeError(f"Task query failed after retries: {exc}") from exc
                self.interruptible_sleep(self.poll_interval)

        raise TimeoutError(
            f"Task did not complete within {self.first_poll_delay + self.max_polls * self.poll_interval} seconds"
        )

    def download_image(self, session: requests.Session, image_url: str) -> Image.Image:
        print(f"[ReachNanoBananaNode] Downloading result image: {image_url}")
        response = self.request_with_interrupt(session, "GET", image_url, total_timeout=30)
        try:
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as exc:
            raise RuntimeError(f"Failed to download result image: {exc}") from exc
        finally:
            response.close()

    def generate(
        self,
        mode: str,
        api_key: str,
        prompt: str,
        model: str,
        aspect_ratio: str,
        resolution: str,
        output_format: str,
        enable_web_search: str,
        seed: int,
        **kwargs: Any,
    ) -> Tuple[Any, str, str]:
        session = self.open_http_session()
        try:
            self.check_interrupted()
            print(f"[ReachNanoBananaNode] Starting generation, mode: {mode}")

            reference_images = self.collect_images(**kwargs)
            callback_url = kwargs.get("callback_url", "").strip()

            self.validate_inputs(
                mode=mode,
                prompt=prompt,
                reference_images=reference_images,
                callback_url=callback_url,
            )

            uploaded_image_urls = (
                self.upload_reference_images(session, reference_images, api_key)
                if reference_images
                else []
            )
            input_payload = self.build_input_payload(
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                output_format=output_format,
                enable_web_search=enable_web_search,
                seed=seed,
                image_urls=uploaded_image_urls,
            )

            payload: Dict[str, Any] = {
                "model": model,
                "input": input_payload,
            }
            if callback_url:
                payload["callback_url"] = callback_url

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            print(f"[ReachNanoBananaNode] Submitting task: {self.API_URL}")
            response = self.request_with_interrupt(
                session,
                "POST",
                self.API_URL,
                total_timeout=60,
                json=payload,
                headers=headers,
            )
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                print(f"[ReachNanoBananaNode] Submit failed response: {response.text[:2000]}")
                raise RuntimeError(f"HTTP {response.status_code} {response.text[:2000]}") from exc

            try:
                submit_response = response.json()
                print(f"[ReachNanoBananaNode] Submit response: {json.dumps(submit_response, ensure_ascii=False)}")

                task_id = self.extract_task_id(submit_response)
                image_url, query_response = self.poll_task_status(session, task_id, api_key)
                result_image = self.download_image(session, image_url)
            finally:
                response.close()

            result_tensor = self.pil_to_tensor(result_image)
            response_text = json.dumps(
                {
                    "submit_url": self.API_URL,
                    "query_url": f"{self.QUERY_URL}/{task_id}",
                    "upload_url": self.FILE_UPLOAD_URL,
                    "request_seed": seed,
                    "uploaded_image_urls": uploaded_image_urls,
                    "submit_response": submit_response,
                    "query_response": query_response,
                },
                ensure_ascii=False,
                indent=2,
            )

            print("[ReachNanoBananaNode] Generation completed")
            return result_tensor, image_url, response_text
        except Exception as exc:
            if self.is_comfy_interrupt(exc):
                raise
            print(f"[ReachNanoBananaNode] Execution failed: {exc}")
            raise Exception(f"ReachNanoBananaNode execution failed: {exc}") from exc
        finally:
            self.close_http_session()


NODE_CLASS_MAPPINGS = {
    "ReachNanoBananaGenerationNode": ReachNanoBananaGenerationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReachNanoBananaGenerationNode": "reach nanobanana",
}
