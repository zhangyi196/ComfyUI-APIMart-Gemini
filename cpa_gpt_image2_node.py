import base64
import concurrent.futures
import io
import json
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

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


def _find_first_url(data: Any) -> str:
    """递归查找 JSON 中第一个 https URL（兜底策略）。"""
    if isinstance(data, str) and data.startswith("https://"):
        return data
    if isinstance(data, dict):
        for v in data.values():
            result = _find_first_url(v)
            if result:
                return result
    if isinstance(data, list):
        for item in data:
            result = _find_first_url(item)
            if result:
                return result
    return ""


def _extract_upload_url(response_data: Dict[str, Any]) -> str:
    """自适应上传响应解析 — 支持 Reach、APIMart 及未知格式。

    优先级：
    1. 有 code 字段 → Reach 格式（校验 code=200，提取 data.url）
    2. 顶层有 url   → APIMart 格式（直接返回）
    3. 递归查找第一个 https:// 开头的值（兜底）
    """
    if "code" in response_data:
        if response_data.get("code") != 200:
            raise ValueError(f"参考图上传失败: {response_data}")
        data = response_data.get("data", {})
        url = data.get("url")
        if url:
            return url

    url = response_data.get("url")
    if isinstance(url, str) and url:
        return url

    return _find_first_url(response_data)


class CPAGPTImage2GenerationNode:
    """ComfyUI node for CPA GPT Image 2 async generation."""

    DEFAULT_BASE_URL = "http://localhost:8317/"
    FILE_UPLOAD_URL = "https://file.reachapi.ai/file/uploads"
    MODEL_NAME = "gpt-image-2"
    SIZE_PATTERN = re.compile(r"^(\d+)x(\d+)$")
    RETRYABLE_SUBMIT_STATUS_CODES = {502, 503, 504}
    SUBMIT_RETRY_DELAYS = (2, 4)
    STANDARD_SIZE_MAP = {
        ("1k", "1:1"): "1024x1024",
        ("1k", "3:2"): "1536x1024",
        ("1k", "2:3"): "1024x1536",
        ("1k", "4:3"): "1360x1024",
        ("1k", "3:4"): "1024x1360",
        ("1k", "16:9"): "1824x1024",
        ("1k", "9:16"): "1024x1824",
        ("2k", "1:1"): "2048x2048",
        ("2k", "3:2"): "2048x1152",
        ("2k", "4:3"): "2048x1536",
        ("2k", "3:4"): "1536x2048",
        ("4k", "3:2"): "3840x2160",
        ("4k", "2:3"): "2160x3840",
        ("4k", "4:3"): "3840x2880",
        ("4k", "3:4"): "2880x3840",
    }

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
                "base_url": ("STRING", {"multiline": False, "default": cls.DEFAULT_BASE_URL}),
                "prompt": ("STRING", {"multiline": True}),
                "model": ([cls.MODEL_NAME], {"default": cls.MODEL_NAME}),
                "resolution": (["1k", "2k", "4k"], {"default": "1k"}),
                "aspect_ratio": (["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16"], {"default": "1:1"}),
                "quality": (["low", "medium", "high", "auto"], {"default": "medium"}),
                "background": (["auto", "opaque", "transparent"], {"default": "auto"}),
                "moderation": (["auto", "low"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "custom_size": ("STRING", {"multiline": False, "default": ""}),
                "callback_url": ("STRING", {"multiline": False, "default": ""}),
                "upload_url": ("STRING", {"multiline": False, "default": cls.FILE_UPLOAD_URL}),
                "upload_api_key": ("STRING", {"multiline": False, "default": ""}),
                "submit_timeout": ("INT", {"default": 300, "min": 30, "max": 1800}),
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
                print("[CPAGPTImage2Node] 检测到 ComfyUI 关闭任务，正在断开当前 HTTP 连接")
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
                        raise requests.exceptions.ReadTimeout(f"请求超时：{total_timeout:g} 秒内未收到响应")
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

    def normalize_base_url(self, base_url: str) -> str:
        normalized = base_url.strip()
        if not normalized:
            raise ValueError("base_url 不能为空")
        return normalized.rstrip("/")

    def build_submit_url(self, base_url: str, mode: str) -> str:
        normalized = self.normalize_base_url(base_url)
        if normalized.endswith("/v1/images/generations") or normalized.endswith("/v1/images/edits"):
            return normalized
        endpoint = "/v1/images/edits" if mode == "image-to-image" else "/v1/images/generations"
        return f"{normalized}{endpoint}"

    def build_query_url(self, base_url: str) -> str:
        normalized = self.normalize_base_url(base_url)
        if normalized.endswith("/v1/images/generations") or normalized.endswith("/v1/images/edits"):
            parts = urlsplit(normalized)
            path = parts.path.rstrip("/")
            path_parts = path.split("/")
            query_path = "/".join(path_parts[:-2] + ["tasks"])
            return urlunsplit((parts.scheme, parts.netloc, query_path, parts.query, parts.fragment))
        return f"{normalized}/v1/tasks"

    def resolve_upload_url(self, upload_url: str) -> str:
        normalized = upload_url.strip()
        if not normalized:
            return self.FILE_UPLOAD_URL

        if "://" not in normalized:
            normalized = f"https://{normalized.lstrip('/')}"

        return normalized

    def tensor_to_png_bytes(self, tensor: Any) -> bytes:
        if torch is not None and isinstance(tensor, torch.Tensor):
            img_array = tensor.cpu().detach().numpy()
        elif isinstance(tensor, np.ndarray):
            img_array = tensor
        else:
            try:
                img_array = np.array(tensor)
            except Exception as exc:
                raise ValueError(f"无法将输入转换为图像数组: {exc}") from exc

        if img_array.ndim == 4:
            img_array = img_array[0]

        if img_array.ndim != 3:
            raise ValueError(f"图像张量维度不正确: {img_array.shape}")

        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

        if img_array.shape[2] == 3:
            image = Image.fromarray(img_array, mode="RGB")
        elif img_array.shape[2] == 4:
            image = Image.fromarray(img_array, mode="RGBA")
        else:
            raise ValueError(f"不支持的通道数: {img_array.shape[2]}")

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
        for i in range(1, 15):
            key = f"image_{i}"
            if key in kwargs and kwargs[key] is not None:
                images.append(kwargs[key])
        return images

    def validate_size(self, custom_size: str) -> None:
        match = self.SIZE_PATTERN.fullmatch(custom_size)
        if not match:
            raise ValueError("custom_size 必须是类似 1024x1536 的像素尺寸")

        width = int(match.group(1))
        height = int(match.group(2))
        longest_edge = max(width, height)
        shortest_edge = min(width, height)
        total_pixels = width * height

        if longest_edge > 3840:
            raise ValueError("custom_size 的最长边不能超过 3840")
        if width % 16 != 0 or height % 16 != 0:
            raise ValueError("custom_size 的宽高都必须是 16 的倍数")
        if longest_edge / shortest_edge > 3:
            raise ValueError("custom_size 的长宽比不能超过 3:1")
        if total_pixels < 655360 or total_pixels > 8294400:
            raise ValueError("custom_size 的总像素数必须介于 655360 和 8294400 之间")

    def validate_inputs(
        self,
        mode: str,
        resolution: str,
        aspect_ratio: str,
        custom_size: str,
        reference_images: List[Any],
        background: str,
        output_format: str,
        callback_url: str,
    ) -> None:
        if mode == "image-to-image" and not reference_images:
            raise ValueError("图像编辑模式至少需要 1 张参考图")

        if mode == "text-to-image" and reference_images:
            raise ValueError("文生图模式下请不要传参考图")

        if len(reference_images) > 14:
            raise ValueError("参考图最多支持 14 张")

        if background == "transparent" and output_format == "jpeg":
            raise ValueError("background=transparent 不支持 output_format=jpeg")

        if callback_url and not callback_url.startswith("https://"):
            raise ValueError("callback_url 仅支持 https 地址")

        normalized_size = custom_size.strip()
        if normalized_size:
            self.validate_size(normalized_size)
            return

        if (resolution, aspect_ratio) not in self.STANDARD_SIZE_MAP:
            raise ValueError("当前 resolution 与 aspect_ratio 组合不在标准映射表中，请改用 custom_size")

    def upload_image(
        self,
        session: requests.Session,
        image_tensor: Any,
        api_key: str,
        upload_url: str,
        index: int,
    ) -> str:
        image_bytes = self.tensor_to_png_bytes(image_tensor)
        headers = {"Authorization": f"Bearer {api_key}"}
        files = {
            "file": (f"cpa_gpt_image2_{index}.png", image_bytes, "image/png"),
        }

        print(f"[CPAGPTImage2Node] 上传参考图 {index}: {upload_url}")
        self.check_interrupted()
        response = self.request_with_interrupt(
            session,
            "POST",
            upload_url,
            total_timeout=60,
            headers=headers,
            files=files,
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            if response.status_code == 401:
                raise RuntimeError(
                    "参考图上传鉴权失败：当前上传接口需要它自己的 Bearer API key。"
                    "如果任务提交走的是本地网关，请为 upload_url / upload_api_key 单独配置可用的上传凭证。"
                ) from exc
            raise

        try:
            response_data = response.json()
            print(f"[CPAGPTImage2Node] 上传响应 {index}: {json.dumps(response_data, ensure_ascii=False)}")

            image_url = _extract_upload_url(response_data)
            if not image_url:
                raise ValueError(f"上传响应无法解析 URL: {response_data}")
            if not image_url.startswith("https://"):
                raise ValueError(f"上传后返回的 url 不是 https 地址: {image_url}")
            return image_url
        finally:
            response.close()

    def upload_reference_images(
        self,
        session: requests.Session,
        reference_images: List[Any],
        api_key: str,
        upload_url: str,
    ) -> List[str]:
        image_urls = []
        for index, image_tensor in enumerate(reference_images, start=1):
            image_urls.append(self.upload_image(session, image_tensor, api_key, upload_url, index))
        return image_urls

    def build_input_payload(
        self,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        quality: str,
        background: str,
        moderation: str,
        output_format: str,
        custom_size: str,
        image_urls: List[str],
    ) -> Dict[str, Any]:
        input_payload: Dict[str, Any] = {
            "prompt": prompt,
            "quality": quality,
            "background": background,
            "moderation": moderation,
            "output_format": output_format,
        }

        normalized_size = custom_size.strip()
        if normalized_size:
            input_payload["size"] = normalized_size
        else:
            input_payload["size"] = self.STANDARD_SIZE_MAP[(resolution, aspect_ratio)]

        if image_urls:
            input_payload["images"] = [{"image_url": url} for url in image_urls]

        return input_payload

    def submit_with_retry(
        self,
        session: requests.Session,
        submit_url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        submit_timeout: int,
    ) -> requests.Response:
        max_attempts = len(self.SUBMIT_RETRY_DELAYS) + 1
        last_response: Optional[requests.Response] = None

        for attempt in range(1, max_attempts + 1):
            deadline = time.monotonic() + submit_timeout
            try:
                while True:
                    self.check_interrupted()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError(f"提交请求超时：{submit_timeout} 秒内未收到网关响应")

                    response = self.request_with_interrupt(
                        session,
                        "POST",
                        submit_url,
                        total_timeout=remaining,
                        json=payload,
                        headers=headers,
                    )
                    break
            except requests.exceptions.ReadTimeout as exc:
                raise RuntimeError(f"提交请求超时：{submit_timeout} 秒内未收到网关响应") from exc

            last_response = response
            if response.status_code not in self.RETRYABLE_SUBMIT_STATUS_CODES:
                return response

            print(
                f"[CPAGPTImage2Node] 提交返回 HTTP {response.status_code}，"
                f"第 {attempt}/{max_attempts} 次尝试"
            )
            print(f"[CPAGPTImage2Node] 提交失败响应: {response.text[:2000]}")

            if attempt >= max_attempts:
                return response

            wait_seconds = self.SUBMIT_RETRY_DELAYS[attempt - 1]
            response.close()
            print(f"[CPAGPTImage2Node] {wait_seconds} 秒后重试提交")
            self.interruptible_sleep(wait_seconds)

        if last_response is None:
            raise RuntimeError("提交请求失败：未收到任何响应")
        return last_response

    def extract_task_id(self, response_data: Dict[str, Any]) -> Optional[str]:
        task_id = response_data.get("task_id")
        if isinstance(task_id, str) and task_id:
            return task_id
        return None

    def extract_sync_result(
        self,
        session: requests.Session,
        response_data: Dict[str, Any],
    ) -> Tuple[Image.Image, str]:
        data = response_data.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError(f"接口返回异常: {response_data}")

        first_image = data[0]
        if not isinstance(first_image, dict):
            raise ValueError(f"接口返回异常: {response_data}")

        image_url = first_image.get("url", "")
        b64_json = first_image.get("b64_json")
        if isinstance(b64_json, str) and b64_json:
            try:
                image_bytes = base64.b64decode(b64_json)
                return Image.open(io.BytesIO(image_bytes)), image_url
            except Exception as exc:
                raise RuntimeError(f"解析 b64_json 失败: {exc}") from exc

        if image_url:
            return self.download_image(session, image_url), image_url

        raise ValueError(f"接口未返回可用图片数据: {first_image}")

    def sanitize_response_data(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        def sanitize_value(value: Any) -> Any:
            if isinstance(value, dict):
                sanitized: Dict[str, Any] = {}
                for key, item in value.items():
                    if key == "b64_json" and isinstance(item, str):
                        sanitized[key] = f"<omitted {len(item)} chars>"
                    else:
                        sanitized[key] = sanitize_value(item)
                return sanitized
            if isinstance(value, list):
                return [sanitize_value(item) for item in value]
            return value

        return sanitize_value(response_data)

    def extract_image_url(self, response_data: Dict[str, Any]) -> str:
        status = response_data.get("status")
        if status != "success":
            raise ValueError(f"任务未成功，当前状态: {status}")

        data = response_data.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError(f"任务已成功，但未返回图片数据: {response_data}")

        first_image = data[0]
        image_url = first_image.get("url")
        if not image_url:
            raise ValueError(f"无法解析返回图片地址: {first_image}")
        return image_url

    def poll_task_status(
        self,
        session: requests.Session,
        task_id: str,
        api_key: str,
        query_url: str,
    ) -> Tuple[str, Dict[str, Any]]:
        print(f"[CPAGPTImage2Node] 开始轮询任务状态: {task_id}")
        headers = {"Authorization": f"Bearer {api_key}"}

        self.interruptible_sleep(self.first_poll_delay)

        for poll_count in range(1, self.max_polls + 1):
            try:
                self.check_interrupted()
                response = self.request_with_interrupt(
                    session,
                    "GET",
                    f"{query_url}/{task_id}",
                    total_timeout=15,
                    headers=headers,
                )
                response.raise_for_status()
                try:
                    response_data = response.json()
                    print(
                        f"[CPAGPTImage2Node] 轮询 {poll_count}/{self.max_polls}: "
                        f"{json.dumps(self.sanitize_response_data(response_data), ensure_ascii=False)}"
                    )

                    status = response_data.get("status")
                    if status == "success":
                        return self.extract_image_url(response_data), response_data

                    if status == "failed":
                        raise ValueError(f"任务失败: {response_data.get('msg') or response_data}")

                    if status in {"queued", "generating"}:
                        self.interruptible_sleep(self.poll_interval)
                        continue

                    if response_data.get("code") != 200:
                        raise ValueError(f"任务查询失败: {response_data}")

                    print(f"[CPAGPTImage2Node] 遇到未识别状态 {status}，继续轮询")
                    self.interruptible_sleep(self.poll_interval)
                finally:
                    response.close()
            except requests.exceptions.RequestException as exc:
                print(f"[CPAGPTImage2Node] 查询任务失败: {exc}")
                if poll_count == self.max_polls:
                    raise RuntimeError(f"轮询任务状态失败: {exc}") from exc
                self.interruptible_sleep(self.poll_interval)

        raise TimeoutError(f"任务在 {self.first_poll_delay + self.max_polls * self.poll_interval} 秒内未完成")

    def download_image(self, session: requests.Session, image_url: str) -> Image.Image:
        print(f"[CPAGPTImage2Node] 下载结果图片: {image_url}")
        try:
            response = self.request_with_interrupt(session, "GET", image_url, total_timeout=30)
            response.raise_for_status()
            try:
                return Image.open(io.BytesIO(response.content))
            finally:
                response.close()
        except Exception as exc:
            raise RuntimeError(f"下载结果图片失败: {exc}") from exc

    def generate(
        self,
        mode: str,
        api_key: str,
        base_url: str,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        quality: str,
        background: str,
        moderation: str,
        output_format: str,
        seed: int,
        **kwargs: Any,
    ) -> Tuple[Any, str, str]:
        session = self.open_http_session()
        try:
            self.check_interrupted()
            print(f"[CPAGPTImage2Node] 开始生成，模式: {mode}")
            submit_url = self.build_submit_url(base_url, mode)
            query_url = self.build_query_url(submit_url)
            print(f"[CPAGPTImage2Node] 提交地址: {submit_url}")
            print(f"[CPAGPTImage2Node] 查询地址: {query_url}")

            reference_images = self.collect_images(**kwargs)
            custom_size = kwargs.get("custom_size", "")
            callback_url = kwargs.get("callback_url", "").strip()
            upload_url = self.resolve_upload_url(kwargs.get("upload_url", ""))
            upload_api_key = kwargs.get("upload_api_key", "").strip() or api_key
            submit_timeout = int(kwargs.get("submit_timeout", 300))

            self.validate_inputs(
                mode=mode,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                custom_size=custom_size,
                reference_images=reference_images,
                background=background,
                output_format=output_format,
                callback_url=callback_url,
            )

            image_urls = (
                self.upload_reference_images(session, reference_images, upload_api_key, upload_url)
                if reference_images
                else []
            )
            input_payload = self.build_input_payload(
                prompt=prompt,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                quality=quality,
                background=background,
                moderation=moderation,
                output_format=output_format,
                custom_size=custom_size,
                image_urls=image_urls,
            )

            payload: Dict[str, Any] = {
                "model": model,
                **input_payload,
            }
            if seed > 0:
                payload["seed"] = seed
            if callback_url:
                payload["callback_url"] = callback_url

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            response = self.submit_with_retry(
                session=session,
                submit_url=submit_url,
                payload=payload,
                headers=headers,
                submit_timeout=submit_timeout,
            )

            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                print(f"[CPAGPTImage2Node] 提交失败响应: {response.text[:2000]}")
                raise RuntimeError(f"HTTP {response.status_code} {response.text[:2000]}") from exc

            try:
                response_data = response.json()
                sanitized_response_data = self.sanitize_response_data(response_data)
                print(f"[CPAGPTImage2Node] 提交响应: {json.dumps(sanitized_response_data, ensure_ascii=False)}")

                task_id = self.extract_task_id(response_data)
                if task_id:
                    image_url, final_response = self.poll_task_status(session, task_id, api_key, query_url)
                    result_image = self.download_image(session, image_url)
                else:
                    result_image, image_url = self.extract_sync_result(session, response_data)
                    final_response = response_data
            finally:
                response.close()

            result_tensor = self.pil_to_tensor(result_image)
            response_text = json.dumps(
                {
                    "submit_url": submit_url,
                    "query_url": query_url,
                    "upload_url": upload_url,
                    "request_seed": seed,
                    "submit_response": sanitized_response_data,
                    "query_response": self.sanitize_response_data(final_response),
                    "uploaded_image_urls": image_urls,
                },
                ensure_ascii=False,
                indent=2,
            )

            print("[CPAGPTImage2Node] 处理完成")
            return result_tensor, image_url, response_text
        except Exception as exc:
            if self.is_comfy_interrupt(exc):
                raise
            print(f"[CPAGPTImage2Node] 执行失败: {exc}")
            raise Exception(f"CPAGPTImage2Node 执行失败: {exc}") from exc
        finally:
            self.close_http_session()


NODE_CLASS_MAPPINGS = {
    "CPAGPTImage2GenerationNode": CPAGPTImage2GenerationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CPAGPTImage2GenerationNode": "CPA gpt image2",
}
