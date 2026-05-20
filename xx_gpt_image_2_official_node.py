import base64
import io
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import numpy as np
import requests
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


class xxGPTImage2OfficialGenerationNode:
    """ComfyUI node for GPT-Image-2 official generation via APImart."""

    HIGH_RES_SIZES = {"16:9", "9:16", "2:1", "1:2", "21:9", "9:21"}

    @staticmethod
    def derive_query_url(base_url: str) -> str:
        parsed = urlparse(base_url)
        parts = parsed.path.rstrip("/").split("/")
        new_path = "/".join(parts[:-2] + ["tasks"])
        return urlunparse(parsed._replace(path=new_path))

    def __init__(self):
        self.poll_interval = 4
        self.max_polls = 75
        self.first_poll_delay = 10

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mode": (["text-to-image", "image-to-image"], {"default": "text-to-image"}),
                "api_key": ("STRING", {"multiline": False}),
                "base_url": ("STRING", {"multiline": False, "default": "https://api.apimart.ai/v1/images/generations"}),
                "prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"multiline": False, "default": "gpt-image-2"}),
                "api_mode": (["openai", "apimart"], {"default": "openai"}),
                "n": (["1", "2", "3", "4"], {"default": "1"}),
                "size": ("STRING", {"multiline": False, "default": "1024x1024"}),
                "quality": (["low", "medium", "high", "auto"], {"default": "high"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "resolution": (["1k", "2k", "4k"], {"default": "1k"}),
                "background": (["auto", "opaque", "transparent"], {"default": "auto"}),
                "moderation": (["auto", "low"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100}),
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
                "mask_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "response")
    FUNCTION = "generate"
    CATEGORY = "image/generation"

    def tensor_to_base64(self, tensor: Any, max_size: int = 2048, quality: int = 85) -> str:
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

        # 压缩大图，避免 base64 载荷过大导致 API 502
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)

        # RGB 图用 JPEG 压缩（体积大幅减小），RGBA 图保持 PNG
        has_alpha = image.mode == "RGBA"
        mime_type = "png" if has_alpha else "jpeg"

        buffer = io.BytesIO()
        if has_alpha:
            image.save(buffer, format="PNG")
        else:
            image.save(buffer, format="JPEG", quality=quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{mime_type};base64,{encoded}"

    def pil_to_tensor(self, image: Image.Image) -> Any:
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        if torch is not None:
            return torch.from_numpy(img_array)
        return img_array

    def collect_images(self, **kwargs: Any) -> List[str]:
        image_urls = []
        for i in range(1, 11):
            key = f"image_{i}"
            if key in kwargs and kwargs[key] is not None:
                image_urls.append(self.tensor_to_base64(kwargs[key]))
        return image_urls

    def collect_mask(self, **kwargs: Any) -> Optional[str]:
        mask_image = kwargs.get("mask_image")
        if mask_image is None:
            return None
        return self.tensor_to_base64(mask_image)

    def validate_inputs(
        self,
        mode: str,
        size: str,
        resolution: str,
        image_urls: List[str],
        mask_url: Optional[str],
        output_format: str,
        output_compression: int,
    ) -> None:
        if mode == "image-to-image" and not image_urls:
            raise ValueError("图生图模式至少需要 1 张参考图")

        if mode == "text-to-image" and mask_url:
            raise ValueError("mask_image 仅能在图生图模式下使用")

        if mask_url and not image_urls:
            raise ValueError("使用 mask_image 时必须同时提供参考图")

        if len(image_urls) > 10:
            raise ValueError("参考图最多支持 10 张")

        if resolution in {"2k", "4k"} and size != "auto" and size not in self.HIGH_RES_SIZES:
            allowed = ", ".join(sorted(self.HIGH_RES_SIZES))
            raise ValueError(f"{resolution} 仅支持以下比例: {allowed}")

        if output_format == "png" and output_compression != 100:
            raise ValueError("PNG 输出时 output_compression 必须为 100")

    def extract_task_id(self, response_data: Dict[str, Any]) -> str:
        if response_data.get("code") != 200:
            raise ValueError(f"接口返回异常: {response_data}")

        data = response_data.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError(f"接口未返回任务信息: {response_data}")

        task_id = data[0].get("task_id")
        if not task_id:
            raise ValueError(f"响应中缺少 task_id: {response_data}")
        return task_id

    def extract_image_url(self, response_data: Dict[str, Any]) -> str:
        data = response_data.get("data", {})
        status = data.get("status")
        if status != "completed":
            raise ValueError(f"任务未完成，当前状态: {status}")

        result = data.get("result", {})
        images = result.get("images", [])
        if not images:
            raise ValueError("任务已完成，但未返回图片")

        first_image = images[0]
        url_value = first_image.get("url", [])
        if isinstance(url_value, list) and url_value:
            return url_value[0]
        if isinstance(url_value, str) and url_value:
            return url_value
        raise ValueError(f"无法解析返回图片地址: {first_image}")

    def poll_task_status(self, task_id: str, api_key: str, query_url: str) -> Tuple[str, Dict[str, Any]]:
        print(f"[GPTImage2OfficialNode] 开始轮询任务状态: {task_id}")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        time.sleep(self.first_poll_delay)

        for poll_count in range(1, self.max_polls + 1):
            try:
                response = requests.get(f"{query_url}/{task_id}", headers=headers, timeout=10)
                response.raise_for_status()
                response_data = response.json()
                print(f"[GPTImage2OfficialNode] 轮询 {poll_count}/{self.max_polls}: {json.dumps(response_data, ensure_ascii=False)}")

                if response_data.get("code") != 200:
                    raise ValueError(f"任务查询失败: {response_data}")

                data = response_data.get("data", {})
                status = data.get("status")

                if status == "completed":
                    image_url = self.extract_image_url(response_data)
                    print(f"[GPTImage2OfficialNode] 任务完成，结果地址: {image_url}")
                    return image_url, response_data

                if status in {"failed", "cancelled"}:
                    error_message = data.get("error") or data.get("message") or "未知错误"
                    raise ValueError(f"任务失败: {error_message}")

                if status in {"submitted", "pending", "processing"}:
                    time.sleep(self.poll_interval)
                    continue

                print(f"[GPTImage2OfficialNode] 遇到未识别状态 {status}，继续轮询")
                time.sleep(self.poll_interval)
            except requests.exceptions.RequestException as exc:
                print(f"[GPTImage2OfficialNode] 查询任务失败: {exc}")
                if poll_count == self.max_polls:
                    raise RuntimeError(f"轮询任务状态失败: {exc}") from exc
                time.sleep(self.poll_interval)

        raise TimeoutError(f"任务在 {self.first_poll_delay + self.max_polls * self.poll_interval} 秒内未完成")

    def download_image(self, image_url: str) -> Image.Image:
        print(f"[GPTImage2OfficialNode] 下载结果图片: {image_url}")
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as exc:
            raise RuntimeError(f"下载结果图片失败: {exc}") from exc

    def _extract_openai_image(self, response_data: Dict[str, Any]) -> Tuple[Image.Image, str]:
        data = response_data.get("data", [])
        if not isinstance(data, list) or not data:
            raise ValueError(f"响应中缺少 data 字段: {response_data}")

        item = data[0]
        b64 = item.get("b64_json")
        if b64:
            image_bytes = base64.b64decode(b64)
            return Image.open(io.BytesIO(image_bytes)), ""

        url = item.get("url")
        if url:
            return self.download_image(url), url

        raise ValueError(f"无法从 OpenAI 响应中提取图片: {item}")

    def _is_task_based_response(self, response_data: Dict[str, Any]) -> bool:
        data = response_data.get("data")
        if isinstance(data, list) and data and "task_id" in data[0]:
            return True
        return False

    def _build_payload(
        self,
        api_mode: str,
        prompt: str,
        model: str,
        n: str,
        size: str,
        quality: str,
        seed: int,
        image_urls: List[str],
        mask_url: Optional[str],
        **apimart_opts: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": int(n),
            "size": size,
            "quality": quality,
        }

        if seed > 0:
            payload["seed"] = seed

        if api_mode == "apimart":
            resolution = apimart_opts.get("resolution", "1k")
            background = apimart_opts.get("background", "auto")
            moderation = apimart_opts.get("moderation", "auto")
            output_format = apimart_opts.get("output_format", "png")
            output_compression = apimart_opts.get("output_compression", 100)

            payload["resolution"] = resolution
            payload["background"] = background
            payload["moderation"] = moderation
            payload["output_format"] = output_format
            if output_format in {"jpeg", "webp"}:
                payload["output_compression"] = output_compression

        if image_urls:
            payload["image_urls"] = image_urls
        if mask_url:
            payload["mask_url"] = mask_url

        return payload

    def generate(
        self,
        mode: str,
        api_key: str,
        base_url: str,
        prompt: str,
        model: str,
        api_mode: str,
        n: str,
        size: str,
        quality: str,
        seed: int,
        **kwargs: Any,
    ) -> Tuple[Any, str, str]:
        try:
            print(f"[GPTImage2OfficialNode] 开始生成，模式: {mode}, API: {api_mode}")
            api_url = base_url.strip().rstrip("/")
            print(f"[GPTImage2OfficialNode] API URL: {api_url}")

            resolution = kwargs.get("resolution", "1k")
            background = kwargs.get("background", "auto")
            moderation = kwargs.get("moderation", "auto")
            output_format = kwargs.get("output_format", "png")
            output_compression = kwargs.get("output_compression", 100)

            image_urls = self.collect_images(**kwargs)
            mask_url = self.collect_mask(**kwargs)
            if image_urls or mask_url:
                self.validate_inputs(mode, size, resolution, image_urls, mask_url, output_format, output_compression)

            payload = self._build_payload(
                api_mode=api_mode,
                prompt=prompt,
                model=model,
                n=n,
                size=size,
                quality=quality,
                seed=seed,
                image_urls=image_urls,
                mask_url=mask_url,
                resolution=resolution,
                background=background,
                moderation=moderation,
                output_format=output_format,
                output_compression=output_compression,
            )

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            print(f"[GPTImage2OfficialNode] 提交请求: {api_url}")
            response = requests.post(api_url, json=payload, headers=headers, timeout=300)
            response.raise_for_status()

            response_data = response.json()
            print(f"[GPTImage2OfficialNode] 提交响应: {json.dumps(response_data, ensure_ascii=False)}")

            if "error" in response_data:
                error_message = response_data.get("error", {}).get("message", "未知错误")
                raise ValueError(f"接口返回错误: {error_message}")

            if self._is_task_based_response(response_data):
                print("[GPTImage2OfficialNode] 检测到异步任务模式")
                query_url = self.derive_query_url(api_url)
                task_id = self.extract_task_id(response_data)
                image_url, final_response = self.poll_task_status(task_id, api_key, query_url)
                result_image = self.download_image(image_url)
                response_text = json.dumps(final_response, ensure_ascii=False, indent=2)
            else:
                print("[GPTImage2OfficialNode] 检测到 OpenAI 直返模式")
                result_image, image_url = self._extract_openai_image(response_data)
                response_text = json.dumps(response_data, ensure_ascii=False, indent=2)

            result_tensor = self.pil_to_tensor(result_image)
            print("[GPTImage2OfficialNode] 处理完成")
            return result_tensor, image_url, response_text
        except Exception as exc:
            print(f"[GPTImage2OfficialNode] 执行失败: {exc}")
            raise Exception(f"GPTImage2OfficialNode 执行失败: {exc}") from exc


NODE_CLASS_MAPPINGS = {
    "xxGPTImage2OfficialGenerationNode": xxGPTImage2OfficialGenerationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "xxGPTImage2OfficialGenerationNode": "GPT Image 2 Official (Custom URL)",
}
