import base64
import io
import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


class ReachGPTImage2SyncGenerationNode:
    GENERATIONS_URL = "https://api.reachapi.ai/v1/images/generations"
    EDITS_URL = "https://api.reachapi.ai/v1/images/edits"
    MODEL_NAME = "gpt-image-2"
    SIZE_PATTERN = re.compile(r"^(\d+)x(\d+)$")
    STANDARD_SIZE_MAP = {
        ("1k", "1:1"): "1024x1024",
        ("1k", "3:2"): "1536x1024",
        ("1k", "2:3"): "1024x1536",
        ("1k", "4:3"): "1536x1152",
        ("1k", "3:4"): "1152x1536",
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

    def get_direct_session(self) -> requests.Session:
        if not hasattr(self, "_direct_session"):
            session = requests.Session()
            session.trust_env = False
            self._direct_session = session
        return self._direct_session

    def request_with_proxy_fallback(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        try:
            return requests.request(method, url, **kwargs)
        except requests.exceptions.ProxyError as exc:
            print(f"[ReachGPTImage2SyncNode] 系统代理请求失败，改为直连重试: {exc}")
            session = self.get_direct_session()
            return session.request(method, url, **kwargs)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mode": (["text-to-image", "image-to-image"], {"default": "text-to-image"}),
                "api_key": ("STRING", {"multiline": False}),
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
    ) -> None:
        if mode == "image-to-image" and not reference_images:
            raise ValueError("图像编辑模式至少需要 1 张参考图")

        if mode == "text-to-image" and reference_images:
            raise ValueError("文生图模式下请不要传参考图")

        if len(reference_images) > 14:
            raise ValueError("参考图最多支持 14 张")

        if background == "transparent" and output_format == "jpeg":
            raise ValueError("background=transparent 不支持 output_format=jpeg")

        normalized_size = custom_size.strip()
        if normalized_size:
            self.validate_size(normalized_size)
            return

        if (resolution, aspect_ratio) not in self.STANDARD_SIZE_MAP:
            raise ValueError("当前 resolution 与 aspect_ratio 组合不在标准映射表中，请改用 custom_size")

    def build_size_value(self, resolution: str, aspect_ratio: str, custom_size: str) -> str:
        normalized_size = custom_size.strip()
        if normalized_size:
            return normalized_size
        return self.STANDARD_SIZE_MAP[(resolution, aspect_ratio)]

    def build_generation_payload(
        self,
        prompt: str,
        model: str,
        size: str,
        quality: str,
        background: str,
        moderation: str,
        output_format: str,
        seed: int,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "background": background,
            "moderation": moderation,
            "output_format": output_format,
        }
        if seed > 0:
            payload["seed"] = seed
        return payload

    def build_edit_form_data(
        self,
        prompt: str,
        model: str,
        size: str,
        quality: str,
        background: str,
        moderation: str,
        output_format: str,
        reference_images: List[Any],
        seed: int,
    ) -> Tuple[Dict[str, str], List[Tuple[str, Tuple[str, bytes, str]]]]:
        data = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "background": background,
            "moderation": moderation,
            "output_format": output_format,
        }
        if seed > 0:
            data["seed"] = str(seed)
        files: List[Tuple[str, Tuple[str, bytes, str]]] = []
        for index, image_tensor in enumerate(reference_images, start=1):
            files.append(
                (
                    "image",
                    (f"reach_gpt_image2_sync_{index}.png", self.tensor_to_png_bytes(image_tensor), "image/png"),
                )
            )
        return data, files

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

    def download_image(self, image_url: str) -> Image.Image:
        print(f"[ReachGPTImage2SyncNode] 下载结果图片: {image_url}")
        try:
            response = self.request_with_proxy_fallback("GET", image_url, timeout=30)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as exc:
            raise RuntimeError(f"下载结果图片失败: {exc}") from exc

    def extract_sync_result(self, response_data: Dict[str, Any]) -> Tuple[Image.Image, str]:
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
            return self.download_image(image_url), image_url

        raise ValueError(f"接口未返回可用图片数据: {first_image}")

    def generate(
        self,
        mode: str,
        api_key: str,
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
        try:
            print(f"[ReachGPTImage2SyncNode] 开始生成，模式: {mode}")
            reference_images = self.collect_images(**kwargs)
            custom_size = kwargs.get("custom_size", "")

            self.validate_inputs(
                mode=mode,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                custom_size=custom_size,
                reference_images=reference_images,
                background=background,
                output_format=output_format,
            )

            size = self.build_size_value(resolution, aspect_ratio, custom_size)
            headers = {"Authorization": f"Bearer {api_key}"}

            if mode == "image-to-image":
                submit_url = self.EDITS_URL
                data, files = self.build_edit_form_data(
                    prompt=prompt,
                    model=model,
                    size=size,
                    quality=quality,
                    background=background,
                    moderation=moderation,
                    output_format=output_format,
                    reference_images=reference_images,
                    seed=seed,
                )
                response = self.request_with_proxy_fallback(
                    "POST",
                    submit_url,
                    data=data,
                    files=files,
                    headers=headers,
                    timeout=360,
                )
            else:
                submit_url = self.GENERATIONS_URL
                payload = self.build_generation_payload(
                    prompt=prompt,
                    model=model,
                    size=size,
                    quality=quality,
                    background=background,
                    moderation=moderation,
                    output_format=output_format,
                    seed=seed,
                )
                response = self.request_with_proxy_fallback(
                    "POST",
                    submit_url,
                    json=payload,
                    headers={**headers, "Content-Type": "application/json"},
                    timeout=360,
                )

            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                print(f"[ReachGPTImage2SyncNode] 提交失败响应: {response.text[:2000]}")
                raise RuntimeError(f"HTTP {response.status_code} {response.text[:2000]}") from exc

            response_data = response.json()
            sanitized_response_data = self.sanitize_response_data(response_data)
            print(f"[ReachGPTImage2SyncNode] 提交响应: {json.dumps(sanitized_response_data, ensure_ascii=False)}")

            result_image, image_url = self.extract_sync_result(response_data)
            result_tensor = self.pil_to_tensor(result_image)
            response_text = json.dumps(
                {
                    "submit_url": submit_url,
                    "mode": mode,
                    "request_seed": seed,
                    "reference_image_count": len(reference_images),
                    "submit_response": sanitized_response_data,
                },
                ensure_ascii=False,
                indent=2,
            )

            print("[ReachGPTImage2SyncNode] 处理完成")
            return result_tensor, image_url, response_text
        except Exception as exc:
            print(f"[ReachGPTImage2SyncNode] 执行失败: {exc}")
            raise Exception(f"ReachGPTImage2SyncNode 执行失败: {exc}") from exc


NODE_CLASS_MAPPINGS = {
    "ReachGPTImage2SyncGenerationNode": ReachGPTImage2SyncGenerationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReachGPTImage2SyncGenerationNode": "reach gpt image2 sync",
}
