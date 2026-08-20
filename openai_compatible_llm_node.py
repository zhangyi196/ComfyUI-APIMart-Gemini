import base64
import io
import json
from typing import Any, Dict, List
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests
from PIL import Image

try:
    import torch
except (ImportError, OSError):
    torch = None


class OpenAICompatibleLLMNode:
    """ComfyUI node for OpenAI-compatible chat completion APIs."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    FILE_UPLOAD_URL = "https://file.reachapi.ai/file/uploads"
    MODEL_CHOICES = ["gpt-5.6-sol", "gpt-5.6-luna"]
    THINKING_CHOICES = ["medium", "high", "xhigh"]
    IMAGE_MODE_CHOICES = ["none", "base64", "reach_upload"]
    API_FORMAT_CHOICES = ["chat_completions", "responses"]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        optional_images = {f"image_{index}": ("IMAGE",) for index in range(1, 11)}
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "password": True}),
                "base_url": ("STRING", {"multiline": False, "default": cls.DEFAULT_BASE_URL}),
                "model": (cls.MODEL_CHOICES, {"default": cls.MODEL_CHOICES[0]}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True}),
                "thinking_mode": (cls.THINKING_CHOICES, {"default": "medium"}),
                "image_mode": (cls.IMAGE_MODE_CHOICES, {"default": "none"}),
            },
            "optional": {
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 131072}),
                "api_format": (cls.API_FORMAT_CHOICES, {"default": "chat_completions"}),
                "upload_url": ("STRING", {"multiline": False, "default": cls.FILE_UPLOAD_URL}),
                "upload_api_key": ("STRING", {"multiline": False, "password": True, "default": ""}),
                "request_timeout": ("INT", {"default": 300, "min": 10, "max": 1800}),
                **optional_images,
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "response_json")
    FUNCTION = "generate"
    CATEGORY = "llm"

    def normalize_base_url(self, base_url: str) -> str:
        normalized = base_url.strip()
        if not normalized:
            raise ValueError("base_url 不能为空")
        return normalized.rstrip("/")

    def build_chat_url(self, base_url: str) -> str:
        normalized = self.normalize_base_url(base_url)
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    def build_responses_url(self, base_url: str) -> str:
        normalized = self.normalize_base_url(base_url)
        if normalized.endswith("/responses"):
            return normalized
        return f"{normalized}/responses"

    def resolve_upload_url(self, upload_url: str) -> str:
        normalized = upload_url.strip()
        if not normalized:
            return self.FILE_UPLOAD_URL
        if "://" not in normalized:
            normalized = f"https://{normalized.lstrip('/')}"
        parts = urlsplit(normalized)
        if not parts.scheme or not parts.netloc:
            raise ValueError("upload_url 必须是有效的 HTTP(S) 地址")
        return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), parts.query, parts.fragment))

    def tensor_to_png_bytes(self, tensor: Any) -> bytes:
        if torch is not None and isinstance(tensor, torch.Tensor):
            image_array = tensor.detach().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            image_array = tensor
        else:
            image_array = np.asarray(tensor)

        if image_array.ndim == 4:
            image_array = image_array[0]
        if image_array.ndim != 3:
            raise ValueError(f"图像张量维度不正确: {image_array.shape}")

        if image_array.size == 0:
            raise ValueError("图像张量不能为空")
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
        else:
            image_array = image_array.clip(0, 255).astype(np.uint8)

        channels = image_array.shape[2]
        if channels == 3:
            image = Image.fromarray(image_array, mode="RGB")
        elif channels == 4:
            image = Image.fromarray(image_array, mode="RGBA")
        else:
            raise ValueError(f"不支持的图像通道数: {channels}")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def collect_images(self, **kwargs: Any) -> List[Any]:
        images = []
        for index in range(1, 11):
            image = kwargs.get(f"image_{index}")
            if image is not None:
                images.append(image)
        return images

    def encode_image_data_url(self, image_tensor: Any) -> str:
        image_bytes = self.tensor_to_png_bytes(image_tensor)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def upload_image(
        self,
        image_tensor: Any,
        api_key: str,
        upload_url: str,
        index: int,
        timeout: int,
    ) -> str:
        image_bytes = self.tensor_to_png_bytes(image_tensor)
        response = requests.post(
            upload_url,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (f"openai_llm_{index}.png", image_bytes, "image/png")},
            timeout=timeout,
        )
        try:
            response.raise_for_status()
            response_data = response.json()
            if response_data.get("code") != 200:
                raise ValueError(f"图像上传失败: {response_data}")
            data = response_data.get("data")
            image_url = data.get("url") if isinstance(data, dict) else None
            if not isinstance(image_url, str) or not image_url:
                raise ValueError(f"图像上传响应缺少 data.url: {response_data}")
            if not image_url.startswith("https://"):
                raise ValueError(f"图像上传后返回的 URL 不是 HTTPS 地址: {image_url}")
            return image_url
        finally:
            response.close()

    def build_image_parts(
        self,
        image_mode: str,
        images: List[Any],
        api_key: str,
        upload_url: str,
        timeout: int,
    ) -> List[Dict[str, Any]]:
        if image_mode == "none" or not images:
            if image_mode != "none" and not images:
                raise ValueError(f"image_mode={image_mode} 时至少需要 1 张输入图片")
            return []
        if image_mode == "base64":
            return [
                {"type": "image_url", "image_url": {"url": self.encode_image_data_url(image)}}
                for image in images
            ]
        if image_mode == "reach_upload":
            return [
                {
                    "type": "image_url",
                    "image_url": {"url": self.upload_image(image, api_key, upload_url, index, timeout)},
                }
                for index, image in enumerate(images, start=1)
            ]
        raise ValueError(f"不支持的 image_mode: {image_mode}")

    def build_payload(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        thinking_mode: str,
        image_parts: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        user_content: Any = [{"type": "text", "text": prompt}]
        user_content.extend(image_parts)
        messages: List[Dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if thinking_mode not in self.THINKING_CHOICES:
            raise ValueError(f"不支持的 thinking_mode: {thinking_mode}")
        payload["reasoning_effort"] = thinking_mode
        return payload

    def build_responses_payload(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        thinking_mode: str,
        image_parts: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        if thinking_mode not in self.THINKING_CHOICES:
            raise ValueError(f"不支持的 thinking_mode: {thinking_mode}")

        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for part in image_parts:
            if part.get("type") != "image_url":
                raise ValueError(f"无法将图像内容转换为 Responses 格式: {part}")
            image_url = part.get("image_url", {}).get("url")
            if not isinstance(image_url, str) or not image_url:
                raise ValueError(f"图像内容缺少 URL: {part}")
            content.append({"type": "input_image", "image_url": image_url})

        payload: Dict[str, Any] = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": thinking_mode},
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_prompt.strip():
            payload["instructions"] = system_prompt
        return payload

    def extract_text(self, response_data: Dict[str, Any]) -> str:
        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"API 响应缺少 choices: {response_data}")
        message = choices[0].get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            if text_parts:
                return "".join(text_parts)
        if isinstance(message, dict) and isinstance(message.get("refusal"), str):
            return message["refusal"]
        raise ValueError(f"API 响应缺少可读文本: {response_data}")

    def extract_responses_text(self, response_data: Dict[str, Any]) -> str:
        output_text = response_data.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        output = response_data.get("output")
        if isinstance(output, list):
            text_parts = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        text_parts.append(part["text"])
            if text_parts:
                return "".join(text_parts)

        # Some OpenAI-compatible gateways return Chat Completions-shaped data
        # even when the Responses endpoint is selected.
        return self.extract_text(response_data)

    def generate(
        self,
        api_key: str,
        base_url: str,
        model: str,
        system_prompt: str,
        prompt: str,
        thinking_mode: str,
        image_mode: str,
        **kwargs: Any,
    ) -> tuple[str, str]:
        if not api_key.strip():
            raise ValueError("api_key 不能为空")
        if not prompt.strip():
            raise ValueError("prompt 不能为空")

        timeout = int(kwargs.get("request_timeout", 300))
        api_format = kwargs.get("api_format", "chat_completions")
        if api_format not in self.API_FORMAT_CHOICES:
            raise ValueError(f"不支持的 api_format: {api_format}")
        images = self.collect_images(**kwargs)
        upload_url = self.resolve_upload_url(kwargs.get("upload_url", "")) if image_mode == "reach_upload" else ""
        upload_api_key = kwargs.get("upload_api_key", "").strip() or api_key
        image_parts = self.build_image_parts(
            image_mode=image_mode,
            images=images,
            api_key=upload_api_key,
            upload_url=upload_url,
            timeout=timeout,
        )
        temperature = float(kwargs.get("temperature", 1.0))
        max_tokens = int(kwargs.get("max_tokens", 4096))
        if api_format == "responses":
            request_url = self.build_responses_url(base_url)
            payload = self.build_responses_payload(
                model=model,
                system_prompt=system_prompt,
                prompt=prompt,
                thinking_mode=thinking_mode,
                image_parts=image_parts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            request_url = self.build_chat_url(base_url)
            payload = self.build_payload(
                model=model,
                system_prompt=system_prompt,
                prompt=prompt,
                thinking_mode=thinking_mode,
                image_parts=image_parts,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        chat_url = self.build_chat_url(base_url)
        print(
            f"[OpenAICompatibleLLMNode] API 格式: {api_format}，请求模型: {model}，思考模式: {thinking_mode}，"
            f"图像模式: {image_mode}，图像数量: {len(images) if image_mode != 'none' else 0}"
        )
        response = requests.post(
            request_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        try:
            if not response.ok:
                raise RuntimeError(f"LLM 请求失败（HTTP {response.status_code}）: {response.text[:2000]}")
            response_data = response.json()
        finally:
            response.close()

        text = self.extract_responses_text(response_data) if api_format == "responses" else self.extract_text(response_data)
        return text, json.dumps(response_data, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "OpenAICompatibleLLMNode": OpenAICompatibleLLMNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAICompatibleLLMNode": "OpenAI Compatible LLM",
}
