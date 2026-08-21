import io
import json
from typing import Any, Dict, List, Tuple
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests
from PIL import Image

try:
    import torch
except (ImportError, OSError):
    torch = None


class APIMartOpenAIResponsesLLMNode:
    """通过 APIMart Responses/Chat Completions API 生成文本，支持图片上传。"""

    DEFAULT_BASE_URL = "https://api.apimart.ai/v1"
    DEFAULT_UPLOAD_URL = "https://api.apimart.ai/v1/uploads/images"
    MODEL_CHOICES = ["gpt-5.6-luna", "gpt-5.6-sol"]
    API_FORMAT_CHOICES = ["responses", "comp"]
    THINKING_CHOICES = ["medium", "high", "xhigh"]
    IMAGE_MODE_CHOICES = ["none", "upload"]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        optional_images = {f"image_{index}": ("IMAGE",) for index in range(1, 11)}
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "password": True}),
                "base_url": ("STRING", {"multiline": False, "default": cls.DEFAULT_BASE_URL}),
                "api_format": (cls.API_FORMAT_CHOICES, {"default": "responses"}),
                "model": (cls.MODEL_CHOICES, {"default": cls.MODEL_CHOICES[0]}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True}),
                "thinking_mode": (cls.THINKING_CHOICES, {"default": "high"}),
                "image_mode": (cls.IMAGE_MODE_CHOICES, {"default": "none"}),
            },
            "optional": {
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 131072}),
                "stream": ("BOOLEAN", {"default": False}),
                "upload_url": ("STRING", {"multiline": False, "default": cls.DEFAULT_UPLOAD_URL}),
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

    def build_responses_url(self, base_url: str) -> str:
        normalized = self.normalize_base_url(base_url)
        if normalized.endswith("/responses"):
            return normalized
        return f"{normalized}/responses"

    def build_comp_url(self, base_url: str) -> str:
        normalized = self.normalize_base_url(base_url)
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    def resolve_upload_url(self, upload_url: str) -> str:
        normalized = upload_url.strip() or self.DEFAULT_UPLOAD_URL
        if "://" not in normalized:
            normalized = f"https://{normalized.lstrip('/')}"
        parts = urlsplit(normalized)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
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
        if image_array.ndim != 3 or image_array.size == 0:
            raise ValueError(f"图像张量无效: {image_array.shape}")
        if image_array.shape[2] not in (3, 4):
            raise ValueError(f"不支持的图像通道数: {image_array.shape[2]}")
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
        else:
            image_array = image_array.clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image_array, mode="RGBA" if image_array.shape[2] == 4 else "RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def split_image_input(self, image: Any) -> List[Any]:
        """将 ComfyUI IMAGE batch 拆为独立图像，避免批次只发送第一张。"""
        shape = getattr(image, "shape", None)
        if shape is not None and len(shape) == 4 and int(shape[0]) > 1:
            return [image[index : index + 1] for index in range(int(shape[0]))]
        return [image]

    def collect_images(self, **kwargs: Any) -> List[Any]:
        images: List[Any] = []
        for index in range(1, 11):
            image = kwargs.get(f"image_{index}")
            if image is not None:
                images.extend(self.split_image_input(image))
        return images

    def upload_image(
        self,
        image_tensor: Any,
        api_key: str,
        upload_url: str,
        index: int,
        timeout: int,
    ) -> str:
        image_bytes = self.tensor_to_png_bytes(image_tensor)
        if len(image_bytes) > 20 * 1024 * 1024:
            raise ValueError(f"第 {index} 张图片超过 APIMart 20 MB 限制")
        response = requests.post(
            upload_url,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (f"apimart_llm_{index}.png", image_bytes, "image/png")},
            timeout=timeout,
        )
        try:
            self.raise_for_status_with_body(response, f"第 {index} 张图片上传")
            response_data = response.json()
            image_url = response_data.get("url") if isinstance(response_data, dict) else None
            if not isinstance(image_url, str) or not image_url.startswith(("http://", "https://")):
                raise ValueError(f"APIMart 上传响应缺少有效的 url: {response_data}")
            print(f"[APIMartOpenAIResponsesLLMNode] 图片 {index} 上传完成", flush=True)
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
        if image_mode == "none":
            return []
        if image_mode != "upload":
            raise ValueError(f"不支持的 image_mode: {image_mode}")
        if not images:
            raise ValueError("image_mode=upload 时至少需要 1 张输入图片")
        urls = [
            self.upload_image(image, api_key, upload_url, index, timeout)
            for index, image in enumerate(images, start=1)
        ]
        return [{"type": "input_image", "image_url": url} for url in urls]

    def build_payload(
        self,
        api_format: str,
        model: str,
        system_prompt: str,
        prompt: str,
        thinking_mode: str,
        image_parts: List[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        stream: bool,
    ) -> Dict[str, Any]:
        if model not in self.MODEL_CHOICES:
            raise ValueError(f"不支持的 APIMart 模型: {model}")
        if api_format not in self.API_FORMAT_CHOICES:
            raise ValueError(f"不支持的 api_format: {api_format}")
        if thinking_mode not in self.THINKING_CHOICES:
            raise ValueError(f"不支持的 thinking_mode: {thinking_mode}")
        if api_format == "comp":
            content: Any = [{"type": "text", "text": prompt}]
            for image_part in image_parts:
                content.append(
                    {"type": "image_url", "image_url": {"url": image_part["image_url"]}}
                )
            messages: List[Dict[str, Any]] = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content})
            return {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "reasoning_effort": thinking_mode,
                "stream": bool(stream),
            }

        input_items: List[Dict[str, Any]] = []
        if system_prompt.strip():
            input_items.append(
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]}
            )
        input_items.append(
            {"role": "user", "content": [{"type": "input_text", "text": prompt}, *image_parts]}
        )
        payload: Dict[str, Any] = {
            "model": model,
            "input": input_items,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": bool(stream),
            "reasoning": {"effort": thinking_mode},
        }
        return payload

    def extract_text(self, response_data: Dict[str, Any]) -> str:
        output_text = response_data.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text
        choices = response_data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                if text:
                    return text
        output = response_data.get("output")
        if isinstance(output, list):
            text = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    text.extend(part.get("text", "") for part in content if isinstance(part, dict))
            if "".join(text):
                return "".join(text)
        nested_data = response_data.get("data")
        if isinstance(nested_data, dict):
            try:
                return self.extract_text(nested_data)
            except ValueError:
                pass
        raise ValueError(f"APIMart 响应缺少可读文本: {response_data}")

    def extract_stream_delta(self, event_data: Dict[str, Any]) -> str:
        delta = event_data.get("delta")
        if isinstance(delta, str):
            return delta
        choices = event_data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            delta_data = choice.get("delta", {})
            if isinstance(delta_data, dict) and isinstance(delta_data.get("content"), str):
                return delta_data["content"]
            message = choice.get("message", {})
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
        return ""

    def read_sse_response(self, response: requests.Response, timeout: int) -> Dict[str, Any]:
        text_parts: List[str] = []
        events: List[Dict[str, Any]] = []
        event_name = "message"
        data_lines: List[str] = []

        def finish_event() -> bool:
            nonlocal event_name, data_lines
            if not data_lines:
                return False
            raw_data = "\n".join(data_lines).strip()
            data_lines = []
            if raw_data == "[DONE]":
                return True
            try:
                event_data = json.loads(raw_data)
            except json.JSONDecodeError as exc:
                raise ValueError(f"APIMart SSE 返回无效 JSON: {raw_data[:1000]}") from exc
            if isinstance(event_data, dict):
                event_type = event_data.get("type") if event_name == "message" else event_name
                events.append({"event": event_type or "message"})
                delta = self.extract_stream_delta(event_data)
                if delta:
                    text_parts.append(delta)
                if event_type in {"response.completed", "response.done", "done"}:
                    return True
            event_name = "message"
            return False

        for raw_line in response.iter_lines(decode_unicode=True):
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
            line = (line or "").rstrip("\r")
            if not line:
                if finish_event():
                    break
            elif line.startswith("event:"):
                event_name = line[6:].lstrip() or "message"
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        else:
            finish_event()

        text = "".join(text_parts)
        if not text:
            raise ValueError(f"APIMart SSE 完成但没有可读文本，事件数={len(events)}")
        return {"output_text": text, "events": events, "stream": True}

    def raise_for_status_with_body(self, response: requests.Response, context: str) -> None:
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            body = getattr(response, "text", "") or "<empty>"
            raise RuntimeError(f"APIMart {context}失败（HTTP {response.status_code}）: {body[:4000]}") from exc

    def generate(
        self,
        api_key: str,
        base_url: str,
        model: str,
        system_prompt: str,
        prompt: str,
        image_mode: str,
        api_format: str,
        thinking_mode: str,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        if not api_key.strip():
            raise ValueError("api_key 不能为空")
        if not prompt.strip():
            raise ValueError("prompt 不能为空")
        timeout = int(kwargs.get("request_timeout", 300))
        stream = bool(kwargs.get("stream", False))
        if api_format not in self.API_FORMAT_CHOICES:
            raise ValueError(f"不支持的 api_format: {api_format}")
        images = self.collect_images(**kwargs)
        upload_api_key = kwargs.get("upload_api_key", "").strip() or api_key
        upload_url = self.resolve_upload_url(kwargs.get("upload_url", ""))
        image_parts = self.build_image_parts(image_mode, images, upload_api_key, upload_url, timeout)
        payload = self.build_payload(
            api_format=api_format,
            model=model,
            system_prompt=system_prompt,
            prompt=prompt,
            thinking_mode=thinking_mode,
            image_parts=image_parts,
            temperature=float(kwargs.get("temperature", 1.0)),
            top_p=float(kwargs.get("top_p", 1.0)),
            max_tokens=int(kwargs.get("max_tokens", 4096)),
            stream=stream,
        )
        request_url = self.build_responses_url(base_url) if api_format == "responses" else self.build_comp_url(base_url)
        print(
            f"[APIMartOpenAIResponsesLLMNode] 提交请求: format={api_format}, model={model}, "
            f"thinking={thinking_mode}, images={len(images)}, stream={stream}, url={request_url}",
            flush=True,
        )
        response = requests.post(
            request_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
            },
            json=payload,
            timeout=timeout,
            stream=stream,
        )
        try:
            self.raise_for_status_with_body(response, "Responses 请求")
            if stream:
                response_data = self.read_sse_response(response, timeout)
            else:
                response_data = response.json()
        finally:
            response.close()
        text = self.extract_text(response_data)
        return text, json.dumps(response_data, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "APIMartOpenAIResponsesLLMNode": APIMartOpenAIResponsesLLMNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APIMartOpenAIResponsesLLMNode": "APIMart OpenAI Responses LLM",
}
