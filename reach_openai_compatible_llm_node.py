import base64
import concurrent.futures
import io
import json
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests
from PIL import Image

try:
    import torch
except (ImportError, OSError):
    torch = None

try:
    import comfy.model_management as comfy_model_management
except ImportError:
    comfy_model_management = None


class ReachOpenAICompatibleLLMNode:
    """ComfyUI node for ReachAPI OpenAI-compatible LLM Responses and Chat APIs."""

    DEFAULT_BASE_URL = "https://api.reachapi.ai/v1"
    FILE_UPLOAD_URL = "https://file.reachapi.ai/file/uploads"
    MODEL_CHOICES = ["gpt-5.6-sol", "gpt-5.6-luna"]
    THINKING_CHOICES = ["medium", "high", "xhigh"]
    IMAGE_MODE_CHOICES = ["none", "base64", "reach_upload"]
    API_FORMAT_CHOICES = ["responses", "chat_completions"]
    ACTIVE_STATUSES = {"queued", "pending", "processing", "generating", "in_progress", "running"}
    SUCCESS_STATUSES = {"success", "succeeded", "completed", "done"}
    FAILED_STATUSES = {"failed", "failure", "cancelled", "canceled", "error"}

    def __init__(self):
        self.poll_interval = 4
        self.max_polls = 90
        self.first_poll_delay = 2
        self._session_lock = threading.Lock()
        self._active_session: Optional[requests.Session] = None

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
                "image_mode": (cls.IMAGE_MODE_CHOICES, {"default": "reach_upload"}),
            },
            "optional": {
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 131072}),
                "upload_url": ("STRING", {"multiline": False, "default": cls.FILE_UPLOAD_URL}),
                "upload_api_key": ("STRING", {"multiline": False, "password": True, "default": ""}),
                "request_timeout": ("INT", {"default": 300, "min": 10, "max": 1800}),
                "poll_interval": ("INT", {"default": 4, "min": 1, "max": 60}),
                "max_polls": ("INT", {"default": 90, "min": 1, "max": 600}),
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

    def build_submit_url(self, base_url: str, api_format: str) -> str:
        normalized = self.normalize_base_url(base_url)
        endpoint = "/responses" if api_format == "responses" else "/chat/completions"
        if normalized.endswith(endpoint):
            return normalized
        return f"{normalized}{endpoint}"

    def build_query_url(self, base_url: str) -> str:
        normalized = self.normalize_base_url(base_url)
        parts = urlsplit(normalized)
        path = parts.path.rstrip("/")
        if path.endswith("/responses") or path.endswith("/chat/completions"):
            path = path.rsplit("/", 1)[0]
        query_path = f"{path}/tasks" if path else "/tasks"
        return urlunsplit((parts.scheme, parts.netloc, query_path, parts.query, parts.fragment))

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
                print("[ReachOpenAICompatibleLLMNode] 检测到 ComfyUI 关闭任务，正在断开 HTTP 连接")
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
        progress_label = kwargs.pop("progress_label", f"{method} {url}")
        progress_interval = float(kwargs.pop("progress_interval", 10))
        timeout = kwargs.pop("timeout", (10, total_timeout))
        if isinstance(timeout, tuple):
            timeout = (timeout[0], min(timeout[1], total_timeout))
        else:
            timeout = min(float(timeout), total_timeout)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(session.request, method, url, timeout=timeout, **kwargs)
            next_progress = time.monotonic() + max(progress_interval, 1.0)
            while True:
                self.check_interrupted()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    future.cancel()
                    raise requests.exceptions.ReadTimeout(f"请求超时：{total_timeout:g} 秒内未收到响应")
                try:
                    return future.result(timeout=min(0.2, remaining))
                except concurrent.futures.TimeoutError:
                    now = time.monotonic()
                    if now >= next_progress:
                        elapsed = total_timeout - remaining
                        print(
                            f"[ReachOpenAICompatibleLLMNode] 等待响应头: {progress_label}，"
                            f"已等待 {elapsed:.0f}s/{total_timeout:.0f}s",
                            flush=True,
                        )
                        next_progress = now + max(progress_interval, 1.0)
                    continue
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

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
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
        else:
            image_array = image_array.clip(0, 255).astype(np.uint8)
        if image_array.shape[2] == 3:
            image = Image.fromarray(image_array, mode="RGB")
        elif image_array.shape[2] == 4:
            image = Image.fromarray(image_array, mode="RGBA")
        else:
            raise ValueError(f"不支持的图像通道数: {image_array.shape[2]}")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def collect_images(self, **kwargs: Any) -> List[Any]:
        return [kwargs[f"image_{index}"] for index in range(1, 11) if kwargs.get(f"image_{index}") is not None]

    def encode_image_data_url(self, image_tensor: Any) -> str:
        encoded = base64.b64encode(self.tensor_to_png_bytes(image_tensor)).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def upload_image(
        self,
        session: requests.Session,
        image_tensor: Any,
        api_key: str,
        upload_url: str,
        index: int,
        timeout: int,
    ) -> str:
        response = self.request_with_interrupt(
            session,
            "POST",
            upload_url,
            total_timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (f"reach_openai_llm_{index}.png", self.tensor_to_png_bytes(image_tensor), "image/png")},
        )
        try:
            response.raise_for_status()
            response_data = response.json()
            if response_data.get("code") != 200:
                raise ValueError(f"图像上传失败: {response_data}")
            data = response_data.get("data")
            image_url = data.get("url") if isinstance(data, dict) else None
            if not isinstance(image_url, str) or not image_url.startswith("https://"):
                raise ValueError(f"图像上传响应缺少有效的 data.url: {response_data}")
            return image_url
        finally:
            response.close()

    def build_image_parts(
        self,
        session: requests.Session,
        image_mode: str,
        images: List[Any],
        api_key: str,
        upload_url: str,
        timeout: int,
        api_format: str,
    ) -> List[Dict[str, Any]]:
        if image_mode == "none":
            return []
        if not images:
            raise ValueError(f"image_mode={image_mode} 时至少需要 1 张输入图片")
        if image_mode == "base64":
            urls = [self.encode_image_data_url(image) for image in images]
        elif image_mode == "reach_upload":
            urls = [
                self.upload_image(session, image, api_key, upload_url, index, timeout)
                for index, image in enumerate(images, start=1)
            ]
        else:
            raise ValueError(f"不支持的 image_mode: {image_mode}")
        if api_format == "responses":
            return [{"type": "input_image", "image_url": url} for url in urls]
        return [{"type": "image_url", "image_url": {"url": url}} for url in urls]

    def build_payload(
        self,
        api_format: str,
        model: str,
        system_prompt: str,
        prompt: str,
        thinking_mode: str,
        image_parts: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        if api_format not in self.API_FORMAT_CHOICES:
            raise ValueError(f"不支持的 api_format: {api_format}")
        if model not in self.MODEL_CHOICES:
            raise ValueError(f"不支持的 model: {model}")
        if thinking_mode not in self.THINKING_CHOICES:
            raise ValueError(f"不支持的 thinking_mode: {thinking_mode}")

        if api_format == "responses":
            content = [{"type": "input_text", "text": prompt}, *image_parts]
            payload: Dict[str, Any] = {
                "model": model,
                "input": [{"role": "user", "content": content}],
                "reasoning": {"effort": thinking_mode},
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                # Reach Responses API 使用 SSE 返回长耗时任务的增量结果。
                "stream": True,
            }
            if system_prompt.strip():
                payload["instructions"] = system_prompt
            return payload

        content = [{"type": "text", "text": prompt}, *image_parts]
        messages: List[Dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": thinking_mode,
        }

    def _finish_sse_event(
        self,
        event_name: str,
        data_lines: List[str],
        events: List[Dict[str, Any]],
        text_parts: List[str],
    ) -> Optional[Dict[str, Any]]:
        """解析一个 SSE 事件，并返回事件 JSON。"""
        if not data_lines:
            return None
        raw_data = "\n".join(data_lines).strip()
        if raw_data == "[DONE]":
            print("[ReachOpenAICompatibleLLMNode] SSE completed: [DONE]", flush=True)
            return {"event": event_name, "data": "[DONE]"}
        try:
            event_data = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Reach SSE 返回了无效 JSON（event={event_name}）: {raw_data[:500]}") from exc
        if not isinstance(event_data, dict):
            event_data = {"value": event_data}
        # 某些 OpenAI 兼容网关把事件类型放在 data.type，而不是 event 行。
        if event_name == "message" and isinstance(event_data.get("type"), str):
            event_name = event_data["type"]
        event = {"event": event_name, "data": event_data}
        events.append(event)
        print(f"[ReachOpenAICompatibleLLMNode] SSE event: {event_name}", flush=True)

        if event_name in {"response.output_text.delta", "response.text.delta"}:
            delta = event_data.get("delta")
            if isinstance(delta, str):
                text_parts.append(delta)
                print(
                    f"[ReachOpenAICompatibleLLMNode] 已接收文本增量: "
                    f"{len(delta)} chars（累计 {sum(len(part) for part in text_parts)} chars）",
                    flush=True,
                )
        elif event_name in {"response.output_text.done", "response.text.done"}:
            done_text = event_data.get("text")
            if isinstance(done_text, str) and done_text and not text_parts:
                text_parts.append(done_text)
        if event_name in {"response.failed", "response.error", "error"} or event_data.get("error"):
            raise RuntimeError(f"Reach Responses SSE 任务失败: {event_data}")
        return event

    def read_sse_response(self, response: requests.Response, total_timeout: float = 300) -> Dict[str, Any]:
        """读取 Reach Responses 的 SSE，持续输出进度并合并文本增量。"""
        events: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        current_event = "message"
        data_lines: List[str] = []
        plain_lines: List[str] = []
        last_data: Dict[str, Any] = {}

        # 使用较小的 chunk，避免代理将多个 SSE 事件缓冲后才交给 ComfyUI。
        # 在独立线程读取 socket，主线程每秒检查中断并输出等待进度。否则代理长时间
        # 缓冲 SSE 时，ComfyUI 看不到任何日志，也无法及时响应取消任务。
        line_queue: "queue.Queue[Any]" = queue.Queue()
        def read_lines() -> None:
            try:
                for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                    line_queue.put(("line", line))
                line_queue.put(("done", None))
            except BaseException as exc:
                line_queue.put(("error", exc))

        reader = threading.Thread(target=read_lines, name="reach-sse-reader", daemon=True)
        reader.start()
        started_at = time.monotonic()
        last_heartbeat = started_at
        while True:
            self.check_interrupted()
            elapsed = time.monotonic() - started_at
            if elapsed >= total_timeout:
                raise requests.exceptions.ReadTimeout(
                    f"SSE 流在 {total_timeout:g} 秒内未完成（已收到 {len(events)} 个事件）"
                )
            try:
                item_type, item_value = line_queue.get(timeout=1.0)
            except queue.Empty:
                if time.monotonic() - last_heartbeat >= 10:
                    print(
                        f"[ReachOpenAICompatibleLLMNode] SSE 仍在处理中，已等待 {elapsed:.0f}s，"
                        f"已收到 {len(events)} 个事件",
                        flush=True,
                    )
                    last_heartbeat = time.monotonic()
                continue
            if item_type == "error":
                raise item_value
            if item_type == "done":
                break
            raw_line = item_value
            if isinstance(raw_line, bytes):
                raw_line = raw_line.decode("utf-8", errors="replace")
            line = raw_line.rstrip("\r") if raw_line is not None else ""
            if not line:
                event = self._finish_sse_event(current_event, data_lines, events, text_parts)
                if event is not None and isinstance(event.get("data"), dict):
                    last_data = event["data"]
                current_event = "message"
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                current_event = line[6:].lstrip() or "message"
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            else:
                # 兼容少数不返回 Content-Type 的中转站：它们可能仍返回完整 JSON。
                plain_lines.append(line)

        # 允许没有以空行结尾的最后一个事件。
        event = self._finish_sse_event(current_event, data_lines, events, text_parts)
        if event is not None and isinstance(event.get("data"), dict):
            last_data = event["data"]

        if not events and not text_parts and plain_lines:
            raw_body = "\n".join(plain_lines).strip()
            try:
                json_body = json.loads(raw_body)
            except json.JSONDecodeError:
                json_body = None
            if isinstance(json_body, dict):
                print("[ReachOpenAICompatibleLLMNode] SSE 连接返回普通 JSON，使用兼容解析", flush=True)
                return json_body

        text = "".join(text_parts)
        result: Dict[str, Any] = {
            **last_data,
            "output_text": text,
            "events": events,
            "stream": True,
        }
        if not text:
            # 某些网关只在 completed 事件中携带完整 output。
            try:
                text = self.extract_text(result)
            except ValueError:
                pass
            if text:
                result["output_text"] = text
        if not text:
            raise ValueError(f"Reach SSE 完成但没有可读文本: {result}")
        print(
            f"[ReachOpenAICompatibleLLMNode] SSE completed，累计文本: {len(text)} chars",
            flush=True,
        )
        return result

    def is_sse_response(self, response: requests.Response) -> bool:
        content_type = str(getattr(response, "headers", {}).get("Content-Type", "")).lower()
        if "text/event-stream" in content_type:
            return True
        # 有些中转站不带 Content-Type；真实 requests.Response 提供 iter_lines，
        # 测试/兼容的 JSON 响应通常没有该方法。
        return not content_type and callable(getattr(response, "iter_lines", None))

    def extract_task_id(self, response_data: Dict[str, Any]) -> Optional[str]:
        candidates = [
            response_data.get("task_id"),
            response_data.get("id") if response_data.get("object") in {"task", "response.task"} else None,
        ]
        data = response_data.get("data")
        if isinstance(data, dict):
            candidates.extend([data.get("task_id"), data.get("id")])
        return next((value for value in candidates if isinstance(value, str) and value), None)

    def get_status(self, response_data: Dict[str, Any]) -> Optional[str]:
        status = response_data.get("status")
        if isinstance(status, str):
            return status.lower()
        data = response_data.get("data")
        if isinstance(data, dict) and isinstance(data.get("status"), str):
            return data["status"].lower()
        return None

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
        for key in ("result", "response", "data"):
            nested = response_data.get(key)
            if isinstance(nested, dict):
                try:
                    return self.extract_text(nested)
                except ValueError:
                    continue
        raise ValueError(f"接口响应缺少可读文本: {response_data}")

    def poll_task_status(
        self,
        session: requests.Session,
        task_id: str,
        api_key: str,
        query_url: str,
        poll_interval: int,
        max_polls: int,
    ) -> Dict[str, Any]:
        self.interruptible_sleep(self.first_poll_delay)
        headers = {"Authorization": f"Bearer {api_key}"}
        for poll_count in range(1, max_polls + 1):
            response = self.request_with_interrupt(
                session,
                "GET",
                f"{query_url}/{task_id}",
                total_timeout=30,
                headers=headers,
            )
            try:
                response.raise_for_status()
                response_data = response.json()
                status = self.get_status(response_data)
                print(
                    f"[ReachOpenAICompatibleLLMNode] 轮询 {poll_count}/{max_polls}: "
                    f"status={status or 'unknown'}"
                )
                if status in self.SUCCESS_STATUSES:
                    return response_data
                if status in self.FAILED_STATUSES:
                    raise RuntimeError(f"Reach LLM 任务失败: {response_data.get('msg') or response_data}")
                if status not in self.ACTIVE_STATUSES and self.extract_text(response_data):
                    return response_data
            finally:
                response.close()
            self.interruptible_sleep(poll_interval)
        raise TimeoutError(f"Reach LLM 任务在 {max_polls * poll_interval} 秒内未完成: {task_id}")

    def generate(
        self,
        api_key: str,
        base_url: str,
        api_format: str,
        model: str,
        system_prompt: str,
        prompt: str,
        thinking_mode: str,
        image_mode: str,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        if not api_key.strip():
            raise ValueError("api_key 不能为空")
        if not prompt.strip():
            raise ValueError("prompt 不能为空")
        session = self.open_http_session()
        try:
            timeout = int(kwargs.get("request_timeout", 300))
            poll_interval = int(kwargs.get("poll_interval", self.poll_interval))
            max_polls = int(kwargs.get("max_polls", self.max_polls))
            images = self.collect_images(**kwargs)
            upload_url = self.resolve_upload_url(kwargs.get("upload_url", "")) if image_mode == "reach_upload" else ""
            upload_api_key = kwargs.get("upload_api_key", "").strip() or api_key
            image_parts = self.build_image_parts(
                session=session,
                image_mode=image_mode,
                images=images,
                api_key=upload_api_key,
                upload_url=upload_url,
                timeout=timeout,
                api_format=api_format,
            )
            payload = self.build_payload(
                api_format=api_format,
                model=model,
                system_prompt=system_prompt,
                prompt=prompt,
                thinking_mode=thinking_mode,
                image_parts=image_parts,
                temperature=float(kwargs.get("temperature", 1.0)),
                max_tokens=int(kwargs.get("max_tokens", 4096)),
            )
            submit_url = self.build_submit_url(base_url, api_format)
            query_url = self.build_query_url(base_url)
            print(
                f"[ReachOpenAICompatibleLLMNode] 提交任务: format={api_format}, model={model}, "
                f"images={len(images)}, url={submit_url}"
            )
            request_headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            if api_format == "responses":
                request_headers["Accept"] = "text/event-stream"
            response = self.request_with_interrupt(
                session,
                "POST",
                submit_url,
                total_timeout=timeout,
                progress_label=f"POST {submit_url}",
                headers=request_headers,
                json=payload,
                stream=api_format == "responses",
            )
            try:
                response.raise_for_status()
                if api_format == "responses" and self.is_sse_response(response):
                    print("[ReachOpenAICompatibleLLMNode] 已连接 SSE，开始接收 Reach 后台进度", flush=True)
                    final_data = self.read_sse_response(response, total_timeout=timeout)
                    submit_data = {
                        "stream": True,
                        "events": final_data.get("events", []),
                    }
                else:
                    if api_format == "responses":
                        print(
                            "[ReachOpenAICompatibleLLMNode] Reach 未返回 SSE，按普通 JSON 读取；"
                            "如果请求仍在后台处理，请检查中转站是否支持 stream=true",
                            flush=True,
                        )
                    submit_data = response.json()
                    final_data = None
            finally:
                response.close()

            task_id = self.extract_task_id(submit_data)
            if final_data is None:
                final_data = (
                    self.poll_task_status(session, task_id, api_key, query_url, poll_interval, max_polls)
                    if task_id
                    else submit_data
                )
            text = self.extract_text(final_data)
            result = {
                "submit_url": submit_url,
                "query_url": query_url,
                "task_id": task_id,
                "uploaded_image_count": len(images) if image_mode == "reach_upload" else 0,
                "submit_response": submit_data,
                "final_response": final_data,
            }
            return text, json.dumps(result, ensure_ascii=False, indent=2)
        finally:
            self.close_http_session()


NODE_CLASS_MAPPINGS = {
    "ReachOpenAICompatibleLLMNode": ReachOpenAICompatibleLLMNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReachOpenAICompatibleLLMNode": "Reach OpenAI Compatible LLM",
}
