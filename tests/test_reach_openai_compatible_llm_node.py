import importlib.util
from pathlib import Path
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "reach_openai_compatible_llm_node.py"
spec = importlib.util.spec_from_file_location("reach_openai_compatible_llm_node", MODULE_PATH)
reach_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(reach_module)


class FakeResponse:
    def __init__(self, response_data, status_code=200):
        self.response_data = response_data
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = str(response_data)
        self.closed = False

    def raise_for_status(self):
        if not self.ok:
            raise reach_module.requests.exceptions.HTTPError(self.text)

    def json(self):
        return self.response_data

    def close(self):
        self.closed = True


class DelayedResponse(FakeResponse):
    def __init__(self, delay):
        super().__init__({"output_text": "ok"})
        self.delay = delay

    def iter_lines(self, decode_unicode=False, **kwargs):
        import time
        time.sleep(self.delay)
        yield "event: response.output_text.delta"
        yield 'data: {"delta":"ok"}'
        yield ""


class FakeSSEResponse(FakeResponse):
    def __init__(self, lines, status_code=200):
        super().__init__(None, status_code)
        self.headers = {"Content-Type": "text/event-stream"}
        self.lines = lines

    def iter_lines(self, decode_unicode=False, **kwargs):
        yield from self.lines


class FakeSession:
    def close(self):
        pass


class ReachOpenAICompatibleLLMNodeTests(unittest.TestCase):
    def test_build_urls_use_reach_async_endpoints(self):
        node = reach_module.ReachOpenAICompatibleLLMNode()
        self.assertEqual(
            node.build_submit_url("https://api.reachapi.ai/v1", "responses"),
            "https://api.reachapi.ai/v1/responses",
        )
        self.assertEqual(
            node.build_query_url("https://api.reachapi.ai/v1"),
            "https://api.reachapi.ai/v1/tasks",
        )

    def test_build_responses_payload_uses_input_images(self):
        node = reach_module.ReachOpenAICompatibleLLMNode()
        payload = node.build_payload(
            api_format="responses",
            model="gpt-5.6-sol",
            system_prompt="system",
            prompt="make a prompt",
            thinking_mode="high",
            image_parts=[{"type": "input_image", "image_url": "https://cdn/1.png"}],
            temperature=1.0,
            max_tokens=100,
        )
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["input"][0]["content"][1]["type"], "input_image")
        self.assertEqual(payload["instructions"], "system")

    def test_generate_reads_responses_sse_and_logs_progress(self):
        node = reach_module.ReachOpenAICompatibleLLMNode()
        response = FakeSSEResponse(
            [
                "event: response.created",
                'data: {"id":"resp-123","status":"in_progress"}',
                "",
                "event: response.output_text.delta",
                'data: {"delta":"final "}',
                "",
                "event: response.output_text.delta",
                'data: {"delta":"prompt"}',
                "",
                "event: response.completed",
                'data: {"id":"resp-123","status":"completed"}',
                "",
            ]
        )

        with mock.patch.object(node, "open_http_session", return_value=FakeSession()), \
            mock.patch.object(node, "close_http_session"), \
            mock.patch.object(node, "request_with_interrupt", return_value=response) as request:
            result = node.generate(
                api_key="key",
                base_url="https://api.reachapi.ai/v1",
                api_format="responses",
                model="gpt-5.6-sol",
                system_prompt="",
                prompt="describe all images",
                thinking_mode="medium",
                image_mode="none",
            )

        self.assertEqual(result[0], "final prompt")
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[1:3], ("POST", "https://api.reachapi.ai/v1/responses"))
        self.assertTrue(request.call_args.kwargs["stream"])
        self.assertEqual(request.call_args.kwargs["headers"]["Accept"], "text/event-stream")
        self.assertIn('"response.output_text.delta"', result[1])
        self.assertTrue(response.closed)

    def test_generate_accepts_non_streaming_json_fallback(self):
        node = reach_module.ReachOpenAICompatibleLLMNode()
        response = FakeResponse({"id": "resp-json", "status": "completed", "output_text": "json result"})

        with mock.patch.object(node, "open_http_session", return_value=FakeSession()), \
            mock.patch.object(node, "close_http_session"), \
            mock.patch.object(node, "request_with_interrupt", return_value=response) as request:
            result = node.generate(
                api_key="key",
                base_url="https://api.reachapi.ai/v1",
                api_format="responses",
                model="gpt-5.6-luna",
                system_prompt="",
                prompt="describe all images",
                thinking_mode="xhigh",
                image_mode="none",
            )

        self.assertEqual(result[0], "json result")
        self.assertTrue(request.call_args.kwargs["stream"])
        self.assertTrue(response.closed)

    def test_read_sse_reports_delayed_stream_without_blocking_main_thread(self):
        node = reach_module.ReachOpenAICompatibleLLMNode()
        response = DelayedResponse(0.05)
        result = node.read_sse_response(response, total_timeout=1)
        self.assertEqual(result["output_text"], "ok")

    def test_generate_submits_task_and_polls_until_complete_as_legacy_fallback(self):
        node = reach_module.ReachOpenAICompatibleLLMNode()
        node.first_poll_delay = 0
        submit = FakeResponse({"code": 200, "task_id": "task-123"})
        queued = FakeResponse({"code": 200, "status": "queued"})
        completed = FakeResponse({"code": 200, "status": "success", "output_text": "final prompt"})
        responses = [submit, queued, completed]

        with mock.patch.object(node, "open_http_session", return_value=FakeSession()), \
            mock.patch.object(node, "close_http_session"), \
            mock.patch.object(node, "request_with_interrupt", side_effect=responses) as request, \
            mock.patch.object(node, "interruptible_sleep"):
            result = node.generate(
                api_key="key",
                base_url="https://api.reachapi.ai/v1",
                api_format="responses",
                model="gpt-5.6-sol",
                system_prompt="",
                prompt="describe all images",
                thinking_mode="medium",
                image_mode="none",
                poll_interval=1,
                max_polls=3,
            )

        self.assertEqual(result[0], "final prompt")
        self.assertEqual(request.call_count, 3)
        self.assertEqual(request.call_args_list[0].args[1:3], ("POST", "https://api.reachapi.ai/v1/responses"))
        self.assertEqual(request.call_args_list[2].args[1:3], ("GET", "https://api.reachapi.ai/v1/tasks/task-123"))

    def test_extract_text_supports_nested_response(self):
        node = reach_module.ReachOpenAICompatibleLLMNode()
        self.assertEqual(node.extract_text({"data": {"output_text": "nested"}}), "nested")


if __name__ == "__main__":
    unittest.main()
