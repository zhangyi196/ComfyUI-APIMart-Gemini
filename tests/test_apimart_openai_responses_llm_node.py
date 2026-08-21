import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "apimart_openai_responses_llm_node.py"
spec = importlib.util.spec_from_file_location("apimart_openai_responses_llm_node", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class FakeResponse:
    def __init__(self, response_data=None, status_code=200, lines=None):
        self.response_data = response_data
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = str(response_data)
        self.headers = {"Content-Type": "application/json"}
        self.lines = lines or []
        self.closed = False

    def raise_for_status(self):
        if not self.ok:
            raise module.requests.exceptions.HTTPError(self.text)

    def json(self):
        return self.response_data

    def iter_lines(self, decode_unicode=False):
        yield from self.lines

    def close(self):
        self.closed = True


class APIMartOpenAIResponsesLLMNodeTests(unittest.TestCase):
    def test_build_urls_and_payload_match_apimart_docs(self):
        node = module.APIMartOpenAIResponsesLLMNode()
        self.assertEqual(node.build_responses_url("https://api.apimart.ai/v1"), "https://api.apimart.ai/v1/responses")
        payload = node.build_payload(
            api_format="responses",
            model="gpt-5.6-luna",
            system_prompt="You are concise",
            prompt="Compare these images",
            thinking_mode="high",
            image_parts=[{"type": "input_image", "image_url": "https://upload.apimart.ai/1.png"}],
            temperature=0.2,
            top_p=0.9,
            max_tokens=1000,
            stream=False,
        )
        self.assertEqual(payload["model"], "gpt-5.6-luna")
        self.assertEqual(payload["input"][0]["role"], "system")
        self.assertEqual(payload["input"][1]["content"][1]["type"], "input_image")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["reasoning"], {"effort": "high"})

    def test_build_comp_payload_uses_chat_completions_shape_and_reasoning_effort(self):
        node = module.APIMartOpenAIResponsesLLMNode()
        payload = node.build_payload(
            api_format="comp",
            model="gpt-5.6-sol",
            system_prompt="Be concise",
            prompt="Describe this image",
            thinking_mode="xhigh",
            image_parts=[{"type": "input_image", "image_url": "https://upload.apimart.ai/1.png"}],
            temperature=1.0,
            top_p=1.0,
            max_tokens=100,
            stream=False,
        )
        self.assertEqual(payload["model"], "gpt-5.6-sol")
        self.assertEqual(payload["reasoning_effort"], "xhigh")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "Be concise"})
        self.assertEqual(payload["messages"][1]["content"][1], {
            "type": "image_url",
            "image_url": {"url": "https://upload.apimart.ai/1.png"},
        })

    def test_model_and_thinking_choices_match_request(self):
        self.assertEqual(module.APIMartOpenAIResponsesLLMNode.MODEL_CHOICES, ["gpt-5.6-luna", "gpt-5.6-sol"])
        self.assertEqual(module.APIMartOpenAIResponsesLLMNode.THINKING_CHOICES, ["medium", "high", "xhigh"])

    def test_collect_images_splits_batch(self):
        node = module.APIMartOpenAIResponsesLLMNode()
        images = node.collect_images(image_1=np.zeros((3, 2, 2, 3), dtype=np.float32))
        self.assertEqual(len(images), 3)

    def test_upload_uses_apimart_top_level_url_and_20mb_limit(self):
        node = module.APIMartOpenAIResponsesLLMNode()
        response = FakeResponse({"url": "https://upload.apimart.ai/f/image/test.png"})
        with mock.patch.object(module.requests, "post", return_value=response) as post:
            image_url = node.upload_image(np.zeros((2, 2, 3), dtype=np.float32), "key", node.DEFAULT_UPLOAD_URL, 1, 30)
        self.assertEqual(image_url, "https://upload.apimart.ai/f/image/test.png")
        self.assertEqual(post.call_args.args[0], "https://api.apimart.ai/v1/uploads/images")
        self.assertTrue(response.closed)

    def test_generate_parses_non_streaming_responses_json(self):
        node = module.APIMartOpenAIResponsesLLMNode()
        response = FakeResponse({"id": "resp-1", "choices": [{"message": {"content": "hello"}}]})
        with mock.patch.object(module.requests, "post", return_value=response) as post:
            result = node.generate(
                api_key="key",
                base_url="https://api.apimart.ai/v1",
                model="gpt-5.6-luna",
                system_prompt="",
                prompt="Hi",
                image_mode="none",
                api_format="responses",
                thinking_mode="medium",
            )
        self.assertEqual(result[0], "hello")
        self.assertEqual(post.call_args.args[0], "https://api.apimart.ai/v1/responses")
        self.assertFalse(post.call_args.kwargs["json"]["stream"])

    def test_extract_text_supports_apimart_nested_data_response(self):
        node = module.APIMartOpenAIResponsesLLMNode()
        self.assertEqual(
            node.extract_text({"code": 200, "data": {"choices": [{"message": {"content": "nested"}}]}}),
            "nested",
        )

    def test_generate_parses_sse_response(self):
        node = module.APIMartOpenAIResponsesLLMNode()
        response = FakeResponse(
            lines=[
                "event: response.created",
                'data: {"type":"response.created"}',
                "",
                'data: {"type":"response.output_text.delta","delta":"hello"}',
                "",
                'data: {"type":"response.completed"}',
                "",
            ]
        )
        response.headers = {"Content-Type": "text/event-stream"}
        with mock.patch.object(module.requests, "post", return_value=response) as post:
            result = node.generate(
                api_key="key",
                base_url="https://api.apimart.ai/v1",
                model="gpt-5.6-sol",
                system_prompt="",
                prompt="Hi",
                image_mode="none",
                api_format="responses",
                thinking_mode="high",
                stream=True,
            )
        self.assertEqual(result[0], "hello")
        self.assertTrue(post.call_args.kwargs["stream"])
        self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()
