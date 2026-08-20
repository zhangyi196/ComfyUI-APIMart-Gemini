import importlib.util
from pathlib import Path
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "openai_compatible_llm_node.py"
spec = importlib.util.spec_from_file_location("openai_compatible_llm_node", MODULE_PATH)
llm_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(llm_module)


class FakeResponse:
    def __init__(self, response_data, status_code=200):
        self.response_data = response_data
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = str(response_data)
        self.closed = False

    def json(self):
        return self.response_data

    def raise_for_status(self):
        if not self.ok:
            raise llm_module.requests.exceptions.HTTPError(self.text)

    def close(self):
        self.closed = True


class OpenAICompatibleLLMNodeTests(unittest.TestCase):
    def test_build_payload_supports_reasoning_and_base64_image(self):
        node = llm_module.OpenAICompatibleLLMNode()
        payload = node.build_payload(
            model="gpt-5.6-sol",
            system_prompt="Be concise",
            prompt="Describe this image",
            thinking_mode="high",
            image_parts=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}],
            temperature=0.5,
            max_tokens=1000,
        )

        self.assertEqual(payload["model"], "gpt-5.6-sol")
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["messages"][1]["content"][1]["type"], "image_url")

    def test_build_payload_maps_medium_reasoning_mode(self):
        node = llm_module.OpenAICompatibleLLMNode()
        payload = node.build_payload("gpt-5.6-luna", "", "Hello", "medium", [], 1.0, 100)
        self.assertEqual(payload["reasoning_effort"], "medium")
        self.assertEqual(payload["messages"], [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}])

    def test_build_responses_payload_uses_input_image_and_reasoning_object(self):
        node = llm_module.OpenAICompatibleLLMNode()
        payload = node.build_responses_payload(
            model="gpt-5.6-sol",
            system_prompt="Be concise",
            prompt="Describe these images",
            thinking_mode="high",
            image_parts=[
                {"type": "image_url", "image_url": {"url": "https://cdn.example.com/1.png"}},
                {"type": "image_url", "image_url": {"url": "https://cdn.example.com/2.png"}},
            ],
            temperature=0.5,
            max_tokens=1000,
        )

        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertEqual(payload["max_output_tokens"], 1000)
        self.assertEqual(payload["instructions"], "Be concise")
        self.assertEqual(
            [part["type"] for part in payload["input"][0]["content"]],
            ["input_text", "input_image", "input_image"],
        )
        self.assertEqual(payload["input"][0]["content"][1]["image_url"], "https://cdn.example.com/1.png")

    def test_image_mode_none_does_not_encode_or_upload_images(self):
        node = llm_module.OpenAICompatibleLLMNode()
        with mock.patch.object(node, "encode_image_data_url") as encode, mock.patch.object(node, "upload_image") as upload:
            parts = node.build_image_parts("none", [object()], "key", node.FILE_UPLOAD_URL, 30)
        self.assertEqual(parts, [])
        encode.assert_not_called()
        upload.assert_not_called()

    def test_image_mode_requires_an_input_image(self):
        node = llm_module.OpenAICompatibleLLMNode()
        with self.assertRaisesRegex(ValueError, "至少需要 1 张输入图片"):
            node.build_image_parts("base64", [], "key", "", 30)

    def test_generate_posts_openai_compatible_payload_and_returns_text(self):
        node = llm_module.OpenAICompatibleLLMNode()
        response = FakeResponse({"id": "chatcmpl-test", "choices": [{"message": {"content": "Hello"}}]})
        with mock.patch.object(llm_module.requests, "post", return_value=response) as post:
            result = node.generate(
                api_key="key",
                base_url="https://example.com/v1",
                model="gpt-5.6-luna",
                system_prompt="",
                prompt="Hi",
                thinking_mode="xhigh",
                image_mode="none",
            )

        self.assertEqual(result[0], "Hello")
        self.assertIn('"chatcmpl-test"', result[1])
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "xhigh")
        self.assertEqual(post.call_args.args[0], "https://example.com/v1/chat/completions")
        self.assertTrue(response.closed)

    def test_generate_uses_responses_endpoint_and_extracts_output_text(self):
        node = llm_module.OpenAICompatibleLLMNode()
        response = FakeResponse({"id": "resp-test", "output_text": "Responses hello"})
        with mock.patch.object(llm_module.requests, "post", return_value=response) as post:
            result = node.generate(
                api_key="key",
                base_url="https://example.com/v1",
                model="gpt-5.6-sol",
                system_prompt="",
                prompt="Hi",
                thinking_mode="medium",
                image_mode="none",
                api_format="responses",
            )

        self.assertEqual(result[0], "Responses hello")
        self.assertEqual(post.call_args.args[0], "https://example.com/v1/responses")
        self.assertEqual(post.call_args.kwargs["json"]["reasoning"], {"effort": "medium"})
        self.assertIn("max_output_tokens", post.call_args.kwargs["json"])
        self.assertTrue(response.closed)

    def test_extract_text_supports_structured_content(self):
        node = llm_module.OpenAICompatibleLLMNode()
        text = node.extract_text(
            {"choices": [{"message": {"content": [{"type": "text", "text": "A"}, {"text": "B"}]}}]}
        )
        self.assertEqual(text, "AB")


if __name__ == "__main__":
    unittest.main()
