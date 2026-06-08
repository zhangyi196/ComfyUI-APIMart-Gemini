import importlib.util
from pathlib import Path
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "cpa_gpt_image2_node.py"
spec = importlib.util.spec_from_file_location("cpa_gpt_image2_node", MODULE_PATH)
cpa_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cpa_module)

REACH_ASYNC_MODULE_PATH = Path(__file__).resolve().parents[1] / "reach_gpt_image2_node.py"
reach_async_spec = importlib.util.spec_from_file_location("reach_gpt_image2_node", REACH_ASYNC_MODULE_PATH)
reach_async_module = importlib.util.module_from_spec(reach_async_spec)
assert reach_async_spec.loader is not None
reach_async_spec.loader.exec_module(reach_async_module)

REACH_SYNC_MODULE_PATH = Path(__file__).resolve().parents[1] / "reach_gpt_image2_sync_node.py"
reach_sync_spec = importlib.util.spec_from_file_location("reach_gpt_image2_sync_node", REACH_SYNC_MODULE_PATH)
reach_sync_module = importlib.util.module_from_spec(reach_sync_spec)
assert reach_sync_spec.loader is not None
reach_sync_spec.loader.exec_module(reach_sync_module)

REACH_NANO_MODULE_PATH = Path(__file__).resolve().parents[1] / "reach_nano_node.py"
reach_nano_spec = importlib.util.spec_from_file_location("reach_nano_node", REACH_NANO_MODULE_PATH)
reach_nano_module = importlib.util.module_from_spec(reach_nano_spec)
assert reach_nano_spec.loader is not None
reach_nano_spec.loader.exec_module(reach_nano_module)


class InterruptProcessingException(Exception):
    pass


class FakeComfyModelManagement:
    def throw_exception_if_processing_interrupted(self):
        raise InterruptProcessingException("interrupted")


class FakeSession:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class CPAGPTImage2InterruptTests(unittest.TestCase):
    def test_check_interrupted_closes_active_session_and_reraises(self):
        node = cpa_module.CPAGPTImage2GenerationNode()
        session = FakeSession()
        node._active_session = session

        with mock.patch.object(cpa_module, "comfy_model_management", FakeComfyModelManagement()):
            with self.assertRaises(InterruptProcessingException):
                node.check_interrupted()

        self.assertTrue(session.closed)
        self.assertIsNone(node._active_session)

    def test_generate_does_not_wrap_comfy_interrupt(self):
        node = cpa_module.CPAGPTImage2GenerationNode()

        with mock.patch.object(node, "open_http_session", return_value=FakeSession()), \
            mock.patch.object(node, "close_http_session"), \
            mock.patch.object(node, "check_interrupted", side_effect=InterruptProcessingException):
            with self.assertRaises(InterruptProcessingException):
                node.generate(
                    mode="text-to-image",
                    api_key="key",
                    base_url="http://localhost:8317/",
                    prompt="prompt",
                    model="gpt-image-2",
                    resolution="1k",
                    aspect_ratio="1:1",
                    quality="medium",
                    background="auto",
                    moderation="auto",
                    output_format="png",
                    seed=0,
                )


class ReachGPTImage2InterruptTests(unittest.TestCase):
    def test_async_check_interrupted_closes_active_session_and_reraises(self):
        node = reach_async_module.ReachGPTImage2GenerationNode()
        session = FakeSession()
        node._active_session = session

        with mock.patch.object(reach_async_module, "comfy_model_management", FakeComfyModelManagement()):
            with self.assertRaises(InterruptProcessingException):
                node.check_interrupted()

        self.assertTrue(session.closed)
        self.assertIsNone(node._active_session)

    def test_async_generate_does_not_wrap_comfy_interrupt(self):
        node = reach_async_module.ReachGPTImage2GenerationNode()

        with mock.patch.object(node, "open_http_session", return_value=FakeSession()), \
            mock.patch.object(node, "close_http_session"), \
            mock.patch.object(node, "check_interrupted", side_effect=InterruptProcessingException):
            with self.assertRaises(InterruptProcessingException):
                node.generate(
                    mode="text-to-image",
                    api_key="key",
                    prompt="prompt",
                    model="gpt-image-2-async",
                    resolution="1k",
                    aspect_ratio="1:1",
                    quality="medium",
                    background="auto",
                    moderation="auto",
                    output_format="png",
                    seed=0,
                )

    def test_sync_check_interrupted_closes_active_sessions_and_reraises(self):
        node = reach_sync_module.ReachGPTImage2SyncGenerationNode()
        session = FakeSession()
        node._active_sessions = [session]

        with mock.patch.object(reach_sync_module, "comfy_model_management", FakeComfyModelManagement()):
            with self.assertRaises(InterruptProcessingException):
                node.check_interrupted()

        self.assertTrue(session.closed)
        self.assertEqual(node._active_sessions, [])

    def test_sync_generate_does_not_wrap_comfy_interrupt(self):
        node = reach_sync_module.ReachGPTImage2SyncGenerationNode()

        with mock.patch.object(node, "open_http_session", return_value=FakeSession()), \
            mock.patch.object(node, "close_http_sessions"), \
            mock.patch.object(node, "check_interrupted", side_effect=InterruptProcessingException):
            with self.assertRaises(InterruptProcessingException):
                node.generate(
                    mode="text-to-image",
                    api_key="key",
                    prompt="prompt",
                    model="gpt-image-2",
                    resolution="1k",
                    aspect_ratio="1:1",
                    quality="medium",
                    background="auto",
                    moderation="auto",
                    output_format="png",
                    seed=0,
                )


class ReachNanoBananaNodeTests(unittest.TestCase):
    def test_mapping_uses_reach_nanobanana_display_name(self):
        self.assertIn("ReachNanoBananaGenerationNode", reach_nano_module.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            reach_nano_module.NODE_DISPLAY_NAME_MAPPINGS["ReachNanoBananaGenerationNode"],
            "reach nanobanana",
        )

    def test_build_input_payload_uses_documented_nanobanana_shape(self):
        node = reach_nano_module.ReachNanoBananaGenerationNode()

        payload = node.build_input_payload(
            prompt="premium product shot",
            aspect_ratio="4:5",
            resolution="2k",
            output_format="png",
            enable_web_search="true",
            seed=12345,
            image_urls=["https://example.com/reference.png"],
        )

        self.assertEqual(
            payload,
            {
                "prompt": "premium product shot",
                "aspect_ratio": "4:5",
                "resolution": "2k",
                "output_format": "png",
                "enable_web_search": True,
                "seed": 12345,
                "image_urls": ["https://example.com/reference.png"],
            },
        )

    def test_validate_inputs_requires_reference_for_image_to_image(self):
        node = reach_nano_module.ReachNanoBananaGenerationNode()

        with self.assertRaises(ValueError):
            node.validate_inputs(
                mode="image-to-image",
                prompt="edit this",
                reference_images=[],
                callback_url="",
            )

    def test_input_types_excludes_reference_image_urls_and_includes_seed(self):
        node = reach_nano_module.ReachNanoBananaGenerationNode()
        input_types = node.INPUT_TYPES()

        self.assertIn("seed", input_types["required"])
        self.assertNotIn("reference_image_urls", input_types["optional"])

    def test_check_interrupted_closes_active_session_and_reraises(self):
        node = reach_nano_module.ReachNanoBananaGenerationNode()
        session = FakeSession()
        node._active_session = session

        with mock.patch.object(reach_nano_module, "comfy_model_management", FakeComfyModelManagement()):
            with self.assertRaises(InterruptProcessingException):
                node.check_interrupted()

        self.assertTrue(session.closed)
        self.assertIsNone(node._active_session)

    def test_generate_does_not_wrap_comfy_interrupt(self):
        node = reach_nano_module.ReachNanoBananaGenerationNode()

        with mock.patch.object(node, "open_http_session", return_value=FakeSession()), \
            mock.patch.object(node, "close_http_session"), \
            mock.patch.object(node, "check_interrupted", side_effect=InterruptProcessingException):
            with self.assertRaises(InterruptProcessingException):
                node.generate(
                    mode="text-to-image",
                    api_key="key",
                    prompt="prompt",
                    model="nanobanana-pro",
                    aspect_ratio="1:1",
                    resolution="2k",
                    output_format="png",
                    enable_web_search="false",
                    seed=0,
                )


if __name__ == "__main__":
    unittest.main()
