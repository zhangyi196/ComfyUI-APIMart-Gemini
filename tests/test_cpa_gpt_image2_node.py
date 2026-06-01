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


if __name__ == "__main__":
    unittest.main()
