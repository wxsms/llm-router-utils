"""Tests for ServerArgs.device default and PortArgs.init_new (single-node)."""
import unittest

from llm_router_utils.sglang.srt.server_args import PortArgs, ServerArgs


class TestServerArgsDevice(unittest.TestCase):
    def test_device_defaults_to_cuda(self):
        sa = ServerArgs(model_path="Qwen/Qwen3-32B")
        self.assertEqual(sa.device, "cuda")

    def test_device_override(self):
        sa = ServerArgs(model_path="X", device="cpu")
        self.assertEqual(sa.device, "cpu")

    def test_tokenizer_path_defaults_to_model_path(self):
        sa = ServerArgs(model_path="Qwen/Qwen3-32B")
        self.assertEqual(sa.tokenizer_path, "Qwen/Qwen3-32B")


class TestPortArgsInitNew(unittest.TestCase):
    def test_init_new_basic(self):
        sa = ServerArgs(model_path="Qwen/Qwen3-32B")
        pa = PortArgs.init_new(sa)
        self.assertGreater(pa.nccl_port, 0)
        self.assertTrue(pa.tokenizer_ipc_name.startswith("ipc://"))
        self.assertTrue(pa.scheduler_input_ipc_name.startswith("ipc://"))
        self.assertTrue(pa.detokenizer_ipc_name.startswith("ipc://"))
        self.assertTrue(pa.rpc_ipc_name.startswith("ipc://"))
        self.assertTrue(pa.metrics_ipc_name.startswith("ipc://"))
        self.assertIsNone(pa.tokenizer_worker_ipc_name)
        self.assertEqual(pa.decoupled_spec_ipc_config, None)
        self.assertTrue(pa.instance_id)
        self.assertEqual(len(pa.instance_id), 12)

    def test_init_new_rejects_dp_attention(self):
        sa = ServerArgs(model_path="X")
        sa.enable_dp_attention = True  # type: ignore[attr-defined]
        with self.assertRaises(NotImplementedError):
            PortArgs.init_new(sa)

    def test_init_new_rejects_decoupled_spec(self):
        sa = ServerArgs(model_path="X")
        sa.decoupled_spec_role = "drafter"  # type: ignore[attr-defined]
        with self.assertRaises(NotImplementedError):
            PortArgs.init_new(sa)


if __name__ == "__main__":
    unittest.main()
