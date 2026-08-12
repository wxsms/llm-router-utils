"""Unit tests for the restored hf_transformers modules.

Verifies that the upstream-copied tokenizer/config loading helpers are
importable, wired into TokenizerManager/ModelConfig, and behave correctly
for the pure-text path.
"""
import unittest
from unittest.mock import MagicMock, patch


class TestHfTransformersImports(unittest.TestCase):
    """Verify the restored modules import and expose upstream symbols."""

    def test_get_tokenizer_importable(self):
        from llm_router_utils.sglang.srt.utils.hf_transformers.tokenizer import (
            get_tokenizer,
        )

        self.assertTrue(callable(get_tokenizer))

    def test_get_config_importable(self):
        from llm_router_utils.sglang.srt.utils.hf_transformers.config import (
            get_config,
        )

        self.assertTrue(callable(get_config))

    def test_package_re_exports(self):
        from llm_router_utils.sglang.srt.utils import hf_transformers

        # A representative subset of upstream exports.
        for name in (
            "get_tokenizer",
            "get_config",
            "get_hf_text_config",
            "get_context_length",
            "attach_additional_stop_token_ids",
            "check_gguf_file",
            "normalize_rope_scaling_compat",
        ):
            self.assertTrue(hasattr(hf_transformers, name), name)

    def test_mistral_helpers_present(self):
        from llm_router_utils.sglang.srt.utils.hf_transformers import mistral_utils

        self.assertTrue(hasattr(mistral_utils, "is_mistral_model"))
        self.assertTrue(hasattr(mistral_utils, "load_mistral_config"))
        self.assertTrue(hasattr(mistral_utils, "is_bare_tekken_checkpoint"))
        self.assertTrue(hasattr(mistral_utils, "patch_mistral_common_tokenizer"))

    def test_patches_module_slimmed(self):
        from llm_router_utils.sglang.srt.utils import hf_transformers_patches

        # Kept on-demand helpers.
        self.assertTrue(hasattr(hf_transformers_patches, "normalize_rope_scaling_compat"))
        self.assertTrue(hasattr(hf_transformers_patches, "_ensure_gguf_version"))
        self.assertTrue(hasattr(hf_transformers_patches, "apply_all"))
        # Removed torch-dependent patch functions.
        self.assertFalse(hasattr(hf_transformers_patches, "_patch_image_process_cuda_tensor"))
        self.assertFalse(hasattr(hf_transformers_patches, "_patch_flash_attn_availability"))

    def test_apply_all_is_idempotent(self):
        from llm_router_utils.sglang.srt.utils import hf_transformers_patches

        hf_transformers_patches.apply_all()
        hf_transformers_patches.apply_all()  # second call is a no-op


class TestNoTopLevelTorchImport(unittest.TestCase):
    """Ensure restored files have no module-level ``import torch``."""

    RESTORED_FILES = [
        "src/llm_router_utils/sglang/srt/utils/hf_transformers/common.py",
        "src/llm_router_utils/sglang/srt/utils/hf_transformers/config.py",
        "src/llm_router_utils/sglang/srt/utils/hf_transformers/tokenizer.py",
        "src/llm_router_utils/sglang/srt/utils/hf_transformers/mistral_utils.py",
        "src/llm_router_utils/sglang/srt/utils/hf_transformers/__init__.py",
        "src/llm_router_utils/sglang/srt/utils/hf_transformers_patches.py",
        "src/llm_router_utils/sglang/srt/utils/patch_tokenizer.py",
        "src/llm_router_utils/sglang/srt/configs/model_config_parser_registry.py",
    ]

    def test_no_top_level_torch(self):
        import ast
        import os

        # test/unit/hf_transformers/test_*.py -> repo root is 3 dirs up.
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        for rel in self.RESTORED_FILES:
            path = os.path.join(root, rel)
            with open(path, encoding="utf-8-sig") as f:
                tree = ast.parse(f.read(), filename=path)
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(
                            alias.name == "torch" or alias.name.startswith("torch."),
                            f"top-level import torch in {rel} line {node.lineno}",
                        )
                elif isinstance(node, ast.ImportFrom):
                    self.assertFalse(
                        node.module
                        and (node.module == "torch" or node.module.startswith("torch.")),
                        f"top-level from torch import in {rel} line {node.lineno}",
                    )


class TestStubs(unittest.TestCase):
    """Verify the connector/runai stubs raise/return as expected."""

    def test_create_remote_connector_raises(self):
        from llm_router_utils.sglang.srt.connector import create_remote_connector

        with self.assertRaises(NotImplementedError):
            create_remote_connector("foo://bar")

    def test_is_runai_obj_uri_returns_false(self):
        from llm_router_utils.sglang.srt.utils.runai_utils import is_runai_obj_uri

        self.assertFalse(is_runai_obj_uri("anything"))

    def test_object_storage_model_raises(self):
        from llm_router_utils.sglang.srt.utils.runai_utils import ObjectStorageModel

        with self.assertRaises(NotImplementedError):
            ObjectStorageModel()


class TestTokenizerManagerWiring(unittest.TestCase):
    """Verify TokenizerManager.init_tokenizer_and_processor calls get_tokenizer."""

    @patch("llm_router_utils.sglang.srt.utils.hf_transformers.tokenizer.get_tokenizer")
    def test_init_tokenizer_calls_get_tokenizer(self, mock_get_tok):
        from llm_router_utils.sglang.srt.managers.tokenizer_manager import TokenizerManager
        from llm_router_utils.sglang.srt.server_args import PortArgs, ServerArgs

        mock_tok = MagicMock()
        mock_get_tok.return_value = mock_tok

        tm = TokenizerManager.__new__(TokenizerManager)
        tm.server_args = ServerArgs(model_path="fake-model", tokenizer_mode="auto")
        tm.port_args = PortArgs()
        tm.tokenizer = None
        tm.processor = None
        tm.model_config = None

        tm.init_tokenizer_and_processor()

        mock_get_tok.assert_called_once()
        _, kwargs = mock_get_tok.call_args
        self.assertEqual(kwargs["tokenizer_mode"], "auto")
        self.assertFalse(kwargs["trust_remote_code"])
        self.assertIsNone(kwargs["tokenizer_revision"])
        self.assertEqual(kwargs["tokenizer_backend"], "huggingface")
        self.assertIs(tm.tokenizer, mock_tok)


class TestModelConfigWiring(unittest.TestCase):
    """Verify ModelConfig uses get_config."""

    @patch("llm_router_utils.sglang.srt.utils.hf_transformers.config.get_config")
    def test_model_config_calls_get_config(self, mock_get_config):
        from llm_router_utils.sglang.srt.configs.model_config import ModelConfig

        mock_hf_config = MagicMock()
        mock_hf_config.max_position_embeddings = 4096
        mock_get_config.return_value = mock_hf_config

        cfg = ModelConfig(
            model_path="fake-model",
            trust_remote_code=False,
            revision="main",
        )

        mock_get_config.assert_called_once_with(
            "fake-model",
            trust_remote_code=False,
            revision="main",
        )
        self.assertIs(cfg.hf_config, mock_hf_config)


class TestServerArgsFields(unittest.TestCase):
    """Verify the new ServerArgs fields exist with correct defaults."""

    def test_defaults(self):
        from llm_router_utils.sglang.srt.server_args import ServerArgs

        sa = ServerArgs(model_path="fake-model")
        self.assertIsNone(sa.revision)
        self.assertEqual(sa.tokenizer_backend, "huggingface")

    def test_settable(self):
        from llm_router_utils.sglang.srt.server_args import ServerArgs

        sa = ServerArgs(
            model_path="fake-model",
            revision="v1.0",
            tokenizer_backend="fastokens",
        )
        self.assertEqual(sa.revision, "v1.0")
        self.assertEqual(sa.tokenizer_backend, "fastokens")


class TestCommonHelpers(unittest.TestCase):
    """Verify the upstream helpers copied into common.py behave correctly."""

    def test_is_remote_url(self):
        from llm_router_utils.sglang.srt.utils.common import is_remote_url

        self.assertTrue(is_remote_url("foo://bar"))
        self.assertFalse(is_remote_url("/local/path"))
        self.assertFalse(is_remote_url("relative/path"))

    def test_get_bool_env_var(self):
        from llm_router_utils.sglang.srt.utils.common import get_bool_env_var

        self.assertTrue(get_bool_env_var("NONEXISTENT_XYZ", "true"))
        self.assertFalse(get_bool_env_var("NONEXISTENT_XYZ", "false"))
        self.assertFalse(get_bool_env_var("NONEXISTENT_XYZ", "garbage"))

    def test_flatten_nested_list(self):
        from llm_router_utils.sglang.srt.utils.common import flatten_nested_list

        self.assertEqual(flatten_nested_list([1, [2, [3, 4]], 5]), [1, 2, 3, 4, 5])
        self.assertEqual(flatten_nested_list(42), [42])

    def test_is_hip_is_npu_torch_free(self):
        from llm_router_utils.sglang.srt.utils.common import is_hip, is_npu

        # Should return bool without raising even if torch is absent.
        self.assertIsInstance(is_hip(), bool)
        self.assertIsInstance(is_npu(), bool)

    def test_lru_cache_frozenset_caches(self):
        from llm_router_utils.sglang.srt.utils.common import lru_cache_frozenset

        calls = []

        @lru_cache_frozenset(maxsize=4)
        def fn(a, b):
            calls.append((a, b))
            return a + b

        # Pass two lists (unhashable) to exercise the frozenset conversion.
        self.assertEqual(fn([1, 2], [3, 4]), [1, 2, 3, 4])
        self.assertEqual(fn([1, 2], [3, 4]), [1, 2, 3, 4])  # cached
        self.assertEqual(len(calls), 1)
        fn.cache_clear()
        self.assertEqual(fn([1, 2], [3, 4]), [1, 2, 3, 4])
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
