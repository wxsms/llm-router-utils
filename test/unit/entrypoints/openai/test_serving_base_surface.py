"""Guard: the base-class surface a faithful ``serving_chat.py`` calls must exist.

``serving_chat.py`` is byte-identical to upstream, so it calls the upstream
``OpenAIServingBase`` API. An earlier version of this repo shipped a 28-line
``serving_base.py`` carrying only ``__init__``, which made those calls raise
``AttributeError`` at runtime while every existing test still passed — no test
touched the path.

This test walks ``serving_chat.py`` for ``self.<name>`` reads that the class
does not itself define, and asserts each resolves on the MRO. It is generic: a
future upstream version that starts calling a new base method fails here instead
of in production.

Extraction-inherent attributes are excluded explicitly, with a reason each.
"""

import ast
import pathlib
import unittest

from llm_router_utils.sglang.srt.entrypoints.openai.serving_base import (
    OpenAIServingBase,
)
from llm_router_utils.sglang.srt.entrypoints.openai.serving_chat import (
    OpenAIServingChat,
)

_SERVING_CHAT = (
    pathlib.Path(__file__).resolve().parents[4]
    / "src"
    / "llm_router_utils"
    / "sglang"
    / "srt"
    / "entrypoints"
    / "openai"
    / "serving_chat.py"
)

#: ``self.<name>`` reads that neither the class nor its base defines.
#: Each is set by ``OPEN_``-style assignment elsewhere in the class (so this AST
#: scan cannot see it) or is a documented extraction gap.
_NOT_AN_ATTRIBUTE = {
    # Assigned inside __init__ via tuple unpacking / dict literals.
    "_chat_template_cache",
    "_encoding_adapter",
    "_reasoning_detector",
    "_response_parser",
    "_prompt_text_round_trip_is_lossy",
    "_template_reads_reasoning_content",
    "_tokenizer_auto_adds_specials",
    "_KIMI_K3_GENERATION_STUB_TOKENS",
    "chat_encoding_spec",
    "default_chat_template_kwargs",
    "default_sampling_params",
    "is_gemma4",
    "is_gpt_oss",
    "is_minimax_m3",
    "reasoning_parser",
    "supports_inline_system",
    "template_manager",
    "tokenizer_manager",
    "tool_call_parser",
}

#: Methods the base class must provide because serving_chat calls them.
#: Named explicitly so a regression is reported clearly, not just as a count.
_REQUIRED_BASE_METHODS = (
    "_parse_model_parameter",
    "_resolve_lora_path",
    "create_error_response",
    "create_streaming_error_response",
    "extract_custom_labels",
    "extract_routed_dp_rank_from_header",
    "extract_routing_key",
)


def _self_reads(path: pathlib.Path) -> set[str]:
    """Every ``self.<name>`` attribute read in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }


def _locally_defined(path: pathlib.Path) -> set[str]:
    """Names the class defines itself (methods) plus assignments to self."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    # `self.x = ...` assignments anywhere in the class count as defined.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and isinstance(node.ctx, ast.Store)
        ):
            defined.add(node.attr)
    return defined


class TestServingBaseSurface(unittest.TestCase):
    def test_required_base_methods_exist(self):
        missing = [m for m in _REQUIRED_BASE_METHODS if not hasattr(OpenAIServingBase, m)]
        self.assertEqual(
            missing,
            [],
            "OpenAIServingBase is missing methods a byte-identical "
            f"serving_chat.py calls: {missing}. Restore them rather than "
            "letting serving_chat.py (which must stay upstream-identical) "
            "call a truncated base class.",
        )

    def test_every_self_read_resolves_on_the_instance(self):
        reads = _self_reads(_SERVING_CHAT) - _locally_defined(_SERVING_CHAT)
        reads -= _NOT_AN_ATTRIBUTE
        self.assertGreater(
            len(reads), 0, "AST scan found no inherited self.* reads to check"
        )

        missing = sorted(
            name
            for name in reads
            if not hasattr(OpenAIServingBase, name)
        )
        self.assertEqual(
            missing,
            [],
            "serving_chat.py reads these off the base class but it does not "
            f"provide them: {missing}",
        )

    def test_base_assigns_tokenizer_manager(self):
        """The one attribute every downstream helper depends on.

        It is an *instance* attribute, so check the base ``__init__`` source
        rather than ``hasattr`` on the class.
        """
        self.assertIn("tokenizer_manager", _self_reads(_SERVING_CHAT))
        import inspect

        source = inspect.getsource(OpenAIServingBase.__init__)
        self.assertIn("self.tokenizer_manager = tokenizer_manager", source)


if __name__ == "__main__":
    unittest.main()
