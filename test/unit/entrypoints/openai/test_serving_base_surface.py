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
from types import SimpleNamespace

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


class TestLoraHelpersBehaviour(unittest.TestCase):
    """Behavioural parity for the LoRA helpers, transcribed from upstream.

    A presence-only guard (`hasattr`) previously certified a broken
    implementation: `_resolve_lora_path` had its parameters reversed and used a
    different algorithm, yet every test still passed. These cases pin the
    *results* so that class of defect cannot recur.
    """

    SERVED = "my-model"

    def _serving(self, served=SERVED):
        base = OpenAIServingBase.__new__(OpenAIServingBase)
        base.tokenizer_manager = SimpleNamespace(served_model_name=served)
        return base

    def test_parse_model_parameter(self):
        s = self._serving()
        cases = [
            ("my-model", ("my-model", None)),
            ("other-model", ("other-model", None)),
            ("my-model:adapter", ("my-model", "adapter")),
            ("my-model:", ("my-model", None)),
            ("other:adapter", ("other", "adapter")),
            ("a:b:c", ("a", "b:c")),
            ("  sp : ad  ", ("sp", "ad")),
        ]
        for model, expected in cases:
            with self.subTest(model=model):
                self.assertEqual(s._parse_model_parameter(model), expected)

    def test_parse_preserves_a_served_name_containing_colons(self):
        """An object-storage URI must not be split at its scheme colon."""
        s = self._serving("s3://bucket/model")
        self.assertEqual(
            s._parse_model_parameter("s3://bucket/model"), ("s3://bucket/model", None)
        )
        self.assertEqual(
            s._parse_model_parameter("s3://bucket/model:adapter"),
            ("s3://bucket/model", "adapter"),
        )

    def test_resolve_lora_path_argument_order(self):
        """Signature is (request_model, explicit_lora_path) -- upstream's order.

        A reversed signature silently returned the served model name as the
        adapter for every plain request.
        """
        import inspect

        params = list(inspect.signature(OpenAIServingBase._resolve_lora_path).parameters)
        self.assertEqual(params, ["self", "request_model", "explicit_lora_path"])

    def test_resolve_lora_path_precedence(self):
        s = self._serving()
        # Plain request: explicit path passes through untouched.
        self.assertIsNone(s._resolve_lora_path("my-model", None))
        self.assertEqual(s._resolve_lora_path("my-model", "explicit"), "explicit")
        # Model-embedded adapter wins over the explicit path.
        self.assertEqual(
            s._resolve_lora_path("my-model:from-model", "explicit"), "from-model"
        )
        # A plain served name must NOT be mistaken for an adapter.
        self.assertEqual(s._resolve_lora_path("other-model", None), None)

    def test_resolve_lora_path_supports_batches(self):
        s = self._serving()
        batch = ["a", None]
        self.assertEqual(s._resolve_lora_path("my-model", batch), batch)
        self.assertEqual(s._resolve_lora_path("my-model:x", batch), "x")


class TestHeaderExtractorsBehaviour(unittest.TestCase):
    """Behavioural parity for the header extractors.

    Each was previously certified only by ``hasattr``; making one of them
    ``return None`` unconditionally left the whole suite green.
    """

    def _serving(self, **attrs):
        """Build an instance with the ``__init__``-set attributes populated."""
        base = OpenAIServingBase.__new__(OpenAIServingBase)
        base.allowed_custom_labels = None
        base._custom_labels_header = "x-custom-labels"
        for k, v in attrs.items():
            setattr(base, k, v)
        return base

    @staticmethod
    def _req(headers):
        return SimpleNamespace(headers=headers)

    def test_extract_routing_key(self):
        s = self._serving()
        self.assertEqual(
            s.extract_routing_key(self._req({"x-smg-routing-key": "k1"})), "k1"
        )
        self.assertIsNone(s.extract_routing_key(self._req({})))
        self.assertIsNone(s.extract_routing_key(None))

    def test_extract_routed_dp_rank_from_header(self):
        s = self._serving()
        # Header wins over the body value.
        self.assertEqual(
            s.extract_routed_dp_rank_from_header(
                self._req({"x-data-parallel-rank": "3"}), body_routed_dp_rank=1
            ),
            3,
        )
        # No header -> body value passes through.
        self.assertEqual(
            s.extract_routed_dp_rank_from_header(self._req({}), body_routed_dp_rank=7),
            7,
        )
        self.assertIsNone(s.extract_routed_dp_rank_from_header(None, None))

    def test_extract_routed_dp_rank_rejects_non_integer(self):
        from fastapi import HTTPException

        s = self._serving()
        with self.assertRaises(HTTPException):
            s.extract_routed_dp_rank_from_header(
                self._req({"x-data-parallel-rank": "not-an-int"})
            )

    def test_extract_custom_labels_is_off_by_default(self):
        """No allow-list configured -> never echo labels back."""
        s = self._serving()
        self.assertIsNone(
            s.extract_custom_labels(self._req({"x-custom-labels": '{"a": "b"}'}))
        )

    def test_extract_custom_labels_filters_to_the_allow_list(self):
        """Reachable via ServerArgs, not a hand-set attribute.

        The earlier version assigned ``allowed_custom_labels`` directly -- a
        state no caller could produce while ``__init__`` hard-coded None.
        """
        from llm_router_utils.sglang.srt.server_args import ServerArgs

        sa = ServerArgs(
            model_path="m", tokenizer_metrics_allowed_custom_labels=["keep"]
        )
        s = OpenAIServingBase(SimpleNamespace(server_args=sa))
        got = s.extract_custom_labels(
            self._req({"x-custom-labels": '{"keep": "1", "drop": "2"}'})
        )
        self.assertEqual(got, {"keep": "1"})

    def test_extract_custom_labels_reads_the_configured_header(self):
        from llm_router_utils.sglang.srt.server_args import ServerArgs

        sa = ServerArgs(
            model_path="m",
            tokenizer_metrics_allowed_custom_labels=["keep"],
            tokenizer_metrics_custom_labels_header="x-my-labels",
        )
        s = OpenAIServingBase(SimpleNamespace(server_args=sa))
        self.assertEqual(
            s.extract_custom_labels(self._req({"x-my-labels": '{"keep": "1"}'})),
            {"keep": "1"},
        )

    def test_create_error_response_shape(self):
        s = self._serving()
        resp = s.create_error_response("boom", status_code=422)
        self.assertEqual(resp.status_code, 422)
        import json

        body = json.loads(resp.body)
        self.assertEqual(body["object"], "error")
        self.assertEqual(body["message"], "boom")
        self.assertEqual(body["code"], 422)

    def test_create_streaming_error_response_shape(self):
        import json

        s = self._serving()
        payload = json.loads(s.create_streaming_error_response("boom"))
        self.assertIn("error", payload)
        self.assertEqual(payload["error"]["message"], "boom")


if __name__ == "__main__":
    unittest.main()
