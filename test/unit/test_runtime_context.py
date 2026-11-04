"""Regression tests for the runtime-context / arg-groups shims.

These guard the two defects found during the v0.9.0 code review, both of which
were invisible to the existing suite because they only fire on code paths that
no test exercised:

1. ``_ServingConfig`` used a hand-maintained allowlist that omitted
   ``hf_chat_template_name``, silently costing every model with a dict of named
   HF chat templates its chat template (the ``AttributeError`` was swallowed by
   an upstream ``except Exception`` and degraded to a warning).
2. ``set_runtime_context`` had no callers, so ``get_model()`` failed closed and
   every ``dsv4``/``dsv41``/``inkling`` construction raised an opaque
   ``AttributeError`` instead of the intended ``ImportError``.

The first test is deliberately *generic*: it iterates every attribute the
retained tree reads off the shims and asserts it resolves, so a future upstream
rename fails here rather than silently in production.
"""

import pathlib
import re
import unittest
from unittest.mock import patch

from llm_router_utils.sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolution_result,
    resolving_view,
)
from llm_router_utils.sglang.srt.runtime_context import (
    get_model,
    get_serving,
    set_runtime_context,
)
from llm_router_utils.sglang.srt.server_args import PortArgs, ServerArgs

_SRT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "llm_router_utils"
    / "sglang"
    / "srt"
)


def _attributes_read_off(root_name: str) -> set[str]:
    """Every ``<root_name>().attr`` read across the retained tree."""
    found: set[str] = set()
    for path in _SRT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        found |= set(re.findall(rf"{root_name}\(\)\.(\w+)", text))
    return found


class TestServingConfigSurface(unittest.TestCase):
    def test_every_serving_attribute_read_resolves(self):
        """Generic guard: any field the retained tree reads must resolve."""
        names = _attributes_read_off("get_serving")
        self.assertGreater(len(names), 0, "no get_serving() reads found to check")
        serving = get_serving()
        for name in sorted(names):
            with self.subTest(attribute=name):
                getattr(serving, name)  # raises AttributeError on drift

    def test_passes_through_arbitrary_server_args_fields(self):
        """Not an allowlist: any real ServerArgs field is visible."""
        set_runtime_context(
            server_args=ServerArgs(model_path="m", hf_chat_template_name="tool_use")
        )
        try:
            self.assertEqual(get_serving().hf_chat_template_name, "tool_use")
            # a field no serving-specific code mentions
            self.assertEqual(get_serving().tokenizer_mode, "auto")
        finally:
            set_runtime_context(server_args=None)

    def test_unknown_attribute_still_raises(self):
        set_runtime_context(server_args=ServerArgs(model_path="m"))
        try:
            with self.assertRaises(AttributeError):
                get_serving().definitely_not_a_real_field
        finally:
            set_runtime_context(server_args=None)

    def test_defaults_when_nothing_installed(self):
        set_runtime_context(server_args=None)
        self.assertIsNone(get_serving().hf_chat_template_name)
        self.assertFalse(get_serving().return_input_ids)


class TestNamedChatTemplateSelection(unittest.TestCase):
    """End-to-end: the dict-of-templates branch that defect #1 broke."""

    def test_dict_templates_select_the_configured_name(self):
        from llm_router_utils.sglang.srt.parser.template_manager import TemplateManager

        templates = {"default": "TPL-DEFAULT", "tool_use": "TPL-TOOLUSE"}
        mgr = TemplateManager.__new__(TemplateManager)

        set_runtime_context(
            server_args=ServerArgs(model_path="m", hf_chat_template_name="tool_use")
        )
        try:
            self.assertEqual(
                TemplateManager._select_named_template(mgr, templates, None),
                "TPL-TOOLUSE",
            )
        finally:
            set_runtime_context(server_args=None)

    def test_dict_templates_fall_back_to_first(self):
        from llm_router_utils.sglang.srt.parser.template_manager import TemplateManager

        templates = {"default": "TPL-DEFAULT", "tool_use": "TPL-TOOLUSE"}
        mgr = TemplateManager.__new__(TemplateManager)

        set_runtime_context(server_args=ServerArgs(model_path="m"))
        try:
            self.assertEqual(
                TemplateManager._select_named_template(mgr, templates, None),
                "TPL-DEFAULT",
            )
        finally:
            set_runtime_context(server_args=None)

    def test_missing_name_raises_rather_than_silently_defaulting(self):
        from llm_router_utils.sglang.srt.parser.template_manager import TemplateManager

        mgr = TemplateManager.__new__(TemplateManager)
        set_runtime_context(
            server_args=ServerArgs(model_path="m", hf_chat_template_name="nope")
        )
        try:
            with self.assertRaises(ValueError):
                TemplateManager._select_named_template(
                    mgr, {"default": "D"}, None
                )
        finally:
            set_runtime_context(server_args=None)


class TestRuntimeContextWiring(unittest.TestCase):
    """Defect #2: the setter must actually be called by the manager."""

    def test_tokenizer_manager_installs_the_model_context(self):
        import llm_router_utils.sglang.srt.managers.tokenizer_manager as tmm

        with patch.object(tmm, "ModelConfig") as model_config_cls:
            model_config_cls.return_value.revision = "rev-9"
            model_config_cls.return_value.context_length = 8192
            model_config_cls.return_value.image_token_id = None
            tmm.TokenizerManager(ServerArgs(model_path="dummy"), PortArgs())

        # get_model() must no longer fail closed.
        self.assertEqual(get_model().revision, "rev-9")
        self.assertEqual(get_model().context_length, 8192)

    def test_every_model_attribute_read_resolves_after_wiring(self):
        import llm_router_utils.sglang.srt.managers.tokenizer_manager as tmm

        names = _attributes_read_off("get_model")
        self.assertGreater(len(names), 0, "no get_model() reads found to check")

        with patch.object(tmm, "ModelConfig") as model_config_cls:
            for name in names:
                setattr(model_config_cls.return_value, name, "value-" + name)
            model_config_cls.return_value.image_token_id = None
            tmm.TokenizerManager(ServerArgs(model_path="dummy"), PortArgs())

        for name in sorted(names):
            with self.subTest(attribute=name):
                self.assertEqual(getattr(get_model(), name), "value-" + name)


class TestResolvingViewIsLive(unittest.TestCase):
    """Defect #3: a view must observe declarations made after construction."""

    def test_view_observes_later_declarations(self):
        args = ServerArgs(model_path="m")
        view = resolving_view(args)  # constructed BEFORE the declaration

        declare_resolution(args, "detection", reasoning_parser="deepseek-r1")

        self.assertEqual(view.reasoning_parser, "deepseek-r1")
        self.assertEqual(resolution_result(args, "reasoning_parser"), "deepseek-r1")
        # the raw input is never mutated
        self.assertIsNone(args.reasoning_parser)

    def test_view_is_read_only(self):
        view = resolving_view(ServerArgs(model_path="m"))
        with self.assertRaises(AttributeError):
            view.reasoning_parser = "x"

    def test_last_declaration_wins(self):
        args = ServerArgs(model_path="m")
        declare_resolution(args, "first", reasoning_parser="a")
        declare_resolution(args, "second", reasoning_parser="b")
        self.assertEqual(resolution_result(args, "reasoning_parser"), "b")


if __name__ == "__main__":
    unittest.main()
