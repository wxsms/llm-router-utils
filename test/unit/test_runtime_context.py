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

import ast
import pathlib
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
    reset_runtime_context,
    set_runtime_context,
    use_runtime_context,
)
from llm_router_utils.sglang.srt.server_args import PortArgs, ServerArgs

# The parser fields whose unresolved "auto" sentinel must raise, not leak.
from llm_router_utils.sglang.srt.managers.tokenizer_manager import (
    _PARSER_FIELDS as _PARSER_FIELD_NAMES,
)

_SRT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "llm_router_utils"
    / "sglang"
    / "srt"
)


def _attributes_read_off(root_name: str) -> set[str]:
    """Attribute names read off a shim accessor, found by walking the AST.

    A regex only saw the inline ``get_serving().attr`` shape and missed
    ``serving = get_serving(); serving.attr`` and
    ``getattr(get_serving(), "attr")``. This walks ``Attribute``/``getattr``
    nodes whose value is a direct call to ``root_name``.
    """
    found: set[str] = set()
    for path in _SRT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == root_name
            ):
                found.add(node.attr)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Name)
                and node.args[0].func.id == root_name
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                found.add(node.args[1].value)
    return found


def _config_value_keys() -> set[str]:
    """Field names passed to ``TokenizerManager.config_value("...")``."""
    found: set[str] = set()
    for path in _SRT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "config_value"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                found.add(node.args[0].value)
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
            tmm.TokenizerManager(ServerArgs(model_path="dummy", skip_tokenizer_init=True), PortArgs())

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
            tmm.TokenizerManager(ServerArgs(model_path="dummy", skip_tokenizer_init=True), PortArgs())

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


class TestManagerSurface(unittest.TestCase):
    """Defect C1: attributes serving_chat reads straight off the manager."""

    def test_every_manager_attribute_read_exists(self):
        """Any *reachable* ``tokenizer_manager.<attr>`` read must exist.

        ``serving_chat.py`` is byte-identical to upstream, so it still contains
        the request-*serving* methods (``generate_request``, ``create_abort_task``,
        streaming) that this repo strips at the base class. Those are not
        reachable from the public entry point, so they are excluded here.
        """
        unreachable = {
            "create_abort_task",  # used by stripped OpenAIServingBase.handle_request
            "generate_request",  # ditto; also the streaming generator
        }
        reads: set[str] = set()
        for path in _SRT.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "tokenizer_manager"
                ):
                    reads.add(node.attr)
        reads -= unreachable
        self.assertGreater(len(reads), 0, "no tokenizer_manager reads discovered")

        import llm_router_utils.sglang.srt.managers.tokenizer_manager as tmm

        with patch.object(tmm, "ModelConfig") as model_config_cls:
            model_config_cls.return_value.image_token_id = None
            manager = tmm.TokenizerManager(
                ServerArgs(
                    model_path="m",
                    served_model_name="alias",
                    skip_tokenizer_init=True,
                ),
                PortArgs(),
            )
        for name in sorted(reads):
            with self.subTest(attribute=name):
                self.assertTrue(
                    hasattr(manager, name),
                    f"TokenizerManager is missing {name!r}, read by serving code",
                )

    def test_served_model_name_defaults_to_model_path(self):
        import llm_router_utils.sglang.srt.managers.tokenizer_manager as tmm

        with patch.object(tmm, "ModelConfig") as model_config_cls:
            model_config_cls.return_value.image_token_id = None
            manager = tmm.TokenizerManager(
                ServerArgs(model_path="m", skip_tokenizer_init=True), PortArgs()
            )
        self.assertEqual(manager.served_model_name, "m")

    def test_revision_reaches_the_model_config(self):
        import llm_router_utils.sglang.srt.managers.tokenizer_manager as tmm

        with patch.object(tmm, "ModelConfig") as model_config_cls:
            model_config_cls.return_value.image_token_id = None
            tmm.TokenizerManager(
                ServerArgs(
                    model_path="m", revision="refs/pr/42", skip_tokenizer_init=True
                ),
                PortArgs(),
            )
        self.assertEqual(model_config_cls.call_args.kwargs.get("revision"), "refs/pr/42")


class TestConfigValueResolution(unittest.TestCase):
    """Defect C2: ``config_value`` must return the *resolved* value."""

    def _manager(self, **kwargs):
        import llm_router_utils.sglang.srt.managers.tokenizer_manager as tmm

        with patch.object(tmm, "ModelConfig") as model_config_cls:
            model_config_cls.return_value.image_token_id = None
            return tmm.TokenizerManager(
                ServerArgs(model_path="m", skip_tokenizer_init=True, **kwargs), PortArgs()
            )

    def test_declaration_wins_over_raw_input(self):
        manager = self._manager(tool_call_parser="auto", reasoning_parser="auto")
        declare_resolution(
            manager.server_args,
            "detection",
            tool_call_parser="qwen",
            reasoning_parser="qwen3",
        )
        self.assertEqual(manager.config_value("tool_call_parser"), "qwen")
        self.assertEqual(manager.config_value("reasoning_parser"), "qwen3")

    def test_undeclared_field_falls_back_to_raw_input(self):
        manager = self._manager(tool_call_parser="hermes")
        self.assertEqual(manager.config_value("tool_call_parser"), "hermes")

    def test_grammar_backend_has_a_declared_default(self):
        manager = self._manager()
        self.assertEqual(manager.config_value("grammar_backend"), "xgrammar")

    def test_every_config_value_key_resolves_to_a_usable_value(self):
        """Generic guard over the keys the retained tree actually asks for.

        Each field is constructed *with* the ``auto`` sentinel so the assertion
        can actually fire. The earlier version built ``ServerArgs`` without it,
        which made ``assertNotEqual(value, "auto")`` vacuously true - a guard
        that could never fail.

        There are two acceptable outcomes for the parser fields, and each is
        asserted rather than skipped: either ``config_value`` raises the
        documented ``ValueError`` backstop, or it returns a real value. What must
        never reach a caller is the unresolved ``"auto"`` sentinel, which makes
        ``ToolCallParserEnum.get("auto")`` None (tool calls accepted then
        silently dropped) and ``ReasoningParser`` raise.
        """
        keys = _config_value_keys()
        self.assertGreater(len(keys), 0, "no config_value() reads discovered")
        # Force detection to be a no-op so the sentinel survives to config_value.
        asserted = 0
        raised = 0
        with patch.object(
            type(self._manager()), "resolve_auto_parsers", return_value=None
        ):
            for name in sorted(keys):
                with self.subTest(field=name):
                    if not hasattr(ServerArgs(model_path="m"), name):
                        # Field exists only as a declaration upstream (e.g.
                        # grammar_backend); it cannot be passed to ServerArgs.
                        manager = self._manager()
                    else:
                        manager = self._manager(**{name: "auto"})
                    try:
                        value = manager.config_value(name)
                    except ValueError:
                        # The backstop for an unresolved sentinel is the
                        # documented behaviour for the parser fields; for any
                        # other field it would be a bug.
                        self.assertIn(
                            name,
                            _PARSER_FIELD_NAMES,
                            f"config_value({name!r}) raised but is not a parser field",
                        )
                        raised += 1
                    else:
                        self.assertNotEqual(
                            value,
                            "auto",
                            f"config_value({name!r}) leaked the unresolved sentinel",
                        )
                        asserted += 1
        # Without this, a future change that made every field raise would leave
        # the loop asserting nothing at all and the test would still pass.
        self.assertEqual(
            asserted + raised,
            len(keys),
            "some config_value key was neither asserted nor expected to raise",
        )
        self.assertGreater(
            raised,
            0,
            "no parser field exercised the unresolved-sentinel backstop",
        )

    def test_unresolved_auto_sentinel_raises_instead_of_leaking(self):
        """The backstop: if detection never ran, config_value must not hand a
        caller ``"auto"`` to treat as a parser name."""
        with patch.object(
            type(self._manager()), "resolve_auto_parsers", return_value=None
        ):
            manager = self._manager(tool_call_parser="auto")
        with self.assertRaises(ValueError):
            manager.config_value("tool_call_parser")

    def test_manager_calls_resolve_auto_parsers(self):
        """The resolution must actually run when detection is requested."""
        import llm_router_utils.sglang.srt.managers.tokenizer_manager as tmm

        with patch.object(tmm.TokenizerManager, "resolve_auto_parsers") as resolver:
            with patch.object(tmm, "ModelConfig") as model_config_cls:
                model_config_cls.return_value.image_token_id = None
                tmm.TokenizerManager(
                    ServerArgs(model_path="m", tool_call_parser="auto"),
                    PortArgs(),
                )
        resolver.assert_called_once()

    def test_skipped_when_no_parser_is_auto(self):
        """No ``auto`` means nothing to detect, so no detection cost."""
        import llm_router_utils.sglang.srt.managers.tokenizer_manager as tmm

        with patch.object(tmm.TokenizerManager, "resolve_auto_parsers") as resolver:
            with patch.object(tmm, "ModelConfig") as model_config_cls:
                model_config_cls.return_value.image_token_id = None
                tmm.TokenizerManager(
                    ServerArgs(model_path="m", tool_call_parser="hermes"), PortArgs()
                )
        resolver.assert_not_called()

    def test_skip_tokenizer_init_is_not_bypassed_by_detection(self):
        """``skip_tokenizer_init`` promises no tokenizer load.

        Detection performs one of its own, so it must not run when that flag is
        set - otherwise the contract is silently violated for every ``auto``
        user.
        """
        import llm_router_utils.sglang.srt.managers.tokenizer_manager as tmm

        with patch.object(tmm.TokenizerManager, "resolve_auto_parsers") as resolver:
            with patch.object(tmm.TokenizerManager, "init_tokenizer_and_processor"):
                with patch.object(tmm, "ModelConfig") as model_config_cls:
                    model_config_cls.return_value.image_token_id = None
                    tmm.TokenizerManager(
                        ServerArgs(
                            model_path="m",
                            tool_call_parser="auto",
                            skip_tokenizer_init=True,
                        ),
                        PortArgs(),
                    )
        resolver.assert_not_called()


class TestRuntimeContextScoping(unittest.TestCase):
    """Defects C3/C4: per-instance isolation and clearable state."""

    def tearDown(self):
        reset_runtime_context()

    def test_none_clears_the_slot(self):
        set_runtime_context(
            server_args=ServerArgs(model_path="a", hf_chat_template_name="tool_use")
        )
        self.assertEqual(get_serving().hf_chat_template_name, "tool_use")
        set_runtime_context(server_args=None)
        self.assertIsNone(get_serving().hf_chat_template_name)

    def test_reset_runtime_context_clears_everything(self):
        set_runtime_context(server_args=ServerArgs(model_path="a"))
        reset_runtime_context()
        self.assertIsNone(get_serving().hf_chat_template_name)

    def test_omitted_argument_leaves_slot_unchanged(self):
        set_runtime_context(server_args=ServerArgs(model_path="a", hf_chat_template_name="x"))
        set_runtime_context(model_config=None)  # server_args omitted -> unchanged
        self.assertEqual(get_serving().hf_chat_template_name, "x")

    def test_scope_overrides_process_default_for_its_duration(self):
        """A multi-model router must be able to read each model's own config."""
        a = ServerArgs(model_path="A", hf_chat_template_name="tool_use")
        b = ServerArgs(model_path="B", hf_chat_template_name=None)
        set_runtime_context(server_args=b)  # B registered last: process default

        with use_runtime_context(server_args=a):
            self.assertEqual(get_serving().hf_chat_template_name, "tool_use")
        # restored to the process default afterwards
        self.assertIsNone(get_serving().hf_chat_template_name)


class TestMultiModelIsolation(unittest.TestCase):
    """C1: a multi-model router must not serve A with B's setup-time config.

    ``hf_chat_template_name`` and ``default_chat_template_kwargs`` are read at
    construction, so only a scope around *construction* isolates them. This test
    pins both halves of that: construction-scoped works, request-scoped does not.
    """

    def tearDown(self):
        reset_runtime_context()

    @staticmethod
    def _args(model_path, template_name, effort):
        return ServerArgs(
            model_path=model_path,
            hf_chat_template_name=template_name,
            default_chat_template_kwargs={"reasoning_effort": effort},
            skip_tokenizer_init=True,
        )

    def _build(self, args):
        import llm_router_utils.sglang.srt.managers.tokenizer_manager as tmm

        with patch.object(tmm, "ModelConfig") as model_config_cls:
            model_config_cls.return_value.image_token_id = None
            return tmm.TokenizerManager(args, PortArgs())

    def test_construction_scoped_builds_see_their_own_config(self):
        a = self._args("A", "tool_use", "high")
        b = self._args("B", None, "low")

        with use_runtime_context(server_args=a, model_config=None):
            self.assertEqual(get_serving().hf_chat_template_name, "tool_use")
            self.assertEqual(
                get_serving().default_chat_template_kwargs, {"reasoning_effort": "high"}
            )
            self._build(a)

        with use_runtime_context(server_args=b, model_config=None):
            self.assertIsNone(get_serving().hf_chat_template_name)
            self.assertEqual(
                get_serving().default_chat_template_kwargs, {"reasoning_effort": "low"}
            )
            self._build(b)

    def test_last_writer_bleeds_into_the_process_default(self):
        """Documents the hazard the README warns about."""
        a = self._args("A", "tool_use", "high")
        b = self._args("B", None, "low")
        self._build(a)
        self._build(b)

        # Process default now reflects B, not A.
        self.assertIsNone(get_serving().hf_chat_template_name)
        self.assertEqual(
            get_serving().default_chat_template_kwargs, {"reasoning_effort": "low"}
        )
        # ...while A's own ServerArgs still carries the operator's intent, which
        # is why a caller can recover by reading the instance, not the global.
        self.assertEqual(a.hf_chat_template_name, "tool_use")
        self.assertEqual(a.default_chat_template_kwargs, {"reasoning_effort": "high"})

    def test_request_time_scope_cannot_restore_setup_time_values(self):
        """The trap the docs previously described wrongly.

        A value captured at construction is baked into the instance, so entering
        a scope later changes what *new* reads see, not what was stored.
        Asserted against a real constructed object: the previous version
        compared a dict literal to itself and could never fail.
        """
        a = self._args("A", "tool_use", "high")
        b = self._args("B", None, "low")

        # Build A under A's scope; its __init__ captures A's values.
        with use_runtime_context(server_args=a, model_config=None):
            self._build(a)
            serving_a = self._make_serving(a)
        self.assertEqual(
            serving_a.default_chat_template_kwargs, {"reasoning_effort": "high"}
        )

        # Building B afterwards moves the process default to B.
        self._build(b)
        self.assertEqual(
            get_serving().default_chat_template_kwargs, {"reasoning_effort": "low"}
        )

        # A's instance still holds what it captured, with no scope open.
        self.assertEqual(
            serving_a.default_chat_template_kwargs,
            {"reasoning_effort": "high"},
            "a scope opened after construction must not retroactively change "
            "what the instance captured",
        )

        # A request-time scope does not rewrite the captured attribute either.
        with use_runtime_context(server_args=b):
            self.assertEqual(
                serving_a.default_chat_template_kwargs,
                {"reasoning_effort": "high"},
            )

    def _make_serving(self, args):
        """Construct a real OpenAIServingChat so setup-time capture is exercised."""
        import llm_router_utils.sglang.srt.managers.tokenizer_manager as tmm
        from llm_router_utils.sglang.srt.entrypoints.openai.serving_chat import (
            OpenAIServingChat,
        )
        from llm_router_utils.sglang.srt.parser.template_manager import TemplateManager

        with patch.object(tmm, "ModelConfig") as model_config_cls:
            model_config_cls.return_value.image_token_id = None
            manager = tmm.TokenizerManager(args, PortArgs())
        return OpenAIServingChat(manager, TemplateManager())


class TestServingConfigDefaultsCoverServerArgs(unittest.TestCase):
    """Structural guard for the shim-fidelity defect class.

    ``_ServingConfig`` falls back to ``_DEFAULTS`` for any name absent from the
    bound ``ServerArgs``. A key that exists only in ``_DEFAULTS`` is therefore
    frozen at its default forever and the operator's setting is silently
    ignored -- which is exactly what happened to ``return_input_ids``,
    ``return_output_ids`` and ``trust_request_chat_template``.

    The per-attribute test above cannot catch that: ``_DEFAULTS`` always
    resolves, so the read never fails. This asserts the two sets are aligned.
    """

    def test_every_serving_default_is_a_real_server_args_field(self):
        import dataclasses

        from llm_router_utils.sglang.srt.runtime_context import _ServingConfig
        from llm_router_utils.sglang.srt.server_args import ServerArgs

        field_names = {f.name for f in dataclasses.fields(ServerArgs)}
        orphans = sorted(set(_ServingConfig._DEFAULTS) - field_names)
        self.assertEqual(
            orphans,
            [],
            "these _ServingConfig defaults have no ServerArgs field, so an "
            "operator's value can never take effect: "
            f"{orphans}. Add the field to ServerArgs (see upstream "
            "arg_groups/fields/serving.py).",
        )

    def test_operator_values_survive_the_defaults_table(self):
        """A field that *does* exist must carry the operator's value through."""
        import dataclasses

        from llm_router_utils.sglang.srt.runtime_context import (
            _ServingConfig,
            get_serving,
            reset_runtime_context,
            set_runtime_context,
        )

        bool_fields = [
            f.name
            for f in dataclasses.fields(ServerArgs)
            if f.name in _ServingConfig._DEFAULTS
            and isinstance(f.default, bool)
            and f.default is False
        ]
        self.assertGreater(len(bool_fields), 0, "no inspectable bool defaults found")

        for name in bool_fields:
            with self.subTest(field=name):
                set_runtime_context(server_args=ServerArgs(model_path="m", **{name: True}))
                try:
                    self.assertTrue(
                        getattr(get_serving(), name),
                        f"{name}=True was overridden by _ServingConfig._DEFAULTS",
                    )
                finally:
                    reset_runtime_context()


if __name__ == "__main__":
    unittest.main()
