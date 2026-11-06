"""Structural guard: every keyword ``serving_chat.py`` passes must be a real field.

This is the same defect class that was Critical three times on this branch: a
byte-identical upstream file constructs a *slimmed* class with keyword arguments
that the slimming dropped. Nothing in this repo calls the affected method, so no
runtime test sees the ``TypeError`` -- it surfaces the first time a consumer
composes the documented call chain.

``GenerateReqInput`` was missing 16 keywords that ``_convert_to_internal_request``
passes. This walks every ``ClassName(**kwargs)`` construction in the retained
tree, resolves the class, and asserts each keyword is a field.
"""

import ast
import dataclasses
import importlib
import pathlib
import unittest

#: This file lives at ``test/unit/managers/``, so the repo root is three levels
#: up. An earlier revision used ``parents[4]`` -- copied from
#: ``test/unit/entrypoints/openai/test_serving_base_surface.py``, which sits two
#: directories deeper -- and silently pointed at ``<drive>\src\...``, a path that
#: does not exist. ``_keyword_args`` then returned an empty set for every class
#: and the guard below asserted ``[] == []`` forever. ``test_the_scan_has_
#: call_sites_to_scan`` now pins this: it fails on any path regression instead of
#: passing vacuously.
_SRT = pathlib.Path(__file__).resolve().parents[3] / "src" / "llm_router_utils" / "sglang" / "srt"

#: Floors on how many construction sites the scan must find. These are not exact
#: counts (adding a call site must not break CI); they are high enough that a
#: path or parsing regression drops below them and fails, which is the failure
#: mode that made this file vacuous. ``EmbeddingReqInput`` is deliberately absent:
#: no retained file constructs it today, so it has no floor to defend.
_MIN_CALL_SITES = {
    "GenerateReqInput": 30,
    "MessageProcessingResult": 5,
}

#: Construction sites to audit, and how to import the class they build.
_TARGETS = {
    "GenerateReqInput": "llm_router_utils.sglang.srt.managers.io_struct",
    "EmbeddingReqInput": "llm_router_utils.sglang.srt.managers.io_struct",
    "MessageProcessingResult": (
        "llm_router_utils.sglang.srt.entrypoints.openai.protocol"
    ),
}


def _keyword_args(class_name: str) -> set[str]:
    """Union of keyword names used at every ``class_name(...)`` call site."""
    found: set[str] = set()
    for path in _SRT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == class_name
            ):
                found |= {k.arg for k in node.keywords if k.arg}
    return found


class TestConstructionSurface(unittest.TestCase):
    def test_retained_tree_resolves(self):
        """The scan root must exist before any assertion about it is meaningful.

        Without this, a wrong ``parents[...]`` depth turns every assertion below
        into ``[] == []``. That is not hypothetical: it is exactly what shipped.
        """
        self.assertTrue(
            _SRT.is_dir(),
            f"scan root does not exist: {_SRT}. A wrong parents[...] depth makes "
            "every keyword assertion in this file vacuous.",
        )
        self.assertTrue(
            any(_SRT.rglob("*.py")),
            f"scan root contains no Python files: {_SRT}",
        )

    def test_the_scan_has_call_sites_to_scan(self):
        """Each guarded class must actually be constructed somewhere.

        A path or AST regression shows up here as a drop to zero rather than as a
        silently-passing equality check below.
        """
        for class_name, floor in _MIN_CALL_SITES.items():
            with self.subTest(cls=class_name):
                used = _keyword_args(class_name)
                self.assertGreaterEqual(
                    len(used),
                    floor,
                    f"only {len(used)} keyword(s) found at {class_name}(...) call "
                    f"sites, expected at least {floor}. The scan is no longer "
                    "reaching the retained tree, so the guard below is vacuous.",
                )

    def test_every_constructed_keyword_is_a_field(self):
        for class_name, module_path in _TARGETS.items():
            with self.subTest(cls=class_name):
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)

                if dataclasses.is_dataclass(cls):
                    fields = {f.name for f in dataclasses.fields(cls)}
                else:  # pydantic / msgspec models
                    model_fields = getattr(cls, "model_fields", None)
                    if model_fields is None:
                        model_fields = getattr(cls, "__annotations__", {})
                    fields = set(model_fields)

                used = _keyword_args(class_name)
                missing = sorted(used - fields)
                self.assertEqual(
                    missing,
                    [],
                    f"{class_name} is constructed with keywords it has no field "
                    f"for: {missing}. The caller is upstream-identical, so add "
                    "the fields (and their upstream types/defaults) rather than "
                    "changing the call.",
                )

    def test_generate_req_input_holds_the_documented_call_chain_fields(self):
        """Spot-check the exact keywords that were missing."""
        from llm_router_utils.sglang.srt.managers.io_struct import GenerateReqInput

        names = {f.name for f in dataclasses.fields(GenerateReqInput)}
        for required in (
            "return_sampling_mask",
            "return_text_in_logprobs",
            "cache_salt",
            "routing_key",
            "custom_labels",
            "custom_logit_processor",
            "images_config",
            "video_config",
            "mm_processor_kwargs",
            "image_max_dynamic_patch",
            "video_max_dynamic_patch",
            "max_dynamic_patch",
            "use_audio_in_video",
            "return_prompt_token_ids",
        ):
            with self.subTest(field=required):
                self.assertIn(required, names)


if __name__ == "__main__":
    unittest.main()
