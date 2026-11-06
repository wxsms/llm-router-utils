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

_SRT = (
    pathlib.Path(__file__).resolve().parents[4]
    / "src"
    / "llm_router_utils"
    / "sglang"
    / "srt"
)

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
