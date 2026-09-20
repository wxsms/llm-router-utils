import sys

import pytest

from llm_router_utils.sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from llm_router_utils.sglang.srt.parser.hunyuan_reasoning import normalize_hunyuan_reasoning_effort
from llm_router_utils.sglang.srt.parser.template_detection import ReasoningToggleConfig



@pytest.mark.parametrize(
    ("effort", "normalized"),
    [
        (None, "high"),
        ("none", "no_think"),
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "high"),
        ("high", "high"),
        ("xhigh", "high"),
        ("max", "high"),
    ],
)
def test_hunyuan_reasoning_effort_normalization(effort, normalized):
    request = ChatCompletionRequest(
        model="x",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=effort,
    )

    normalize_hunyuan_reasoning_effort(
        request,
        reasoning_parser="hunyuan",
        reasoning_config=ReasoningToggleConfig(special_case="hunyuan_effort"),
    )

    assert request.reasoning_effort == normalized


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
