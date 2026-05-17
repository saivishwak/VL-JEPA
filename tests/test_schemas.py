import pytest

from llm_jepa.data.schemas import ChatExample


def test_chat_example_requires_assistant_target():
    example = ChatExample.model_validate(
        {
            "messages": [
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": "Answer."},
            ]
        }
    )
    assert example.messages[-1].role == "assistant"


def test_chat_example_rejects_missing_assistant():
    with pytest.raises(ValueError):
        ChatExample.model_validate({"messages": [{"role": "user", "content": "Question?"}]})
