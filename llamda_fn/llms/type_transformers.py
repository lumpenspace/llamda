from typing import Any
from llamda_fn.llms.api_types import (
    LLMessage,
    OaiResponseMessage,
)


def ll_to_oai_message(message: LLMessage, model: str) -> OaiResponseMessage:
    oai_message: dict[str, Any] = {
        "role": message.role,
        "model": model,
        "content": message.content,
        "tools": [(tool_call) for tool_call in message.tool_calls],
    }
    if message.name:
        oai_message["name"] = message.name
    # We don't need to transform tool_calls here
    return OaiResponseMessage(**oai_message)
