from typing import Any, List
from llamda_fn.llms.api_types import (
    Role,
    LlToolCall,
    OaiUserMessage,
    OaiSystemMessage,
    OaiAssistantMessage,
)


def make_oai_message(
    role: Role,
    content: str,
    name: str | None = None,
    tool_calls: List[LlToolCall] | None = None,
    **kwargs: Any,
) -> OaiUserMessage | OaiSystemMessage | OaiAssistantMessage:
    kwargs = {}
    if name:
        kwargs["name"] = name
    match role:
        case "user":
            return OaiUserMessage(
                content=content,
                **kwargs,
            )
        case "system":
            return OaiSystemMessage(
                content=content,
                **kwargs,
            )
        case "assistant":

            if tool_calls:
                kwargs["tool_calls"] = [
                    tool_call.model_dump() for tool_call in tool_calls
                ]
            return OaiAssistantMessage(
                content=content,
                **kwargs,
            )
        case _:
            raise ValueError(f"Invalid role: {role}")


OaiRoleMessage: dict[
    Role, type[OaiUserMessage] | type[OaiSystemMessage] | type[OaiAssistantMessage]
] = {
    "user": OaiUserMessage,
    "system": OaiSystemMessage,
    "assistant": OaiAssistantMessage,
}
