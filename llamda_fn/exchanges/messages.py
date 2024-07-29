"""
This module contains functions related to verbal/conversational messages.
"""

from typing import Any, Optional, Literal

from llamda_fn.llms.api import (
    UserMessage,
    SystemMessage,
    AssistantMessage,
    MessageParam,
)


def to_message(
    text: str,
    role: Literal["user", "system", "assistant"],
    name: Optional[str] = None,
) -> MessageParam:
    """
    Create a message.
    """
    base_dict: dict[str, Any] = {"content": text}
    if name is not None:
        base_dict["name"] = name

    match role:
        case "user":
            message = UserMessage(**base_dict, role="user")
        case "system":
            message = SystemMessage(**base_dict, role="system")
        case "assistant":
            message = AssistantMessage(**base_dict, role="assistant")
    return message
