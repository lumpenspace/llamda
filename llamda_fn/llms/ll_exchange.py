"""LLExchange"""

from collections import UserList
from functools import cached_property
from typing import List, Optional

from .ll_message import LLMessage
from .oai_api_types import OaiMessage

__all__: list[str] = ["LLExchange"]


class LLExchange(UserList[LLMessage]):
    """
    An exchange represents a series of messages between a user and an assistant.
    """

    def __init__(
        self,
        system: Optional[str | LLMessage] = None,
        messages: Optional[List[LLMessage]] = None,
    ) -> None:
        super().__init__()
        if system:
            self.append(
                LLMessage(content=system, role="system")
                if isinstance(system, str)
                else system
            )
        if messages:
            for message in messages:
                if not message.role:
                    raise ValueError(f"Message missing role: {message}")
                self.append(message)

    def ask(self, text: str) -> None:
        """
        Adds a user message to the exchange,  by text.
        """
        self.data.append(LLMessage(role="user", content=text))

    def get_context(self, n: int = 5) -> list[LLMessage]:
        """
        Get the last n messages as context.

        Args:
            n (int): The number of recent messages to return. Defaults to 5.

        Returns:
            list[LLMessage]: A list of the last n messages in the exchange.
        """
        return self.data[-n:]

    def __str__(self) -> str:
        """
        String representation of the exchange.
        """
        return "\n".join(f"{msg.role}: {msg.content}" for msg in self.data)

    @cached_property
    def oai_props(self) -> List[OaiMessage]:
        """
        The exchange as a list of messahes to use with OpenAI-like APIs.
        """
        return [message.oai_props for message in self.data]

    def to_dict(self):
        return [message.model_dump() for message in self]