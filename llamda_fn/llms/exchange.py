from llamda_fn.llms.api_types import LLMessage, Role
from llamda_fn.utils.logger import logger

from collections import UserList
from typing import Any, List, Optional


class Exchange(UserList[LLMessage]):
    """
    An exchange represents a series of messages between a user and an assistant.
    """

    def __init__(
        self,
        system: Optional[str] = None,
        messages: Optional[List[LLMessage]] = None,
    ) -> None:
        super().__init__()
        if system:
            self.append(LLMessage(content=system, role=Role["system"]))
        if messages:
            for message in messages:
                if not message.role:
                    raise ValueError(f"Message missing role: {message}")
                self.append(message)

    def ask(self, text: str) -> None:
        self.data.append(LLMessage(role="user", content=text))

    def append(self, item: LLMessage) -> None:
        """
        Add a message to the exchange.
        """
        logger.msg(item)
        self.data.append(item)

    def get_context(self, n: int = 5) -> list[LLMessage]:
        """
        Get the last n messages as context.
        """
        return self.data[-n:]

    def __str__(self) -> str:
        """
        String representation of the exchange.
        """
        return "\n".join(f"{msg.role}: {msg.content}" for msg in self.data)
