from llamda_fn.llms.api_types import LLMessage


from collections import UserList
from typing import List, Optional


class Exchange(UserList[LLMessage]):
    """
    An exchange represents a series of messages between a user and an assistant.
    """

    def __init__(
        self,
        system_message: Optional[str] = None,
        messages: Optional[List[LLMessage]] = None,
    ) -> None:
        """
        Initialize the exchange.
        """
        super().__init__()
        if system_message:
            self.data.append(LLMessage(content=system_message, role="system"))
        if messages:
            self.data.extend(messages)

    def ask(self, content: str) -> None:
        """
        Add a user message to the exchange.
        """
        self.data.append(LLMessage(content=content, role="user"))

    def append(self, item: LLMessage) -> None:
        """
        Add a message to the exchange.
        """

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