from collections import UserList
from typing import List, Literal, Optional

from openai.types.chat import ChatCompletionMessageParam

from llamda_fn.exchanges.messages import to_message


class Exchange(UserList[ChatCompletionMessageParam]):
    """
    An exchange represents a series of messages between a user and an assistant.
    """

    def __init__(
        self,
        system_message: Optional[str] = None,
        messages: Optional[List[ChatCompletionMessageParam]] = None,
    ):
        """
        Initialize the exchange.
        """
        super().__init__()
        if system_message:
            self.data.append(to_message(system_message, "system"))
        if messages:
            self.data.extend(messages)

    def append(
        self,
        item: str | ChatCompletionMessageParam,
        role: Literal["user", "system", "assistant"] = "user",
    ) -> None:
        """
        Add a message to the exchange.
        """
        if isinstance(item, str):
            self.data.append(to_message(item, role))
        else:
            self.data.append(item)

    def get_last_user_message(self) -> Optional[ChatCompletionMessageParam]:
        """
        Get the last user message in the exchange.
        """
        for message in reversed(self.data):
            if message["role"] == "user":
                return message
        return None

    def get_context(self, n: int = 5) -> List[ChatCompletionMessageParam]:
        """
        Get the last n messages as context.
        """
        return self.data[-n:]

    def __str__(self) -> str:
        """
        String representation of the exchange.
        """
        return "\n".join(
            f"{msg['role']}: {msg.get('content', '')} {msg.get('tool_calls', '')}"
            for msg in self.data
        )
