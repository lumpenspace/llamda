"""
This module contains the console utilities for the penger package.
"""

from typing import Callable, Any
from git import Sequence
from rich.console import Console
from rich.live import Live
from rich.json import JSON
from llamda_fn.llms.ll_tool import LLToolResponse
from llamda_fn.llms.ll_message import LLMessage

from llamda_fn.llms.ll_tool import LLToolCall


actions: dict[str, str] = {
    "message": "💬",
    "thinking": "💭",
    "tool_call": "🔧",
    "tool_response": "⚒️",
    "tool_error": "☭",
    "tool_waiting": "⏳",
}

emojis = {
    "user": "🐈",
    "assistant": "🐙",
    "worker": "🧑🏼‍🔧",
    "system": "📐",
}


class Logger:
    l: Console

    def __init__(self):
        self.l = Console()

    def error(self, message: str) -> None:
        self.l.log(f"❌ {message}")

    def set_live(self, live: bool = False) -> None:
        """toggles the live console"""
        self.l.set_live(Live()) if live else self.l.clear_live()

    def msg(self, message: LLMessage | list[LLMessage]) -> None:
        if isinstance(message, list):
            [self.msg(item) for item in message]
            return
        role, content = message.role, message.content
        self.l.log(
            f"""[b]{emojis.get(str(role))}[/b]\t
            {emojis.get("message" if content else "thinking")} """
        )

    def log(self, *args: Any, **argv: Any):
        self.l.log(*args, **argv)

    @classmethod
    def single(cls, logger: "Logger | None | Any") -> "Logger":
        logger = globals().get("LOG")

        if logger and isinstance(logger, cls):
            return logger
        return Logger()

    def tools(
        self, tool_calls: Sequence[LLToolCall]
    ) -> Callable[[LLToolCall, LLToolResponse], None]:
        calls: int = len(tool_calls)
        done = 0
        self.set_live()
        self.l.log(f"🔧📎: {done}/{calls} tool calls detected.")

        def ok(call: LLToolCall, tool_response: LLToolResponse) -> None:
            nonlocal done
            done += 1
            self.l.log(f"🔧📎: {done}/{calls} tool calls detected.")

            self.l.log(
                f"""⚒️ {call.name}:
            ➡️{self.l.log(call.arguments)}    
            ⬅️➡️{JSON.from_data(tool_response.result)}"""
            )
            if done == calls:
                self.l.log("----------")
                self.l.log("All tool calls completed.")
                self.set_live(False)

        return ok


global LOG
LOG: Logger = Logger.single(globals().get("LOG"))

__all__ = ["LOG"]
