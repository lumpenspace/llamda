"""
This module contains the console utilities for the penger package.
"""

from typing import Callable
from rich.console import Console
from rich.live import Live
from rich.json import JSON

from llamda_fn.llms.api_types import LLMessage, ToolResponse, LlToolCall


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

    def set_live(self, live_console: bool = False) -> None:
        self.set_live(live_console)

    def msg(self, msg: LLMessage) -> None:
        role, content = msg.role, msg.content
        self.l.log(
            f"""[b]{emojis.get(str(role))}[/b]\t\
            {emojis.get("message" if content else "thinking")} """
        )

    def log(self, *args, **argv):
        self.l.log(*args, **argv)

    def tools(
        self, tool_calls: list[LlToolCall]
    ) -> Callable[[LlToolCall, ToolResponse], None]:
        calls: int = len(tool_calls)
        done = 0
        self.set_live()
        self.l.log(f"🔧📎: {done}/{calls} tool calls detected.")

        def ok(call: LlToolCall, tool_response: ToolResponse) -> None:
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


global logger
logger = None

if not logger:
    logger = Logger()

__all__ = ["logger"]
