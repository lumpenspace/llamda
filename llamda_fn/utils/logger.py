"""
This module contains the console utilities for the penger package.
"""

from typing import Sequence
from contextlib import contextmanager
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON
from llamda_fn.llms.ll_tool import LLToolCall
from llamda_fn.llms.ll_message import LLMessage


class Logger:
    def __init__(self):
        self.console = Console()
        self.live = None

    def error(self, message: str) -> None:
        self.console.print(f"[bold red]❌ Error:[/bold red] {message}")

    @contextmanager
    def live_logging(self):
        with Live(console=self.console, auto_refresh=False) as live:
            self.live = live
            yield
            self.live = None

    def msg(self, message: LLMessage) -> None:
        role_emoji = {
            "user": "🐈",
            "assistant": "🐙",
            "worker": "🧑🏼‍🔧",
            "system": "📐",
        }.get(str(message.role), "❓")

        content_emoji = "💬" if message.content else "💭"

        panel = Panel(
            f"{content_emoji} {message.content}",
            title=f"[bold]{role_emoji} {message.role.capitalize()}[/bold]",
            border_style="blue",
        )
        self.console.print(panel)

    def tool_calls(self, tool_calls: Sequence[LLToolCall]) -> None:
        table = Table(title="Tool Calls", show_header=True, header_style="bold magenta")
        table.add_column("Tool", style="dim")
        table.add_column("Arguments")
        table.add_column("Result")

        for call in tool_calls:
            result = self.functions.execute_function(tool_call=call)
            table.add_row(
                call.name, JSON.from_data(call.arguments), JSON.from_data(result.result)
            )

        self.console.print(table)

    @classmethod
    def get_instance(cls) -> "Logger":
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance


LOG = Logger.get_instance()

__all__ = ["LOG"]
