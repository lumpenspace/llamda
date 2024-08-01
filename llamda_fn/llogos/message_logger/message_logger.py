"""
This module contains the console utilities for the penger package.
"""

from rich.console import Console
from rich.panel import Panel

from llamda_fn.llms.ll_message import LLMessage
from llamda_fn.llogos.table_logger import RowStatus
from .interfaces import IMessageLogger
from .role_emojis import get_role_emoji
from .tool_calls_logger import ToolCallsLogger

__all__ = ["IMessageLogger", "MessageLogger"]


class MessageLogger(IMessageLogger):
    """Logs messages and tool calls"""

    def __init__(self, *, console: Console) -> None:
        self.console: Console = console
        self.tool = ToolCallsLogger(console=console)

    def __call__(self, message: LLMessage) -> None:
        if message.tool_calls:
            self.tool.init(message.tool_calls)
        panel = Panel(
            f"{get_role_emoji(message.role)} {message.content}",
            title=f"[bold]{get_role_emoji(message.role)} {message.role.capitalize()}[/bold]",
            border_style="blue",
        )
        self.console.print(panel)

    def update_tool(self, tool_call_id: str, status: RowStatus) -> None:
        """Update the status of a tool call"""
        self.tool.update(tool_call_id, status)