"""Logs tool calls in a live table"""

from typing import Sequence, Optional
from rich.console import Console

from llamda_fn.llms.ll_message import LLToolCall
from llamda_fn.llogos.table_logger import LiveTableLogger, RowStatus


class ToolCallsLogger:
    def __init__(self, console: Console) -> None:
        self.console: Console = console
        self.table_logger: Optional[LiveTableLogger[LLToolCall]] = None

    def init(self, tool_calls: Sequence[LLToolCall]) -> None:
        """Initialize the logger with tool calls"""
        self.table_logger = LiveTableLogger(
            self.console,
            title="Tool Calls",
            cols=[("Name", "name"), ("Arguments", "arguments")],
        )
        for tool_call in tool_calls:
            self.table_logger.add_row(tool_call)

    def update(self, tool_call_id: str, status: RowStatus) -> None:
        if self.table_logger is None:
            return
        for index, row in enumerate(self.table_logger.rows):
            if row.item.tool_call_id == tool_call_id:
                self.table_logger.update_row_status(index, status)
                break
