from abc import ABC, abstractmethod
from rich.console import Console

from llamda_fn.llms.ll_message import LLMessage
from llamda_fn.llogos.table_logger import RowStatus


class IMessageLogger(ABC):
    """Interface for a message logger"""

    def __init__(self, *, console: Console):
        self.console = console

    @abstractmethod
    def __call__(self, message: LLMessage) -> None:
        pass

    @abstractmethod
    def update_tool(self, tool_call_id: str, status: RowStatus) -> None:
        """Update the status of a tool call"""
        pass