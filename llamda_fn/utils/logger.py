"""
This module contains the console utilities for the penger package.
"""

from typing import Any
from rich.console import Console
from rich.live import Live
from rich.json import JSON

console = Console()
error_console = Console(stderr=True)


def live(shell: Console) -> Live:
    """
    Create a live console.
    """
    return Live(console=shell)


emojis = {
    "user": "🙎",
    "assistant": "🤖",
    "tool": "🔧",
    "system": "👽",
}


def log_message(role: str, message: Any, tool_call: bool = False) -> None:
    console.log(f"{emojis[role]}")
    console.log(JSON.from_data(message))


__all__ = ["console", "error_console", "live"]
