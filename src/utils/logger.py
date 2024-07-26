"""
This module contains the console utilities for the penger package.
"""

from rich.console import Console
from rich.live import Live

console = Console()
error_console = Console(stderr=True)


def live(shell: Console) -> Live:
    """
    Create a live console.
    """
    return Live(console=shell)


__all__ = ["console", "error_console", "live"]
