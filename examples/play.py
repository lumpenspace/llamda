from typing import List, Tuple, Dict, Any
import tty
import termios

from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from examples.handle_command_mode import handle_command_mode

from llamda_fn.functions import LlamdaBase
from examples.functions.simple_function_aq import aq
from llamda_fn.llamda import Llamda

console = Console()

import sys


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def handle_chat_mode(ll: Llamda) -> bool:
    console.print("[bold blue]🐱💬[/bold blue]", end="")
    user_input = input_with_getch()
    console.print()  # New line after input

    if user_input == ":":
        return True
    if user_input:
        ll.ask(user_input)
    return False


def input_with_getch() -> str:
    user_input = ""
    while True:
        char = getch()
        if char in ("\r", "\n", ":"):
            return user_input if char != ":" else ":"
        user_input += char
        console.print(char, end="")


def go():
    console.print(
        Panel(
            "[bold green]λλαμδα interactive shell![/bold green]\n"
            "Send a message or type : for command mode",
            title="Welcome",
            border_style="green",
        )
    )

    ll = Llamda(
        system="""You are a cabalistic assistant who is eager to help users
        find weird numerical correspondences between strings.
        """
    )

    def aq_multiple(input_strings: List[str]) -> List[Tuple[str, int]]:
        """
        Calculate the Alphanumeric Quabala (AQ) value for multiple strings.

        This function calculates the AQ value for each string in the input list
        and returns a sorted list of arrays containing the original string and its AQ value
        """
        return sorted([(s, aq(s)) for s in input_strings], key=lambda x: x[1])

    ll.fy()(aq_multiple)

    command_mode = False
    while True:
        if command_mode:
            command_mode = handle_command_mode(ll)
            if not command_mode:
                break
        else:
            command_mode = handle_chat_mode(ll)

    return ll


if __name__ == "__main__":
    go()
