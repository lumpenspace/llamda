from examples.play import console, input_with_getch
from llamda_fn.llamda import Llamda
from llamda_fn.llogos.llogos import Llogos


def handle_command_mode(ll: Llamda) -> bool:
    console.print("[bold red]🐱🖥️![/bold red]", end="")
    user_input = input_with_getch()
    console.print()  # New line after input

    if user_input.lower() == "q":
        console.print("\n[bold red]Exiting the interactive session. Goodbye![/bold red]")
        return False
    elif user_input.lower() == "t":
        console.print(create_tools_table(ll.tools.tools))
    elif user_input.lower() == "l":
        console.print([Llogos.msg(message) for message in ll.exchange])
    elif user_input == ":":
        return False
    else:
        console.print("\n[bold yellow]Unknown command.[/bold yellow]")
    return True