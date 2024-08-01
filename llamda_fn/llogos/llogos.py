"""
Mixin class to add llogoi functionality to a class, implemented as a Singleton.
"""

from typing import Any
from rich.console import Console
import inspect


class Llogos(Console):
    """
    Llogos is a singleton class that extends the Console class from the rich library.
    It provides a simple interface for logging messages to the console.
    """

    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Llogos":
        if Llogos._instance and Llogos._instance._initialized:
            return Llogos._instance

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False  # Initialize here

        from .message_logger import MessageLogger

        cls.msg = MessageLogger(console=cls._instance)
        return cls._instance

    def __init__(self, *args: Any, **kwargs: Any) -> None:

        if Llogos._instance and Llogos._instance._initialized:
            return

        super().__init__(*args, **kwargs)

        self._initialized = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.console, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        print("caller name:", calframe[1][3])

        return self.log(*args, **kwargs)


def get_llogos_instance(*args: Any, **kwargs: Any) -> Llogos:
    instance = Llogos(*args, **kwargs)
    return instance


# Export the initialized instance
llogos = get_llogos_instance()
