"""Represents a row in the table"""

from typing import Generic, Literal, TypeVar

T = TypeVar("T")

RowStatus = Literal["Pending", "Success", "Error"]


class Row(Generic[T]):
    """Represents a row in the table"""

    col_names: list[str]

    def __init__(
        self,
        item: T,
        col_names: list[tuple[str, str]] | None = None,
        status: RowStatus = "Pending",
    ) -> None:
        if col_names is None:
            col_names = [(key, key) for key in item.__dict__.keys()]
        self.col_names: list[str] = [name for name, _ in col_names]
        self.status: RowStatus = status
        self.col_values: list[str] = [""] * len(col_names)
        self.item: T = item
