from types import TracebackType
from typing import Any, Generic, List, Tuple, TypeVar

from rich.live import Live
from rich.table import Table
from rich.console import Console

from .row import Row, RowStatus

T = TypeVar("T")


class LiveTableLogger(Generic[T]):
    def __init__(
        self,
        console: Console,
        title: str,
        cols: List[Tuple[str, str]],
        **kwargs: Any,
    ) -> None:
        self.console = console
        self.title: str = title
        self.cols: List[Tuple[str, str]] = cols
        self.rows: List[Row[T]] = []
        self.table: Table = Table(title=self.title, **kwargs)
        self.live: Live = Live(self.table, console=self.console, refresh_per_second=4)

    def add_row(self, item: T) -> None:
        row = Row(item)
        self.rows.append(row)
        self._update_table()

    def update_row_status(self, index: int, status: RowStatus) -> None:
        if 0 <= index < len(self.rows):
            self.rows[index].status = status
            self._update_table()

    def _update_table(self) -> None:
        self.table.columns.clear()
        if not self.rows:
            return

        # Add columns based on the first row
        for _, col_name in self.cols:
            self.table.add_column(col_name)
        self.table.add_column("Status")

        # Add rows
        for row in self.rows:
            row_values: list[str] = row.col_values + [row.status]
            self.table.add_row(*row_values)

    def __enter__(self):
        self.live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.live.__exit__(exc_type, exc_val, exc_tb)

    def update_table(self) -> None:
        self._update_table()
        self.live.update(self.table)