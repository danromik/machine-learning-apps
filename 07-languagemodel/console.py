import queue
import select
import sys
import termios
import threading
import tty
from rich.console import Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

class Dashboard:
    def __init__(self, total_steps: int, title: str = "training"):
        self.progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )
        self.task = self.progress.add_task(title, total=total_steps)

        self.stats: dict[str, str] = {}
        self.losses: list[float] = []
        self.sample: str = ""
        self.status: str = ""

    def advance(self, step: int, loss: float) -> None:
        self.progress.update(self.task, completed=step)
        self.losses.append(loss)

    def _stats_table(self) -> Table:
        table = Table.grid(padding=(0, 3))
        for _ in range(2):
            table.add_column(justify="right", style="dim")
            table.add_column(justify="left", style="bold")

        items = list(self.stats.items())
        for i in range(0, len(items), 2):
            row: list[str] = []
            for key, value in items[i : i + 2]:
                row += [key, value]
            while len(row) < 4:
                row.append("")
            table.add_row(*row)

        return table

    def render(self) -> Group:
        pieces = [
            self.progress, 
            Panel(self._stats_table(), title="stats", border_style="dim"),
            Panel(
                Text(sparkline(self.losses), style="cyan"),
                title=f"train loss · {len(self.losses)} steps",
                border_style="dim",
            ),
        ]

        if self.sample:
            pieces.append(
                Panel(Text(self.sample, style="italic"), title="sample", border_style="dim")
            )

        pieces.append(Text(self.status, style="yellow"))
        return Group(*pieces)

class KeyReader:
    """Non-blocking single-keypress reader. Use as a context manager"""

    def __init__(self):
        self.keys: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._fd = sys.stdin.fileno()
        self._saved = None
        self._thread = None

    def __enter__(self) -> "KeyReader":
        if not sys.stdin.isatty():
            return self
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1)
                if ch:
                    self.keys.put(ch)

    def get(self) -> str | None:
        """Return the next pending keypress, or None if none is waiting"""
        try:
            return self.keys.get_nowait()
        except queue.Empty:
            return None

    

BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 60) -> str:
    """Render a sequence of floats as a one=line unicode bar chart"""
    if not values:
        return ""

    # downsample to at most `width` points by averaging buckets
    if len(values) > width:
        bucket = len(values) / width
        points = [
            sum(values[int(i * bucket) : int((i + 1) * bucket)])
            / max(1, int((i + 1) * bucket) - int(i * bucket))
            for i in range(width)
        ]
    else:
        points = list(values)

    lo, hi = min(points), max(points)
    span = hi - lo
    if span < 1e-9:
        return BLOCKS[0] * len(points)

    return "".join(BLOCKS[min(7, int((v - lo) / span * 8))] for v in points)

def format_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

