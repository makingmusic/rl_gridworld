"""Rich-based TUI for the balance experiments.

Provides:
- Keyboard:               non-blocking stdin reader (cbreak + select), context manager.
- render_pendulum_ascii:  ASCII art of a pendulum at a given angle.
- build_view:             composes the Rich renderable shown each frame.

The simulation loop lives in the entry script (main_pendulum.py). This module
provides primitives so the same loop pattern can drive cartpole later.
"""

from __future__ import annotations

import math
import queue
import select
import sys
import termios
import threading
import tty
from dataclasses import dataclass

from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ── Keyboard ────────────────────────────────────────────────────────────────


class Keyboard:
    """Non-blocking single-key reader.

    Use as a context manager. While active, stdin is in cbreak mode and a
    background thread pushes each typed character onto a queue. Call drain()
    each tick to consume them. On exit, terminal state is restored.

    If stdin is not a TTY (e.g. piped input), this becomes a no-op and drain()
    always returns an empty list.
    """

    def __init__(self):
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._old_attrs = None
        self._thread: threading.Thread | None = None
        self._active = False

    def __enter__(self) -> "Keyboard":
        if not sys.stdin.isatty():
            return self
        self._old_attrs = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._active:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        if self._old_attrs is not None:
            termios.tcsetattr(
                sys.stdin.fileno(), termios.TCSADRAIN, self._old_attrs
            )
        self._active = False

    def _reader(self) -> None:
        fd = sys.stdin.fileno()
        while not self._stop.is_set():
            ready, _, _ = select.select([fd], [], [], 0.05)
            if not ready:
                continue
            try:
                ch = sys.stdin.read(1)
            except (OSError, ValueError):
                break
            if ch:
                self._queue.put(ch)

    def drain(self) -> list[str]:
        items: list[str] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items


# ── Pendulum ASCII ──────────────────────────────────────────────────────────

# Terminal cells are roughly twice as tall as wide. Multiplying horizontal
# coordinates by this factor keeps the rod's *visual* angle close to its
# physical angle.
_ASPECT = 2.0


def _segment_char(theta: float) -> str:
    """Pick a character that suggests the rod's slope at this angle.

    theta is measured from the upright (vertical), positive = tilted right.
    """
    a = abs(theta)
    if a < math.pi / 8:        # within ~22.5° of vertical
        return "|"
    if a < 3 * math.pi / 8:    # diagonal
        # Rod from pivot (bottom) to tip (top), tilted right: looks like '/'.
        return "/" if theta > 0 else "\\"
    return "_"


def render_pendulum_ascii(theta: float, width: int = 41, height: int = 13) -> Text:
    """Return a Rich Text block drawing the pendulum at angle theta (radians).

    The pivot sits at the bottom-center, the rod extends upward toward the tip.
    Ground is rendered as a row of '=' below the pivot.
    """
    # Make width odd so there's a single center column.
    if width % 2 == 0:
        width += 1
    grid = [[" "] * width for _ in range(height)]

    ground_row = height - 1
    pivot_row = height - 2
    pivot_col = width // 2

    for c in range(width):
        grid[ground_row][c] = "="
    grid[pivot_row][pivot_col] = "O"

    # Rod length in row-units. Leave a row of headroom so the tip never clips.
    rod_len = pivot_row - 1
    tip_row_f = pivot_row - rod_len * math.cos(theta)
    tip_col_f = pivot_col + _ASPECT * rod_len * math.sin(theta)

    char = _segment_char(theta)
    dr = tip_row_f - pivot_row
    dc = tip_col_f - pivot_col
    steps = max(int(round(max(abs(dr), abs(dc)))), 1)
    for i in range(1, steps + 1):
        t = i / steps
        r = int(round(pivot_row + dr * t))
        c = int(round(pivot_col + dc * t))
        if 0 <= r < ground_row and 0 <= c < width:
            grid[r][c] = char

    r_tip = int(round(tip_row_f))
    c_tip = int(round(tip_col_f))
    if 0 <= r_tip < ground_row and 0 <= c_tip < width:
        grid[r_tip][c_tip] = "●"

    text = Text()
    for row_idx, row in enumerate(grid):
        line = "".join(row)
        if row_idx == ground_row:
            text.append(line, style="dim")
        else:
            text.append(line)
        if row_idx != height - 1:
            text.append("\n")
    return text


# ── Telemetry rendering ─────────────────────────────────────────────────────


@dataclass
class Telemetry:
    """Snapshot of what the side panel should show for one frame."""

    title: str
    sim_time: float
    steps: int
    angle_deg: float
    rate_deg_s: float
    control: float
    control_label: str          # short label, e.g. "torque"
    control_units: str          # e.g. "Nm"
    setpoint_deg: float
    error_deg: float
    p_term: float
    i_term: float
    d_term: float
    status: str                 # one-line status (e.g. "balancing", "fell", "reset")

    # Bar bounds — used to scale the visual bars. All bars are centered at 0
    # and saturate at ±limit.
    max_control: float = 1.0    # actuator saturation magnitude (e.g. max torque)
    angle_limit_deg: float = 90.0
    max_rate_deg_s: float = 360.0

    # PID introspection: gains plus the intermediate quantities each term
    # multiplies. Used to show the formula breakdown.
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    error_rad: float = 0.0       # P's input
    integral_rad_s: float = 0.0  # I's input (rad · s)
    derivative_rad_s: float = 0.0  # D's input (rad / s)

    # Control-loop rate; physics advances by 1/tick_rate_hz seconds per tick.
    tick_rate_hz: float = 100.0

    # Recovery tracking (persisted across disturbances).
    last_recovery_ticks: int | None = None
    last_peak_deg: float | None = None
    recovery_state: str = "init"        # init | balanced | disturbed
    recovery_peak_deg: float = 0.0      # peak |angle| of in-progress disturbance


def _bar(value: float, limit: float, width: int = 12,
         pos_style: str = "green", neg_style: str = "red") -> Text:
    """Centered horizontal bar in [-limit, +limit], `width` cells wide.

    The midline '│' marks zero. Positive values fill rightward (green),
    negative fill leftward (red). The bar saturates at the limit.
    """
    if width % 2 == 1:
        width += 1
    half = width // 2
    if limit <= 0:
        v = 0.0
    else:
        v = max(-limit, min(limit, float(value)))
    cells = int(round(half * abs(v) / limit)) if limit > 0 else 0

    t = Text()
    if v >= 0:
        t.append("·" * half, style="dim")
        t.append("│", style="bold white")
        t.append("█" * cells, style=pos_style)
        t.append("·" * (half - cells), style="dim")
    else:
        t.append("·" * (half - cells), style="dim")
        t.append("█" * cells, style=neg_style)
        t.append("│", style="bold white")
        t.append("·" * half, style="dim")
    return t


_BAR_WIDTH = 12


def _state_table(t: Telemetry) -> Table:
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(justify="right", style="bold", no_wrap=True)
    tbl.add_column(width=_BAR_WIDTH + 1, no_wrap=True)   # bar
    tbl.add_column(justify="right", no_wrap=True)         # number

    tbl.add_row(
        "angle",
        _bar(t.angle_deg, t.angle_limit_deg),
        f"{t.angle_deg:+5.1f}°",
    )
    tbl.add_row(
        "rate",
        _bar(t.rate_deg_s, t.max_rate_deg_s),
        f"{t.rate_deg_s:+4.0f}°/s",
    )
    tbl.add_row("", Text("─" * _BAR_WIDTH, style="dim"), "")
    tbl.add_row(
        "target",
        Text("upright", style="dim"),
        f"{t.setpoint_deg:+5.1f}°",
    )
    return tbl


def _pid_table(t: Telemetry) -> Table:
    """Show the PID math being evaluated this tick.

    Each row: term symbol | bar | term value | formula evaluation.

    The formula column displays   gain · input = term   so the user can watch
    how each input quantity (error / integral / derivative) changes as the
    pendulum moves, and how multiplying it by the constant gain produces the
    term's contribution to the control signal.
    """
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(justify="right", style="bold", no_wrap=True)
    tbl.add_column(width=_BAR_WIDTH + 1, no_wrap=True)   # bar  — fixed width
    tbl.add_column(justify="right", width=7, no_wrap=True)  # term value — fixed
    tbl.add_column(style="dim", overflow="fold")          # formula — wraps if needed

    def math_row(label: str, value: float, formula: Text):
        tbl.add_row(label, _bar(value, t.max_control), f"{value:+5.2f}", formula)

    def fmt_formula(gain: float, input_value: float, input_label: str, input_unit: str) -> Text:
        # Render:  = 60 · (+0.147)  [error, rad]
        line = Text()
        line.append("= ", style="dim")
        line.append(f"{gain:g}", style="bold")
        line.append(" · ", style="dim")
        line.append(f"({input_value:+.3f})", style="cyan")
        line.append(f"  [{input_label}, {input_unit}]", style="dim")
        return line

    math_row("P", t.p_term, fmt_formula(t.kp, t.error_rad,       "error",        "rad"))
    math_row("I", t.i_term, fmt_formula(t.ki, t.integral_rad_s,  "∫error·dt",    "rad·s"))
    math_row("D", t.d_term, fmt_formula(t.kd, t.derivative_rad_s,"d(error)/dt",  "rad/s"))

    sep_bar = Text("─" * _BAR_WIDTH, style="dim")
    tbl.add_row("", sep_bar, Text("─────", style="dim"),
                Text(f"P + I + D, clamped to ±{t.max_control:g} {t.control_units}", style="dim"))

    math_row(
        f"{t.control_label} =", t.control,
        Text(f"applied {t.control_units}  (the actuator command)", style="dim"),
    )

    # Blank spacer then a "where did error come from?" line.
    tbl.add_row("", Text(""), "", "")
    err_formula = Text()
    err_formula.append("= setpoint − measurement = ", style="dim")
    err_formula.append(f"{t.setpoint_deg:+.1f}°", style="cyan")
    err_formula.append(" − ", style="dim")
    err_formula.append(f"{t.angle_deg:+.1f}°", style="cyan")
    tbl.add_row(
        "error",
        _bar(t.error_deg, t.angle_limit_deg, pos_style="yellow", neg_style="yellow"),
        f"{t.error_deg:+5.1f}°",
        err_formula,
    )
    return tbl


def _meta_table(t: Telemetry) -> Table:
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(justify="right", style="bold")
    tbl.add_column()
    tbl.add_row("time:",    f"{t.sim_time:6.2f} s")
    tbl.add_row("tick:",    f"{t.steps}")
    tbl.add_row("rate:",    f"{t.tick_rate_hz:g} Hz   (1 tick = {1.0/t.tick_rate_hz*1000:.0f} ms)")
    tbl.add_row("status:",  t.status)
    tbl.add_row("", "")

    # Recovery: last successful peak-to-balanced duration, persisted.
    if t.last_recovery_ticks is not None and t.last_peak_deg is not None:
        last_str = (
            f"{t.last_recovery_ticks} ticks  "
            f"({t.last_recovery_ticks / t.tick_rate_hz:.2f} s)  "
            f"from peak {t.last_peak_deg:+.1f}°"
        )
    else:
        last_str = "—  (not yet balanced)"
    tbl.add_row("last recovery:", last_str)

    # Current excursion (if any) so the user can watch it being measured.
    if t.recovery_state == "disturbed":
        tbl.add_row(
            "  in progress:",
            Text(f"tracking peak  (so far {t.recovery_peak_deg:+.1f}°)", style="cyan"),
        )
    return tbl


def _explanation_table() -> Table:
    """One-liner descriptions of what each PID term does, with the input it uses."""
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="bold magenta", no_wrap=True)
    tbl.add_column(no_wrap=True)
    tbl.add_column(style="dim")

    tbl.add_row(
        "P  proportional",
        Text("→ pushes back proportional to the CURRENT error.", style="white"),
        Text("Big P = aggressive; too much oscillates.", style="dim"),
    )
    tbl.add_row(
        "I  integral",
        Text("→ corrects DRIFT accumulated over time (∫ error · dt).", style="white"),
        Text("Removes steady-state offset; too much overshoots.", style="dim"),
    )
    tbl.add_row(
        "D  derivative",
        Text("→ damps motion based on HOW FAST error is changing.", style="white"),
        Text("Acts like friction; too much amplifies noise.", style="dim"),
    )
    return tbl


def build_view(canvas: Text, telemetry: Telemetry, hint: str) -> Group:
    """Compose the full TUI: animation, telemetry tables, PID legend, hint."""
    anim_panel = Panel(
        Align.center(canvas, vertical="middle"),
        title=telemetry.title,
        border_style="cyan",
        padding=(0, 1),
    )

    body = Table.grid(expand=True, padding=(0, 1))
    body.add_column(ratio=2)               # State + Run stacked
    body.add_column(ratio=3)               # PID, the dense one
    pid_title = (
        f"PID  —  Kp={telemetry.kp:g}   Ki={telemetry.ki:g}   Kd={telemetry.kd:g}"
        f"   (bar scale ±{telemetry.max_control:g} {telemetry.control_units})"
    )
    left_stack = Group(
        Panel(_state_table(telemetry), title="State", border_style="green"),
        Panel(_meta_table(telemetry),  title="Run",   border_style="yellow"),
    )
    body.add_row(
        left_stack,
        Panel(_pid_table(telemetry), title=pid_title, border_style="magenta"),
    )

    legend_panel = Panel(
        _explanation_table(),
        title="What is PID?",
        border_style="dim",
    )

    hint_panel = Panel(Align.center(Text(hint, style="dim")), border_style="dim")

    return Group(anim_panel, body, legend_panel, hint_panel)
