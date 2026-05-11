"""Pendulum + PID + TUI.

Run with:
    UV_PROJECT_ENVIRONMENT=myenv uv run python balance/main_pendulum.py

Controls:
    a / d   apply a left / right impulse to the pendulum (mid-run disturbance)
    r       reset to the initial angle
    q       quit
"""

from __future__ import annotations

import math
import time

from rich.console import Console
from rich.live import Live

from pendulum import Pendulum, PendulumParams
from pid import PID
from recovery import RecoveryTracker
from tui import Keyboard, Telemetry, build_view, render_pendulum_ascii


# ── Configuration ───────────────────────────────────────────────────────────

# Physics
PARAMS = PendulumParams(
    mass=1.0,
    length=1.0,
    gravity=9.81,
    damping=0.1,
    dt=0.01,               # physics step (s) — 100 Hz
    max_torque=20.0,       # Nm
    angle_limit=math.radians(75.0),
    max_time=10_000.0,     # effectively never time out in interactive mode
)

# Initial condition
INITIAL_ANGLE_DEG = 5.0
INITIAL_RATE_DEG_S = 0.0

# PID gains — tuned for the parameters above.
PID_KP = 60.0
PID_KI = 5.0
PID_KD = 10.0

# Keyboard nudge: how much angular velocity to inject when a/d is pressed.
NUDGE_RATE_DEG_S = 90.0

# Rendering
ANIMATION_WIDTH = 51
ANIMATION_HEIGHT = 13
TARGET_FPS = 30.0
TITLE = "Inverted Pendulum + PID"
HINT = (
    "[a]/[d] nudge   [space] pause   [.]/[n] step one tick   "
    "[-]/[+] slower/faster   [1] real-time   [r] reset   [q] quit"
)

# Wall-clock pacing. 1.0 = real-time. 0.25 = quarter speed (1 sim-sec = 4 wall-sec).
# Press '-' / '+' at runtime to change. Physics dt is NOT affected — only the
# sleep between ticks — so the dynamics stay correct.
INITIAL_TIME_SCALE = 0.25
TIME_SCALE_MIN = 0.05
TIME_SCALE_MAX = 4.0


def _initial_state() -> list[float]:
    return [math.radians(INITIAL_ANGLE_DEG), math.radians(INITIAL_RATE_DEG_S)]


def _telemetry(env: Pendulum, ctrl_out, status: str, time_scale: float,
               paused: bool, recovery: RecoveryTracker) -> Telemetry:
    theta, theta_dot = env.state
    speed_str = f"speed {time_scale:.2f}x" if not paused else "PAUSED"
    return Telemetry(
        title=TITLE,
        sim_time=env.time,
        steps=env.steps,
        angle_deg=math.degrees(theta),
        rate_deg_s=math.degrees(theta_dot),
        control=ctrl_out.control,
        control_label="torque",
        control_units="Nm",
        setpoint_deg=0.0,
        error_deg=math.degrees(ctrl_out.error),
        p_term=ctrl_out.p_term,
        i_term=ctrl_out.i_term,
        d_term=ctrl_out.d_term,
        status=f"{status}   |   {speed_str}",
        max_control=PARAMS.max_torque,
        angle_limit_deg=math.degrees(PARAMS.angle_limit),
        max_rate_deg_s=360.0,
        kp=PID_KP,
        ki=PID_KI,
        kd=PID_KD,
        error_rad=ctrl_out.error,
        integral_rad_s=ctrl_out.integral,
        derivative_rad_s=ctrl_out.derivative,
        tick_rate_hz=1.0 / PARAMS.dt,
        last_recovery_ticks=recovery.last_recovery_ticks,
        last_peak_deg=recovery.last_peak_deg,
        recovery_state=recovery.state,
        recovery_peak_deg=recovery.current_peak_deg,
    )


def main() -> None:
    env = Pendulum(PARAMS)
    pid = PID(
        kp=PID_KP, ki=PID_KI, kd=PID_KD,
        setpoint=0.0,
        output_limits=(-PARAMS.max_torque, PARAMS.max_torque),
        integral_limits=(-PARAMS.max_torque, PARAMS.max_torque),
    )

    env.reset(_initial_state())
    pid.reset()
    recovery = RecoveryTracker()
    status = "balancing"
    time_scale = INITIAL_TIME_SCALE
    paused = False
    step_once = False  # one-shot: advance exactly one tick

    console = Console()
    frame_period = 1.0 / TARGET_FPS

    # Render an initial frame so Live has something to show immediately.
    initial_ctrl = pid.update(env.state[0], PARAMS.dt)
    pid.reset()  # don't let that probe pollute the controller state
    view = build_view(
        render_pendulum_ascii(env.state[0], ANIMATION_WIDTH, ANIMATION_HEIGHT),
        _telemetry(env, initial_ctrl, status, time_scale, paused, recovery),
        HINT,
    )

    with Keyboard() as kb, Live(view, console=console, refresh_per_second=TARGET_FPS, screen=True) as live:
        last_render = time.monotonic()
        running = True
        ctrl_out = initial_ctrl
        while running:
            tick_start = time.monotonic()

            for ch in kb.drain():
                if ch == "q":
                    running = False
                elif ch == "a":
                    env.apply_impulse(-math.radians(NUDGE_RATE_DEG_S))
                    status = "nudged left"
                elif ch == "d":
                    env.apply_impulse(+math.radians(NUDGE_RATE_DEG_S))
                    status = "nudged right"
                elif ch == "r":
                    env.reset(_initial_state())
                    pid.reset()
                    recovery.reset_episode()
                    status = "reset"
                elif ch == " ":
                    paused = not paused
                elif ch in (".", "n"):
                    # Advance exactly one tick on the next physics update.
                    # Implies paused so subsequent ticks don't auto-advance.
                    paused = True
                    step_once = True
                elif ch in ("-", "_", ","):
                    time_scale = max(TIME_SCALE_MIN, time_scale / 2.0)
                elif ch in ("+", "="):
                    time_scale = min(TIME_SCALE_MAX, time_scale * 2.0)
                elif ch == "1":
                    time_scale = 1.0

            if not running:
                break

            # Advance physics only when running normally, or when the user
            # explicitly asked for one tick.
            advance = (not paused) or step_once
            if advance:
                ctrl_out = pid.update(env.state[0], PARAMS.dt)
                _, _, done, info = env.step(ctrl_out.control)
                step_once = False

                if done:
                    if info.get("fell"):
                        status = "fell — auto-reset"
                    else:
                        status = "timed out — auto-reset"
                    env.reset(_initial_state())
                    pid.reset()
                    recovery.reset_episode()
                else:
                    recovery.update(math.degrees(env.state[0]), env.steps)
                    if status in ("nudged left", "nudged right", "reset"):
                        # Revert to "balancing" once the pendulum is close to upright again.
                        if abs(env.state[0]) < math.radians(2.0):
                            status = "balancing"

            now = time.monotonic()
            if now - last_render >= frame_period:
                live.update(
                    build_view(
                        render_pendulum_ascii(env.state[0], ANIMATION_WIDTH, ANIMATION_HEIGHT),
                        _telemetry(env, ctrl_out, status, time_scale, paused, recovery),
                        HINT,
                    )
                )
                last_render = now

            # Pace the loop. Physics dt is fixed (correct dynamics); we just
            # sleep longer to slow the wall-clock playback. When paused, idle
            # at ~30 Hz so keyboard input stays responsive without burning CPU.
            elapsed = time.monotonic() - tick_start
            if paused:
                sleep_for = frame_period - elapsed
            else:
                sleep_for = (PARAMS.dt / time_scale) - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    main()
