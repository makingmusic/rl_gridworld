# balance/ — Classical control on balancing problems

A sandbox for balancing problems (inverted pendulum, cartpole, possibly ball-on-beam) solved with hand-written controllers. No gym, no scipy, no new dependencies — just numpy, Rich, and math.

Designed so an RL agent can drop into the same env interface later. See [../plans/balance.md](../plans/balance.md) for the full plan and phasing.

## Phase 1: Inverted pendulum + PID (here now)

A point-mass pendulum on a fixed pivot, actuated by torque at the pivot. State is `[theta, theta_dot]` measured from the upright. A single-loop PID drives the angle to zero.

### Run

```bash
UV_PROJECT_ENVIRONMENT=myenv uv run python balance/main_pendulum.py
```

Or, after `./run.sh`, `uv run python balance/main_pendulum.py` works because the project venv is already on the path.

### Controls

| Key | Action |
|---|---|
| `a` | Apply a leftward angular-velocity impulse (disturbance) |
| `d` | Apply a rightward angular-velocity impulse |
| `space` | Pause / resume |
| `.` / `n` | Advance exactly one tick (auto-pauses) |
| `-` / `,` | Slow down (halve playback speed) |
| `+` / `=` | Speed up (double playback speed) |
| `1` | Reset playback speed to real-time (1.0x) |
| `r` | Reset to the initial angle |
| `q` | Quit |

The pendulum auto-resets if it falls past the angle limit.

The simulation starts at `0.25x` speed (1 simulated second = 4 wall-clock seconds) so you can watch the P/I/D terms react. Physics is unaffected — only wall-clock pacing changes. Current speed (or `PAUSED`) is shown in the Run panel.

### What is a "tick"?

A **tick** is one iteration of the control loop: one call to `pid.update(...)` followed by one call to `env.step(...)`. Each tick advances simulated time by `dt = 0.01 s` (10 ms), so the loop runs at **100 Hz** by default.

The tick rate is a core property of any control system. The faster the loop, the sooner the controller can detect a disturbance and respond. Real-world examples:

- Industrial PLCs: 10 Hz – 1 kHz
- Drone flight controllers: 1 – 8 kHz
- Hard-disk head servos: 10+ kHz

Use `space` + `.` to step the simulation one tick at a time. Watch what changes in a single tick:

- **error** = setpoint − measurement (changes because the pendulum moved during the previous tick)
- **integral** grows by `error · dt`
- **derivative** = `(error_now − error_prev) / dt`
- **P**, **I**, **D** are recomputed from the gains and these inputs
- A new torque is applied for the next tick

If you slow the loop down (raise `PARAMS.dt`, e.g. to 0.1 → 10 Hz), the pendulum will start to oscillate and may fall — the controller can't react fast enough. Try it.

### What you'll see

Five panels:

- **Animation** — ASCII pendulum, tip marked `●`, ground line below the pivot.
- **State** — angle and angular rate, each shown as a horizontal bar centered at 0. The bar saturates at the fall-limit (angle) or ±360°/s (rate). At a glance you see direction and how close to the limit.
- **PID** — each of P / I / D as a horizontal bar centered at 0, with `±max_torque` as the bar bounds. Filled green right of center = positive contribution, red left = negative. The sum row shows the total applied torque (also clamped to `±max_torque`). The error row shows `setpoint − angle` in degrees.
- **Run** — elapsed sim time, current tick, tick rate (Hz), status. Plus **recovery stats**: `last recovery: N ticks (T s) from peak X°` — the time it took to return to balanced after the most recent disturbance (the very first balancing from the initial tilt counts too). When a disturbance is in progress, an `in progress: tracking peak (so far ±X°)` line shows what the tracker is measuring.
- **What is PID?** — one-line explanation of each term, always visible.

The bars are inherently quieter than digits: they only change when a value crosses a cell threshold (~8% of full scale). The numbers next to them give precise readout.

### Recovery time

A "recovery" is measured as the number of ticks between the peak angle of a disturbance and the moment the pendulum settles back inside the balanced threshold (`< 2°`) and stays there for a grace period (50 ticks ≈ 0.5 s).

- Faster recovery (fewer ticks) = better controller tuning.
- The last completed recovery stays on screen until the next one replaces it.
- Manual or auto resets clear the in-progress measurement but **do not** clear the last recorded one.

This is a quick proxy for controller quality. Try changing `PID_KP`, `PID_KI`, `PID_KD` (top of `main_pendulum.py`) and re-running to see how the recovery count changes.

### What P / I / D mean — and what you'll see in the panel

The panel header shows the constant gains: `Kp Ki Kd`. Each row shows the term being computed *right now*:

```
P  bar  +8.80  = 60 · (+0.147)  [error, rad]
I  bar  +0.12  =  5 · (+0.025)  [∫error·dt, rad·s]
D  bar  -4.00  = 10 · (-0.400)  [d(error)/dt, rad/s]
```

Read each row as **`term = gain · input`**, where:

- **P (proportional)** — `Kp · error`. The input is the current error in radians (angle from upright). If you're tilted 10° (0.175 rad) right, P pushes left with `Kp × 0.175` Nm. Watch the error value change as the pendulum tilts. Big Kp = aggressive correction, but too much causes oscillation.
- **I (integral)** — `Ki · ∫error·dt`. The input is the *accumulated* error over time (units: rad·s). Even when the angle is tiny, if it's been off-center for a while, the integral grows and pushes I to cancel that drift. Watch the integral value drift slowly. Too much Ki causes overshoot and slow oscillation.
- **D (derivative)** — `Kd · d(error)/dt`. The input is *how fast the error is changing* (rad/s). If the pendulum is falling fast in one direction, D pushes hard the other way to brake it — like air resistance. Watch this value spike when you press `a` or `d` to nudge. Too much Kd amplifies sensor noise.

The total applied torque is **P + I + D**, clamped to ±max_torque. The bottom row of the panel decomposes the error itself: `error = setpoint − measurement`, so you see where the input to P came from.

#### Why radians?

The PID controller doesn't care about units — it's a generic algorithm. Internally it works in radians because that's what the pendulum environment uses (most physics math is cleaner in radians). The State panel shows the same angle in degrees because that's what humans read more easily. `8.4° = 0.147 rad`. Conversion: `1 rad = 180/π ≈ 57.3°`.

#### Experimenting with the gains

Edit `PID_KP`, `PID_KI`, `PID_KD` at the top of `main_pendulum.py` and re-run. Things to try:

- Set `PID_KI = 0`. The pendulum should still stabilize from a tilt, but if you nudge it gently it may park at a small permanent offset (no integral term to cancel residual error).
- Set `PID_KD = 0`. The pendulum will overshoot and oscillate visibly before settling — no damping.
- Set `PID_KP = 100`. Watch it overcorrect and oscillate. Lower it back down.
- Halve all three gains. The response gets slower and laggier.

## File map

| File | What it does |
|---|---|
| `pendulum.py` | `Pendulum` env: `reset()` / `step(torque)` / dynamics |
| `pid.py` | `PID` class with output clamping, integral anti-windup, and per-term reporting |
| `recovery.py` | `RecoveryTracker` — measures peak-to-balanced tick count after each disturbance |
| `tui.py` | Rich `Live` view, ASCII renderer, bar widgets, non-blocking keyboard reader |
| `main_pendulum.py` | Config + main loop. All hyperparameters live at the top of this file |
| `hardware.md` | Notes on building a physical PID demo (ball-on-beam) with parts available in Singapore |
| `cartpole.py` | (Phase 2 — not yet) cartpole env |
| `main_cartpole.py` | (Phase 2 — not yet) cartpole entry script |

## Tunables (top of `main_pendulum.py`)

| Constant | Default | Notes |
|---|---|---|
| `PARAMS.mass` / `.length` | 1.0 kg / 1.0 m | Standard pendulum |
| `PARAMS.gravity` | 9.81 | Earth |
| `PARAMS.damping` | 0.1 | Viscous friction at the pivot |
| `PARAMS.dt` | 0.01 s | Physics step (100 Hz) |
| `PARAMS.max_torque` | 20 Nm | Actuator saturation |
| `PARAMS.angle_limit` | 75° | Episode ends past this; auto-resets |
| `PID_KP` / `KI` / `KD` | 60 / 5 / 10 | Tuned for the above. Recovers from a 90°/s nudge in ~74 ticks with <6° peak excursion. |
| `NUDGE_RATE_DEG_S` | 90 | How hard `a`/`d` kicks the pendulum |
| `INITIAL_TIME_SCALE` | 0.25 | Wall-clock playback speed at startup (physics unchanged) |
| `TIME_SCALE_MIN` / `MAX` | 0.05 / 4.0 | Bounds for the `-` / `+` keys |
| `TARGET_FPS` | 30 | TUI refresh rate (physics ticks faster than this) |

## Tuning notes

Starting from upright (5°, zero rate), the PID stabilizes in roughly 1.5 s. Under a 90°/s impulse it recovers in ~2 s with peak angle around 5–6°.

If you change pendulum parameters (mass, length, gravity), the gains will need re-tuning:

- Stiffer rod (heavier/longer) → larger `Kp`.
- More damping → smaller `Kd`.
- If steady-state error doesn't decay, raise `Ki`.

## Roadmap

- **Phase 2** — cartpole env + PID. Same TUI primitives.
- **Phase 3** — drop in a DQN or policy-gradient agent against the same `step()`/`reset()` interface, compare against the PID baseline.
