# Plan: `balance/` — Classical control on balancing problems

**Status:** Phase 1 complete (with several pedagogical extras added on top). Phase 2 (cartpole) not yet started.
**Confirmed:** 2026-05-11.

A new experiment subfolder for balancing problems (inverted pendulum → cartpole → maybe ball-on-beam), starting with a hand-written PID controller and a Rich-based TUI. No gym, no scipy, no new dependencies. Designed so an RL agent can plug into the same env interface later.

## Decisions (from intake questions)

| Question | Answer |
|---|---|
| Which problem? | Start with the simplest (inverted pendulum), then graduate to cartpole. |
| Long-term arc? | PID baseline now, RL agent later. Env interface mirrors `GridWorld2D`. |
| TUI scope? | Live ASCII animation + telemetry (no post-run plots in Phase 1). |
| Interaction? | Keyboard disturbances during the run. |

## Subfolder layout

```
balance/
├── README.md             # Mini intro + how to run
├── pendulum.py           # Inverted-pendulum env (Phase 1)
├── cartpole.py           # Cartpole env (Phase 2)
├── pid.py                # Reusable PID class
├── tui.py                # Rich animation + telemetry + keyboard input
├── main_pendulum.py      # Entry: pendulum + PID + TUI (Phase 1)
└── main_cartpole.py      # Entry: cartpole + PID + TUI (Phase 2)
```

Flat, mirrors the existing repo style. Config constants live at the top of each `main_*.py`.

## Env interface (RL drop-in later)

Both envs expose the same surface as `GridWorld2D`:

```python
env.reset(state=None) -> state
env.step(action) -> (next_state, reward, done, info)
```

- **State:** numpy array. Pendulum = `[theta, theta_dot]`. Cartpole = `[x, x_dot, theta, theta_dot]`.
- **Action:** scalar (torque for pendulum, force for cartpole). Continuous for PID; discretize later for DQN.
- **Reward:** shaped (angle² + small action penalty). Defined now, unused by PID.
- **`done`:** angle exceeds threshold, or time budget exhausted.
- **Integration:** semi-implicit Euler at `dt ≈ 0.01s`.

## PID controller

```python
class PID:
    def __init__(self, kp, ki, kd, setpoint=0.0, output_limits=(-F_max, F_max), integral_limits=...)
    def reset(self)
    def update(self, measurement, dt) -> (control, p_term, i_term, d_term)
```

Returns individual P/I/D contributions so the TUI can show what each term is doing.

- **Pendulum:** single PID on angle error (setpoint = 0 = upright).
- **Cartpole:** start with single PID on angle. Add an outer cart-position PID later if drift becomes annoying.

## TUI (Rich)

Single `Live` view, refreshing ~30 Hz, physics ticks faster.

```
─── Inverted Pendulum + PID ────────────────────
                    \
                     \
                      \●
                       \
                        \
                  ═══════╧═══════

  angle:      8.4°    ━━━━━━━──   setpoint: 0.0°
  rate:      -1.2°/s  ──━────
  torque:    +4.5 Nm  ━━━━───

  P: +6.7   I: -1.8   D: -0.4    error: 8.4°
  step: 412   time: 4.12s

  [a]/[d] nudge  [r] reset  [q] quit
─────────────────────────────────────────────────
```

## Keyboard input

Non-blocking stdin via `termios` cbreak + `select.select()`. No new deps.

- `a` / `d` — left/right impulse (avoiding arrow keys due to escape sequences)
- `r` — reset
- `q` — quit cleanly

## Phasing

**Phase 1 — pendulum end-to-end (DONE):**

1. ✅ `balance/pendulum.py` — env with `reset` / `step` / dynamics
2. ✅ `balance/pid.py` — PID class with P/I/D term reporting (also exposes integral and derivative)
3. ✅ `balance/tui.py` — Rich live view + non-blocking keyboard, bar widgets, formula display
4. ✅ `balance/main_pendulum.py` — wires it together, tuned default gains (Kp=60, Ki=5, Kd=10)
5. ✅ `balance/README.md`
6. ✅ Top-level `README.md` and `AGENTS.md` updated

**Phase 1.5 — pedagogical extras added during use (DONE):**

- ✅ Horizontal bar visualizations for P/I/D/torque/angle/rate (centered at 0, ±limit bounds)
- ✅ Formula breakdown shown live: `P = Kp · error`, `I = Ki · ∫error·dt`, `D = Kd · d(error)/dt`, with the input value highlighted in cyan
- ✅ Wall-clock time scaling (`-` / `+` / `1` keys), default 0.25× so the user can read the bars
- ✅ Pause (`space`) and single-tick step (`.` / `n`) for tick-by-tick observation
- ✅ Tick-rate display (Hz + ms) in the Run panel, explaining the 100 Hz control-loop frequency
- ✅ `RecoveryTracker` (`balance/recovery.py`): measures ticks from peak deflection back to balanced, persists the last result on screen
- ✅ "What is PID?" legend panel always visible

**Phase 2 — cartpole (NOT STARTED):**

7. `balance/cartpole.py`
8. `balance/main_cartpole.py`
9. Update `balance/README.md`

**Phase 3 — RL agent (separate session):** plug a DQN or policy-gradient agent into the same env interface and compare against the PID baseline.

## Defaults

- **Pendulum:** mass 1 kg, length 1 m, gravity 9.81, damping 0.1, dt 0.01s, max torque ~20 Nm.
- **PID:** tuned during Phase 1 verification. Starting point ~ `Kp=30, Ki=0, Kd=5`.

## Out of scope

- gym / gymnasium / scipy / new dependencies
- Tests, LQR, multi-controller comparison
- Matplotlib post-run plots (easy to add later if wanted)
