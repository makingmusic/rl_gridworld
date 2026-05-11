"""PID controller.

A standard PID with output clamping and integral anti-windup. update() returns
the individual P/I/D contributions in addition to the final control signal so
the TUI can show which term is doing the work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PIDOutput:
    control: float       # final, clamped control signal
    p_term: float        # Kp · error
    i_term: float        # Ki · integral
    d_term: float        # Kd · derivative
    error: float         # setpoint − measurement
    integral: float      # accumulated ∫ error · dt (after anti-windup clamping)
    derivative: float    # d(error)/dt this tick


class PID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        setpoint: float = 0.0,
        output_limits: tuple[float, float] = (-math.inf, math.inf),
        integral_limits: tuple[float, float] | None = None,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.out_lo, self.out_hi = output_limits
        if integral_limits is None:
            self.i_lo, self.i_hi = -math.inf, math.inf
        else:
            self.i_lo, self.i_hi = integral_limits
        self._integral = 0.0
        self._prev_error: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def update(self, measurement: float, dt: float) -> PIDOutput:
        error = self.setpoint - measurement

        p_term = self.kp * error

        self._integral += error * dt
        # Anti-windup: clamp the accumulated integral.
        self._integral = max(self.i_lo, min(self.i_hi, self._integral))
        i_term = self.ki * self._integral

        if self._prev_error is None or dt <= 0.0:
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / dt
        d_term = self.kd * derivative
        self._prev_error = error

        raw = p_term + i_term + d_term
        clamped = max(self.out_lo, min(self.out_hi, raw))

        # If output saturated and integral is pushing further into saturation,
        # back the integral off (conditional integration) to prevent windup.
        if clamped != raw and ((raw > self.out_hi and error > 0) or (raw < self.out_lo and error < 0)):
            self._integral -= error * dt
            self._integral = max(self.i_lo, min(self.i_hi, self._integral))
            i_term = self.ki * self._integral

        return PIDOutput(
            control=clamped,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
            error=error,
            integral=self._integral,
            derivative=derivative,
        )
