"""Inverted pendulum environment.

Single rigid pendulum on a fixed pivot, actuated by a torque at the pivot.
State convention: theta is the angle measured from the upright (vertical)
position, positive counter-clockwise. theta=0 means balanced upright.

The interface mirrors GridWorld2D so an RL agent can later replace the PID
controller without changing the surrounding loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PendulumParams:
    mass: float = 1.0          # kg
    length: float = 1.0        # m  (distance from pivot to point mass)
    gravity: float = 9.81      # m/s^2
    damping: float = 0.1       # viscous friction at the pivot
    dt: float = 0.01           # integration step (s)
    max_torque: float = 20.0   # Nm, applied torque is clipped to [-max, +max]
    angle_limit: float = math.radians(60.0)  # episode ends if |theta| exceeds this
    max_time: float = 30.0     # episode time budget (s)


class Pendulum:
    def __init__(self, params: PendulumParams | None = None):
        self.params = params if params is not None else PendulumParams()
        self.state = np.zeros(2, dtype=np.float64)
        self.time = 0.0
        self.steps = 0

    def reset(self, state: np.ndarray | tuple[float, float] | None = None) -> np.ndarray:
        if state is None:
            self.state = np.array([math.radians(5.0), 0.0], dtype=np.float64)
        else:
            self.state = np.asarray(state, dtype=np.float64).copy()
        self.time = 0.0
        self.steps = 0
        return self.state.copy()

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        p = self.params
        torque = float(np.clip(action, -p.max_torque, p.max_torque))

        theta, theta_dot = self.state

        # Equation of motion for a point-mass pendulum, with the convention
        # that theta is measured from the upright. Gravity acts to grow |theta|.
        #   I * theta_ddot = m*g*L*sin(theta) - b*theta_dot + tau
        # with I = m*L^2.
        inertia = p.mass * p.length * p.length
        theta_ddot = (
            p.mass * p.gravity * p.length * math.sin(theta)
            - p.damping * theta_dot
            + torque
        ) / inertia

        # Semi-implicit Euler: update velocity first, then position with it.
        theta_dot = theta_dot + theta_ddot * p.dt
        theta = theta + theta_dot * p.dt

        self.state = np.array([theta, theta_dot], dtype=np.float64)
        self.time += p.dt
        self.steps += 1

        fell = abs(theta) > p.angle_limit
        timed_out = self.time >= p.max_time
        done = fell or timed_out

        reward = -(theta * theta) - 0.001 * (torque * torque)

        info = {
            "torque": torque,
            "fell": fell,
            "timed_out": timed_out,
            "time": self.time,
        }
        return self.state.copy(), reward, done, info

    def apply_impulse(self, delta_theta_dot: float) -> None:
        """External disturbance: instantly change angular velocity by delta."""
        self.state[1] += float(delta_theta_dot)
