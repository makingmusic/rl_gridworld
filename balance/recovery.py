"""Recovery-time tracker.

Measures how many ticks the controller takes to return the pendulum from the
peak of a disturbance back to "balanced", where "balanced" means the angle
stayed within a small threshold for a grace period.

State machine:
- init:       just started; waiting to settle into balanced for the first time.
- balanced:   |angle| has stayed below threshold for >= grace_ticks.
- disturbed:  |angle| crossed the threshold (a nudge or initial tilt). We track
              the peak |angle| while in this state. When |angle| comes back
              under the threshold and stays there for grace_ticks, we record
              the recovery (ticks since the peak) and return to balanced.

Manual or auto resets abandon the in-progress recovery but keep the last
recorded result so it stays visible on screen.
"""

from __future__ import annotations


class RecoveryTracker:
    def __init__(self, balanced_threshold_deg: float = 2.0, grace_ticks: int = 50):
        self.threshold_deg = balanced_threshold_deg
        self.grace_ticks = grace_ticks

        # Persisted across disturbances; only updated on a successful recovery.
        self.last_recovery_ticks: int | None = None
        self.last_peak_deg: float | None = None

        self._state = "init"          # init | balanced | disturbed
        self._peak_deg = 0.0
        self._peak_tick = 0
        self._balanced_streak = 0     # consecutive ticks below threshold

    @property
    def state(self) -> str:
        return self._state

    @property
    def current_peak_deg(self) -> float:
        """Peak |angle| of the disturbance currently being tracked (0 if balanced)."""
        return self._peak_deg

    def update(self, angle_deg: float, tick: int) -> None:
        within = abs(angle_deg) < self.threshold_deg

        if self._state == "balanced":
            if not within:
                # Disturbance just started.
                self._state = "disturbed"
                self._peak_deg = angle_deg
                self._peak_tick = tick
                self._balanced_streak = 0

        elif self._state == "disturbed":
            if abs(angle_deg) > abs(self._peak_deg):
                self._peak_deg = angle_deg
                self._peak_tick = tick
            if within:
                self._balanced_streak += 1
                if self._balanced_streak >= self.grace_ticks:
                    # Recovery confirmed. The recovery time we report is the
                    # number of ticks from the peak to NOW (the moment the
                    # grace period completed).
                    self.last_recovery_ticks = tick - self._peak_tick
                    self.last_peak_deg = self._peak_deg
                    self._state = "balanced"
                    self._peak_deg = 0.0
            else:
                self._balanced_streak = 0

        else:  # init
            if within:
                self._balanced_streak += 1
                if self._balanced_streak >= self.grace_ticks:
                    self._state = "balanced"
            else:
                # Initial tilt counts as a disturbance — track the peak.
                self._state = "disturbed"
                if abs(angle_deg) > abs(self._peak_deg):
                    self._peak_deg = angle_deg
                    self._peak_tick = tick
                self._balanced_streak = 0

    def reset_episode(self) -> None:
        """Called on manual or auto reset. Last recorded recovery is kept."""
        self._state = "init"
        self._peak_deg = 0.0
        self._peak_tick = 0
        self._balanced_streak = 0
