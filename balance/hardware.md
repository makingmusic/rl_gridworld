# Hardware: build a physical PID demo

A companion to the simulation in this folder. Once you've felt how P / I / D behave in the TUI, the next step is wiring up the same loop to real hardware so the physics happens in the real world instead of being integrated by `pendulum.py`.

This doc covers what to build for a first physical PID demo, given the constraints of being in Singapore with an Arduino / ESP / Raspberry Pi already on hand and access to Sim Lim Square + a home 3D printer.

## Pick: ball-on-beam (1D)

A short beam pivots in the middle (or near one end). A servo tilts the beam. A ball rolls along a groove on the beam. A distance sensor at one end tells you where the ball is. The PID adjusts the tilt angle to keep the ball at a target position. You poke the ball with your finger as the disturbance and watch the controller fight back.

It's the same feel as the pendulum sim, but the unstable axis is "ball wants to roll off" instead of "pendulum wants to fall."

### Why this over the alternatives

- **Ball-in-tube (fan version):** needs a clear acrylic tube (annoying to source — hardware stores at Bras Basah have them but you'll be hunting), a 12 V PC fan with PWM, a MOSFET to drive it, and a 12 V supply. More electrical complexity for the same lesson.
- **Self-balancing two-wheel robot:** needs an MPU6050 + sensor fusion + two motor drivers + wheels + chassis tuning. You'll spend a weekend on hardware before you ever tune a PID.
- **Maglev:** electromagnets, hall sensor, and a ~kHz control loop. Cool, but the timing is unforgiving for a first build.

## Bill of materials (~S$15–20)

| Part | Where | Approx S$ |
|---|---|---|
| SG90 micro-servo (or MG90S for metal gears / less slop) | Sim Lim — hobby electronics shops on L3 / L4 | 3–6 |
| HC-SR04 ultrasonic distance sensor (or VL53L0X laser ToF — see gotchas) | Sim Lim, same shops | 2–8 |
| Ping-pong ball | Decathlon / any sports shop | 1 |
| Jumper wires + breadboard | Probably already on hand | – |
| Arduino Uno / Nano or ESP32 | Already on hand | – |
| 5 V power (USB is fine for an SG90) | Already on hand | – |

## What to 3D-print

- A **beam** with a small V-groove or rail down the length so the ball tracks straight and doesn't fall off sideways. Length ~20–30 cm.
- A **servo horn coupling** that converts the servo's rotation into a tilt of the beam. Keep the pivot close to the beam's centre of mass so the servo isn't fighting gravity.
- A **base / stand** that holds the servo, the pivot bearing (or just a friction pivot), and the sensor at one end pointing along the beam toward the ball.

Total print time roughly 1–2 hours.

## Two gotchas to know going in

1. **Ultrasonic sensors have a ~2 cm minimum range.** Position the sensor so the ball never gets closer than that. Or use a **VL53L0X laser ToF sensor** instead (~S$8, also at Sim Lim) — works down to ~3 cm, faster sample rate, less noisy.
2. **SG90 servos have backlash and limited resolution.** This is *good news* for learning — the imperfect actuator makes PID tuning feel real instead of toy-clean. If it bothers you later, swap to MG90S (metal gears, ~S$2 more).

## Porting the simulation code

Once it's mechanically working, the controller code in `balance/pid.py` and `balance/recovery.py` ports almost directly to Arduino / MicroPython / CircuitPython. The loop shape is the same:

```
loop:
    measurement = read_sensor()              # was env.state[0]
    out = pid.update(measurement, dt)        # unchanged
    servo.write(setpoint_angle + out.control) # was env.step(...)
    sleep(dt)
```

Things that change vs. the sim:

- `dt` is your loop's real wall-clock period (`millis()` deltas on Arduino). The "tick rate" is now a real engineering parameter — too slow and the ball escapes.
- Sensor noise is real. You may want a small low-pass filter on the measurement before feeding it to the PID.
- The actuator (servo) has its own internal control loop, so you're really commanding a setpoint, not a torque. That's fine — PID doesn't care.
- Gains will not be the same as the sim. Start from zero, raise Kp until it oscillates, back off, then add Kd, then Ki. Classic Ziegler-Nichols-ish hand-tuning.

## Suggested progression

1. Get the ball to sit still at the centre with **P only** (Ki = Kd = 0). It will oscillate.
2. Add **D** to damp the oscillation. The ball should now hold near centre but may park slightly off if the beam isn't perfectly level.
3. Add **I** to cancel that residual offset.
4. Add disturbance: poke the ball gently and watch the recovery. Port `RecoveryTracker` to count loop iterations from peak displacement to settled.
5. Try varying the loop rate. Run the loop at 20 Hz vs 200 Hz and feel the difference.

That progression mirrors the "experiment with the gains" section of `balance/README.md`.
