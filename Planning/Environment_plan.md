# Environmental Context System

This plan details the implementation of the Environmental Context System as described in the QuarryFlow Nexus Build Plan. It transitions the simulation from static driver behaviors to dynamic, context-aware probability distributions shifted by time, weather, seasons, and an Indian holiday calendar.

## Proposed Changes

---

### Environment Package

We will create a new package `src/quarryflow/environment/` to house the context logic.

#### [NEW] [context.py](file:///c:/Games/QuarryFlow-Crossing/src/quarryflow/environment/context.py)
- Define Enums: `Season`, `WeatherCondition`, `TimeSlot`, `DayType`, `MoodTag`.
- Define `HolidayInfo` and `EnvironmentContext` dataclasses.

#### [NEW] [holidays_india.json](file:///c:/Games/QuarryFlow-Crossing/src/quarryflow/data/holidays_india.json)
- Create the 25-entry static JSON table defining the holidays, traffic multipliers, and mood tags.

#### [NEW] [calendar_engine.py](file:///c:/Games/QuarryFlow-Crossing/src/quarryflow/environment/calendar_engine.py)
- Logic to resolve a timestamp to a `Season`, `TimeSlot`, and `DayType`, including looking up dates in `holidays_india.json`.

#### [NEW] [weather_engine.py](file:///c:/Games/QuarryFlow-Crossing/src/quarryflow/environment/weather_engine.py)
- Logic to sample `WeatherCondition` using the seasonal prior probabilities defined in the design spec.
- Helpers to calculate continuous `road_wetness`, `visibility_norm`, and `temperature_c`.

#### [NEW] [modifier.py](file:///c:/Games/QuarryFlow-Crossing/src/quarryflow/environment/modifier.py)
- The core math layer implementing the logit/sigmoid probability shifts.
- Implements the Two-Dimensional Aggression Model (`P_aggressive` and `A_magnitude`).
- Combines time, weather, and day type delta matrices to compute final spawn probabilities.

---

### Core Domain & Configuration

#### [MODIFY] [domain_types.py](file:///c:/Games/QuarryFlow-Crossing/src/quarryflow/domain_types.py)
- Update `DriverProfileSpec` to remove flat fields (`aggression`, `gate_rush_bias`, `idling_propensity`, `encroachment_bias`) and replace them with a unified `base_probs: dict[str, float]` dict (and add `aggression_magnitude`).
- Introduce `EnvironmentContext` to `EpisodeMetrics` and `CrossingStateSnapshot` if needed for data reporting.

#### [MODIFY] [config.py](file:///c:/Games/QuarryFlow-Crossing/src/quarryflow/config.py)
- Update `DRIVER_PROFILES` to define `base_probs` for `impatience`, `aggression`, `aggression_magnitude`, `gate_rush`, `caution`, `lateral_squeeze`, and `engine_idle` per the table in section 6.

---

### Simulation Physics & Integration

#### [MODIFY] [simulator.py](file:///c:/Games/QuarryFlow-Crossing/src/quarryflow/simulator.py)
- `RailwayCrossingSimulator` or `create_episode` will be updated to accept an `EnvironmentContext`.
- Vehicle spawners will call `compute_driver_probabilities` from `modifier.py` to resolve dynamic behaviors instead of reading fixed configs.
- Update `_apply_vehicle_mix_shift` to adjust traffic composition during holidays.
- Modify car-following and physical update steps to apply `road_wetness` limits to max speed, acceleration, and `A_magnitude`.

#### [MODIFY] [behaviors.py](file:///c:/Games/QuarryFlow-Crossing/src/quarryflow/behaviors.py)
- Ensure the newly mapped probabilities correctly trigger vehicle logic (e.g. using `lateral_squeeze_force`, `barrier_proximity_risk`, `gap_compression`).

---

### UI & Dashboard Integration

#### [MODIFY] [streamlit_app.py](file:///c:/Games/QuarryFlow-Crossing/app/streamlit_app.py)
#### [MODIFY] [dashboard_data.py](file:///c:/Games/QuarryFlow-Crossing/src/quarryflow/dashboard_data.py)
- Inject UI controls to set the `EnvironmentContext` (or let it randomly generate based on a date/time).
- Display current Context (Weather, Temp, Season, Holiday) on the live dashboard.

## Verification Plan

### Automated Tests
- Run `pytest` or `scripts/test_mcts.py` with mock contexts to ensure the simulator still executes cleanly under extreme weather.
- Verify logit math doesn't produce NaN or invalid probabilities.

### Manual Verification
- Launch the Streamlit dashboard.
- Simulate a `HEATWAVE` during `EVENING_RUSH` and observe aggressive gate-rush behavior.
- Simulate `FOG` and observe slow, patient, high-headway behavior.
- Validate that the simulation metrics (delay, throughput) shift logically based on the environment.
