QuarryFlow Nexus
Full Build Plan — FAR AWAY Hackathon
Agentic Railway Level Crossing Intelligence Network
June 2026 · India's Biggest International Hackathon
Themes: Railways · Agentic & Autonomous Systems · Logistics & Transit
1. Executive Summary
QuarryFlow Nexus is a multi-crossing, agentic railway level-crossing intelligence network built on QuarryFlow-Crossing, which previously won 2nd place at KnowledgeQuarry (Delhi University). The project is not a rewrite — it is a meaningful extension of a working system.
India has approximately 31,846 railway level crossings, averaging 49 per 100 kilometres of track. Unmanned level crossing accidents account for 56% of all deaths in train accidents. Most competing hackathon submissions address this problem with CV-based barrier detection — identifying vehicles, pedestrians, and barrier states. QuarryFlow Nexus addresses the problem that no one else is modelling: the chaotic surge that occurs the moment a barrier reopens, where the absence of a road median causes vehicles from both directions to squeeze simultaneously into the shared crossing box, creating gridlock even after the gate lifts.
The central technical contribution of this submission is the QuarryFlow-Crossing Environmental Context System: a probabilistic, physics- and agent-based microscopic traffic simulation that makes driver behaviour dynamically vary with real-world environmental conditions — seasons, weather, time of day, and an India-specific holiday calendar. Driver attributes such as impatience, aggression, gate-rush likelihood, and caution are no longer hard-coded values. They are context-shifted base probabilities sampled at each vehicle spawn, producing chaotic, realistic traffic that no other team's simulation will replicate.
The pitch: 
"Every other team built a gate sensor. We built a model of the human beings waiting behind the gate — and then we gave that model a calendar."
2. Baseline — What Is Already Built
The following components exist in the QuarryFlow-Crossing repository and form the foundation for all extensions described in this plan.
2.1 Simulation Core (simulator.py + behaviors.py)
A physics- and agent-based microscopic traffic simulation implemented entirely from scratch. Key properties:
Heterogeneous vehicle agents: cars, bikes, auto-rickshaws, each with distinct physical parameters (length, width, acceleration, lateral flexibility, fuel/emission profiles).
Six psychological driver profiles: cautious, compliant, opportunistic, assertive, aggressive, reckless.
Gaussian-sampled reaction times and minimum gap requirements per profile.
Car-following dynamics with safe braking distance calculations at each time step.
Lateral squeeze/filtering behaviour — vehicles drift into adjacent space stochastically based on aggressiveness and lateral flexibility.
Conflict-freeze logic — bidirectional entry into the crossing box triggers a 1.5–7 second delay modelling the gridlock moment.
Metric integrals: average wait time, fairness gap (L vs R cleared vehicles), fuel/emission integration.
2.2 Policy Selector (Gradient Boosting Regressor)
A surrogate ML model trained on simulation output to select among three release policies:
free_flow — open both sides simultaneously
alternating — alternate L/R releases in fixed intervals
adaptive — dynamically adjust release side based on queue state
2.3 Streamlit Dashboard
A real-time visualisation of simulation state: queue lengths, active policy, vehicle counts, wait metrics, emissions log.
2.4 Test Suite and ASSUMPTIONS.md
Documented assumptions about vehicle behaviour, crossing geometry, and driver profiles. This documentation was cited as a strength by KnowledgeQuarry judges.
3. Environmental Context System — Core New Feature
This is the primary new technical contribution of the submission. Every driver profile attribute that was previously a fixed value becomes a context-shifted base probability. The simulator samples from those shifted distributions at spawn time.
3.1 Core Philosophy
No hard-coded behaviour values. Instead, every driver attribute is computed as:
final_prob = sigmoid( logit(base_prob) + Δ_time + Δ_weather + Δ_holiday + Δ_temperature )
This additive combination in logit space naturally prevents saturation at 0 and 1 without manual clamping. The result is smooth, realistic probability curves rather than clipped sums. A representative example:
Example: 
90% chance of being impatient for an office worker during weekday morning rush in May heatwave. Not "aggressiveness = 0.8" — a distribution that shifts with reality.
3.2 The Three Environmental Axes
Time — six time slots from night truck runs to morning rush to evening rush
Weather/Season — seven weather conditions sampled from India-realistic seasonal priors
Calendar/Holiday — 25 India-specific holiday entries including gazetted, religious, and Uttarakhand regional events; lunar-calendar festivals (Eid, Muharram) handled via a separate dynamic file keyed by year
3.3 New Module Map
src/quarryflow/
├── environment/
│   ├── __init__.py
│   ├── context.py          # Enums + EnvironmentContext dataclass
│   ├── calendar_engine.py  # Date → Season → Holiday resolution
│   ├── weather_engine.py   # Weather state machine, seasonal priors
│   └── modifier.py         # Probability shift layer — core logic
├── data/
│   └── holidays_india.json # 25-entry holiday table
├── simulator.py            # MODIFIED — accepts EnvironmentContext at spawn
├── behaviors.py            # MODIFIED — uses context-shifted probabilities
├── config.py               # MODIFIED — base_probability fields added
└── dashboard.py            # MODIFIED — context panel + live dist display
3.4 context.py — Enums and Dataclasses
Five core enumerations define the environmental state space:
Enum
Values
Notes
Season
WINTER, SPRING, SUMMER, MONSOON, AUTUMN
Monsoon (Jun–Sep) most consequential for India
WeatherCondition
CLEAR, CLOUDY, LIGHT_RAIN, HEAVY_RAIN, FOG, THUNDERSTORM, HEATWAVE
FOG = North India winter staple; HEATWAVE = Dry + >42°C
TimeSlot
NIGHT, EARLY_MORNING, MORNING_RUSH, MIDDAY, EVENING_RUSH, NIGHT_MARKET
Evening rush often worse than morning in Indian cities
DayType
WEEKDAY, SATURDAY, SUNDAY, HOLIDAY, FESTIVAL
FESTIVAL = dense + chaotic (Diwali eve, Eid morning)
The EnvironmentContext dataclass carries: timestamp, season, weather, time_slot, day_type, temperature_c, visibility_norm (0–1), road_wetness (0–1), and optional HolidayInfo.
3.5 Holiday Table — holidays_india.json
25 entries covering gazetted public holidays, major religious festivals, and Uttarakhand regional events. Each entry specifies:
name — human-readable holiday name
day_type — HOLIDAY (reduced traffic) or FESTIVAL (dense/chaotic)
traffic_multiplier — relative to normal weekday volume (range: 0.30 for Diwali morning to 2.50 for Diwali Eve)
vehicle_mix_shift — per-type fractional deltas summing to zero (e.g., Dussehra: bikes +0.25, trucks −0.40)
mood_tag — relaxed / solemn / festive / chaotic
Representative entries from the table:
Date
Holiday
Type
Traffic ×
Mood
10-20
Diwali Eve (Chhoti Diwali)
FESTIVAL
2.50×
chaotic
10-12
Dussehra
FESTIVAL
1.80×
chaotic
09-05
Ganesh Chaturthi (eve)
FESTIVAL
1.60×
chaotic
10-21
Diwali (day of)
FESTIVAL
0.30×
relaxed
08-15
Independence Day
HOLIDAY
0.40×
solemn
01-14
Makar Sankranti / Uttarayani
FESTIVAL
1.40×
festive
Lunar-calendar dates (Eid, Muharram, etc.) are stored separately in holidays_dynamic.json keyed by year and resolved at runtime.
3.6 Weather Engine — Seasonal State Machine
Weather is sampled at episode start from a seasonal prior distribution and can transition slowly across a long simulation run. The seasonal priors are calibrated for North India / Uttarakhand:
Season
Clear
Cloudy
Lt Rain
Hvy Rain
Fog
T-Storm
Heatwave
WINTER
0.30
0.25
0.08
0.02
0.35 ★
0.00
0.00
SPRING
0.55
0.25
0.10
0.05
0.05
0.00
0.00
SUMMER
0.40
0.15
0.05
0.00
0.00
0.10
0.30 ★
MONSOON
0.03
0.12
0.30
0.35 ★
0.00
0.20
0.00
AUTUMN
0.50
0.25
0.15
0.05
0.05
0.00
0.00
★ = peak probability for that condition in that season.
Derived fields from weather state: visibility_norm (1.00 clear → 0.20 fog), road_wetness (0.00 dry → 0.95 thunderstorm), temperature sampled from calibrated (season, weather) ranges.
4. Two-Dimensional Aggression Model
This is the most important conceptual upgrade to the driver behaviour system. A single aggression probability cannot capture real human behaviour. The key insight is that wanting to be aggressive and how aggressively you act are two separate phenomena with different input sensitivities.
Example: 
Fog suppresses what a driver physically does (magnitude) without reducing the mental decision to try (probability). Heat rage amplifies intensity more than it changes whether the decision is made at all. Separating these prevents the physically impossible behaviour of a gridlocked driver attempting to rush at full speed.
4.1 The Two-Dimensional Model
Dimension
Symbol
Description
Aggression Probability
P_aggressive
Probability that a driver enters an aggressive behavioural mode this episode [0→1]
Aggression Magnitude
A_magnitude
Given aggressive mode: intensity multiplier controlling how aggressively the driver acts [0→1]
P_aggressive  = sigmoid( logit(base_P) + Σ context_deltas_P )
A_magnitude   = sigmoid( logit(base_A) + Σ context_deltas_A )
4.2 Physical Parameters Controlled by A_magnitude
When P_aggressive resolves as True (Bernoulli draw), A_magnitude scales the following physical parameters:
Parameter
Formula
Effect
gate_rush_speed
base_speed × (1 + 0.6 × A)
Up to 60% faster gate approach
gap_compression
min_gap × (1 − 0.7 × A)
Up to 70% tighter following distance
lateral_squeeze_force
base_squeeze × (1 + 0.8 × A)
Up to 80% more lateral drift
reaction_time_compress
rt × (1 − 0.5 × A)
Up to 50% shorter reaction time
barrier_proximity_risk
safe_dist × (1 − 0.65 × A)
Attempts to pass closer to barrier
queue_jump_positions
floor(3 × A)
Attempts to skip 0–3 vehicles in queue
4.3 Context Deltas for P_aggressive (Likelihood)
Context Factor
Shift Direction
Delta (logit)
Rationale
MORNING_RUSH time slot
↑ Higher P
+0.80
Late-for-work stress
EVENING_RUSH time slot
↑ Higher P
+1.00
Worst of the day in Indian cities
NIGHT time slot
↑ Higher P
+0.60
Fatigue aggression
HEATWAVE weather
↑↑ Higher P
+1.00
Well-documented heat–aggression link
HEAVY_RAIN weather
↓ Lower P
−0.20
Caution tempers decision
FOG weather
↓↓ Lower P
−0.80
Cannot see → do not attempt
FESTIVAL day type
↑ Higher P
+0.80
Festive excitement + chaos
HOLIDAY day type
↓ Lower P
−0.60
Relaxed, no deadline
High traffic density (>85th pctile)
↑ Higher P
+0.50
Frustration buildup
Long accumulated wait (>4 min)
↑ Higher P
+0.60
Frustration curve steepens
Social contagion (neighbour rushed)
↑ Higher P
+0.40
"Others are doing it" effect
Temperature > 42°C
↑ Higher P
+1.20
Heat judgment impairment peak
4.4 Context Deltas for A_magnitude (Intensity)
Context Factor
Shift Direction
Delta (logit)
Rationale
HEATWAVE weather
↑↑ Higher A
+1.20
Heat amplifies intensity most
FOG weather
↓↓ Lower A
−1.50
Physical constraint — cannot rush blind
HEAVY_RAIN weather
↓ Lower A
−0.80
Wet surface prevents aggressive manoeuvres
High road_wetness (>0.7)
↓ Lower A
−0.60
Physics: traction limited
MORNING_RUSH time slot
↑ Higher A
+0.70
"Cannot be late" urgency
FESTIVAL + chaotic mood
↑ Higher A
+0.90
Social disinhibition in crowds
High visibility_norm (>0.85)
↑ Higher A
+0.30
Can see clearly → attempts bolder moves
Low visibility_norm (<0.30)
↓ Lower A
−1.20
Physical constraint dominates
Bike vehicle type
↑ Higher A
+0.50
Smaller vehicle → more squeeze options
Truck vehicle type
↓ Lower A
−0.80
Large vehicle cannot manoeuvre aggressively
4.5 Frustration Accumulation Curve
Wait time feeds a non-linear frustration accumulator that independently raises both P and A:
frustration = tanh(wait_minutes / τ)        # τ = 3.0 minutes (half-saturation)
P_delta_wait = +0.80 × frustration
A_delta_wait = +0.60 × frustration
This produces a sigmoid-shaped frustration curve: minimal effect for short waits, rapid rise between 2–5 minutes, plateau above 8 minutes. The tanh function prevents runaway values without explicit clamping.
5. modifier.py — The Probability Shift Layer
The modifier module is the computational heart of the environmental context system. Every vehicle spawn passes through here. The public API is:
compute_driver_probabilities(driver_type, base_probs, ctx, rng) → dict[str, float]
It combines time, weather, day type, temperature, and the new two-dimensional aggression model into final spawn-time probabilities for six driver attributes: impatience, aggression (P), aggression_magnitude (A), gate_rush, caution, lateral_squeeze, engine_idle.
5.1 Driver Attribute Delta Tables
Each attribute has three delta tables (time_slot, weather, day_type) plus a continuous temperature function. Below is the complete impatience table as representative:
Condition
Attribute: Impatience Δ
Notes
MORNING_RUSH
+1.20
Largest single-slot impatience surge
EVENING_RUSH
+1.00
Second highest — often worse in India
NIGHT_MARKET
+0.30
Social rush to get home or reach venue
FOG (weather)
−0.60
Forced slow — patience from necessity
HEATWAVE (weather)
+0.80
Heat stress + wanting AC on other side
FESTIVAL (day type)
+0.50
Running to event / mela
HOLIDAY (day type)
−0.70
No deadline = relaxed
Temperature > 40°C
+0.70 continuous
Peaks beyond 40°C
5.2 Temperature as Continuous Modifier
Temperature is the only factor treated as a continuous variable (others are categorical). Key thresholds:
Aggression: +1.20 above 42°C, +0.60 above 36°C, −0.30 below 8°C (cold = sluggish, not angry)
Impatience: +0.70 above 40°C, −0.20 below 8°C
Caution: +0.60 below 6°C (ice-risk awareness), −0.50 above 43°C (judgment impaired by heat)
6. Driver Profiles — config.py Base Probabilities
Each of the six driver profiles gains a base_probs block. The modifier layer replaces these with context-shifted values at spawn time. The static fields (reaction_time_mean, min_gap, etc.) remain unchanged.
Profile
impatience
aggression P
aggr. mag.
gate_rush
caution
lat. squeeze
engine_idle
cautious
0.10
0.05
0.05
0.03
0.90
0.05
0.30
compliant
0.25
0.15
0.15
0.10
0.70
0.15
0.40
opportunistic
0.60
0.45
0.45
0.40
0.35
0.55
0.60
assertive
0.70
0.60
0.60
0.55
0.20
0.65
0.70
aggressive
0.85
0.80
0.80
0.75
0.10
0.80
0.80
reckless
0.95
0.95
0.95
0.95
0.03
0.95
0.90
Note: aggression_magnitude is now a separate field from the aggression probability. Even a reckless driver at 0.95 base magnitude will have that value compressed by fog or heavy rain — the context always wins over the base.
7. simulator.py — Integration Points
Three targeted changes to the existing simulator. The goal is minimum-invasive integration — the simulation physics remain unchanged.
7.1 Episode Factory Takes a Context
def create_episode(ctx: EnvironmentContext, rng: np.random.Generator) -> Episode:
    vehicle_mix = _apply_vehicle_mix_shift(BASE_VEHICLE_MIX, ctx)
    arrival_rate = BASE_ARRIVAL_RATE * (ctx.holiday_info.traffic_multiplier
                                        if ctx.holiday_info else 1.0)
    return Episode(ctx=ctx, vehicle_mix=vehicle_mix, arrival_rate=arrival_rate)
7.2 Vehicle Spawn Samples Two-Dimensional Aggression
def spawn_vehicle(driver_type, ctx, rng) -> Vehicle:
    base_probs = DRIVER_PROFILES[driver_type]['base_probs']
    shifted    = compute_driver_probabilities(driver_type, base_probs, ctx, rng)
    p_agg      = shifted['aggression']
    a_mag      = shifted['aggression_magnitude']
    is_aggressive = rng.random() < p_agg
    magnitude     = a_mag if is_aggressive else 0.0
    behaviours = {attr: rng.random() < prob
                  for attr, prob in shifted.items()
                  if attr != 'aggression_magnitude'}
    return Vehicle(..., aggression_magnitude=magnitude, ...)
7.3 Road Wetness Modifies Physics
effective_max_speed = vehicle.max_speed * (1 - 0.4 * ctx.road_wetness)
effective_accel     = vehicle.accel     * (1 - 0.3 * ctx.road_wetness)
# A_magnitude is further clamped by road_wetness at execution time:
effective_A = vehicle.aggression_magnitude * (1 - 0.6 * ctx.road_wetness)
8. Data Generation Pipeline
The data is not generated by another machine learning model, nor is it simplified statically hardcoded metrics. The project uses a physics- and agent-based microscopic traffic simulation environment implemented from scratch. Every episode row now carries context columns enabling the surrogate model and policy selector to condition on environmental state.
8.1 How the Simulation Produces Data
Heterogeneous agents — vehicles spawned with distinct physical and psychological profiles from config.py
Microscopic movement — car-following dynamics, safe braking, lateral drift modelled per tick
Conflict freeze — bidirectional entry into the crossing box triggers 1.5–7 second delay
Metric integration — wait time, fairness gap, fuel/emissions computed from true simulator state
Context injection — EnvironmentContext shifts all behaviour probabilities before spawn; rod physics shift at tick level
8.2 Episode Schema with Context Columns
episode_id | season | weather | time_slot | day_type | temp_c | visibility |
road_wet   | traffic_mult | p_agg_mean | a_mag_mean | conflict_count |
wait_avg   | fairness_gap | fuel_kg | policy_chosen | ...
8.3 Coverage Targets
Axis
Values
Notes
Season
5
WINTER, SPRING, SUMMER, MONSOON, AUTUMN
WeatherCondition
7
Including FOG and HEATWAVE
TimeSlot
6
Including NIGHT and NIGHT_MARKET
DayType
5
Including FESTIVAL (distinct from HOLIDAY)
Unique context combinations
1,050
5 × 7 × 6 × 5
Episodes per context
~500
For statistical robustness
Total target rows
~525,000
Achievable in a few hours of simulation compute
9. QuarryFlow Nexus — Extended Architecture
The Environmental Context System integrates into the broader QuarryFlow Nexus multi-crossing agent network described in the build plan. The four components are:
Component A — MCTS Rollout Agent (upgrade to policy selector)
Replaces the Gradient Boosting Regressor with an MCTS-style rollout search:
State: per-direction queue length/wait by vehicle type, barrier status, predicted train ETA, neighbouring crossing queue states
Actions: free_flow, alternating, batched_release, emergency_priority
Evaluation: for each candidate action, run the existing stochastic simulator forward K ticks × M Monte Carlo rollouts; score = −(total wait + max-queue penalty + emergency-delay penalty)
Reused from Critical-Mass: search loop structure, transposition table (rollout caching), iterative deepening for time-budget management
Every rollout now runs with a full EnvironmentContext — the agent sees the weather and time when it simulates ahead, not a neutral context
Component B — Edge CV Node
MobileNetV2 + SE attention, INT8 quantised (continuity with Forsaken-Apex edge target)
Two jobs from one camera: classify/count queued vehicles by type while traffic is present; scan rail surface for defects during gaps between trains
Vehicle counts feed real queue state into Component A
Component C — Multi-Crossing Network
Extend simulation from one crossing to N (2–3 for the demo) on a shared corridor
Each crossing runs its own Component-A agent
Agents exchange predicted queue states with adjacent crossings each tick — lightweight negotiation without multi-agent RL
Component D — Network Dashboard
Extends existing QuarryFlow-Crossing Streamlit app
Aggregates state across all agents: which crossing is trending toward backup, anomaly flags
Context panel: date/time/weather controls, live probability distribution bars per driver type
The key demo moment: drag the time slider from 8 AM to 5 PM on Diwali Eve in a heatwave and watch driver probability bars shift in real time
10. Dashboard Context Panel — Streamlit Additions
The dashboard gains a real-time context panel and live driver behaviour distribution display. This is the single most compelling live demo element.
10.1 Sidebar Context Control
with st.sidebar:
    st.header('Environment Context')
    sim_date = st.date_input('Date', value=date.today())
    sim_time = st.slider('Hour of Day', 0, 23, 8)
    override_weather = st.selectbox('Weather Override',
                        ['Auto (seasonal prior)'] + [w.name for w in WeatherCondition])
    ctx = build_context(sim_date, sim_time, override_weather)
    st.metric('Season',       ctx.season.name)
    st.metric('Temperature',  f'{ctx.temperature_c:.1f}°C')
    st.metric('Visibility',   f'{ctx.visibility_norm*100:.0f}%')
    if ctx.holiday_info:
        st.warning(f'🎉 {ctx.holiday_info.name} — '
                   f'{ctx.holiday_info.traffic_multiplier}× traffic')
10.2 Live Probability Distribution Display
st.subheader('Driver Behaviour Probabilities — Current Context')
for dtype in DRIVER_PROFILES:
    base   = DRIVER_PROFILES[dtype]['base_probs']
    shifted = compute_driver_probabilities(dtype, base, ctx, rng)
    df = pd.DataFrame({'base': base, 'now': shifted}).T
    st.write(f'**{dtype}**')
    st.bar_chart(df)
The demo narrative for this panel: set the date to Diwali Eve (Oct 20), hour to 18:00 (evening rush), weather to HEATWAVE. Every driver type's impatience, aggression probability, and aggression magnitude bars visibly spike. Set it to Republic Day (Jan 26) morning, clear weather, watch them all drop. This is the live proof that the environmental context system works.
11. Implementation Order
Estimated at 6–8 hours for a developer familiar with the existing codebase. The dashboard panel alone (Step 9) is sufficient for a compelling demo.
#
Time
What
Output
1
30 min
Write context.py — Season, WeatherCondition, TimeSlot, DayType enums + EnvironmentContext dataclass
context.py ✓
2
30 min
Write calendar_engine.py + holidays_india.json (25 entries + lunar dynamic file structure)
calendar_engine.py, holidays_india.json ✓
3
45 min
Write weather_engine.py — seasonal prior tables, sample_weather(), sample_temperature(), derive_visibility(), derive_road_wetness()
weather_engine.py ✓
4
75 min
Write modifier.py — TIME_DELTAS, WEATHER_DELTAS, DAY_TYPE_DELTAS tables + two-dimensional aggression model (P and A separate tables) + temperature_delta() + compute_driver_probabilities()
modifier.py ✓ — core logic
5
30 min
Update config.py — add base_probs block (including aggression_magnitude) to all 6 driver profiles
config.py ✓
6
60 min
Update simulator.py — episode factory with ctx, spawn_vehicle() with two-dimensional aggression draw, road_wetness physics scaling on speed/accel/A
simulator.py ✓
7
30 min
Regenerate training data with context columns (season, weather, time_slot, day_type, temp_c, p_agg_mean, a_mag_mean)
~525K episode rows
8
30 min
Retrain surrogate model / policy selector with new context features
Updated model ✓
9
45 min
Dashboard panel — date/time/weather sidebar controls + live probability bar charts per driver type
dashboard.py ✓ — demo-ready
10
60 min
Multi-crossing extension: extend sim to 2–3 crossings, basic neighbour state-sharing, network dashboard layer
Nexus architecture ✓
Total estimated time: 7.5 hours. Steps 1–9 cover the full Environmental Context System. Step 10 completes the Nexus multi-agent architecture.
12. Build Priorities
Tier
Item
Source
Notes
MVP
Environmental Context System (Sections 3–8 of this plan)
New (this plan)
The core differentiator
MVP
Two-dimensional aggression model
New (Section 4)
P_aggressive + A_magnitude
MVP
Extend sim to 2–3 crossings on one corridor
QuarryFlow-Crossing
Foundation for agent network
MVP
MCTS rollout agent (Component A)
Critical-Mass + QF sim
Compare vs GBR baseline live
MVP
Basic neighbour state-sharing (Component C)
New
Even a shared dict counts
MVP
Dashboard context panel + live probability bars
QF-Crossing Streamlit
Best demo moment
MVP
3–4 graded scenarios runnable live
openenv-vendor-compliance
Most visually dramatic first
Should-have
Component B simplified — webcam + pretrained detector for live vehicle counting
Forsaken-Apex (lightweight)
Full retrain is stretch
Should-have
Expand to 8–12 graded scenarios
openenv-vendor-compliance
Strong stress-test moment
Should-have
Bottleneck/anomaly layer on dashboard
VisualizeSim
Only if analysis.py confirmed
Stretch
PCB design (KiCad) for sensor node
New
Radar/ultrasonic + camera, NXP MCU
Stretch
Physical sensor-node prototype
New
Solar-powered, rail-mountable
Stretch
Full rail-defect retrain on RailVista
Forsaken-Apex
Pairs with PCB for edge story
Stretch
Trained value network for Component A
Critical-Mass
Real RL story, not demo-critical
13. Demo Script (2–5 Min Video or ≤15 Slides)
#
Segment
What to Show
1
The Problem
31,846 crossings in India. 56% of train accident deaths at unmanned crossings. Show the bidirectional gridlock moment — vehicles squeezing from both sides when the barrier lifts.
2
Baseline
Fixed-policy crossing gridlocks after the barrier reopens. Static driver profiles — no context awareness. Old GBR selector.
3
Context System
Open the dashboard. Set date to Diwali Eve (Oct 20), time 18:00, HEATWAVE. Watch all driver probability bars spike. Set to Republic Day morning, clear weather — watch them all drop. This is the live proof.
4
Two-Dim Aggression
Show a fog scenario: P_aggressive stays moderate, A_magnitude collapses. Show a heatwave scenario: both spike. Explain why separating these prevents the physically impossible gridlocked-driver-at-full-speed output.
5
MCTS Agent
Switch on the rollout-search agent. Show it choosing differently from the baseline, with a brief visualisation of candidate policies being evaluated ahead of time.
6
Network
Show 2–3 crossings negotiating: one signals backup, its neighbour adjusts policy. Inject an ambulance-priority scenario — agent reprioritises live.
7
Test Suite
Run the graded scenario suite. Show N/N scenarios passed.
8
Operational Q
Close with: "How many minutes of delay should we expect at crossing X-47 on Diwali Eve at 6 PM in heavy monsoon rain?" That is a real operational question. That is real-world impact.
14. Open Questions / Next Steps
Confirm what VisualizeSim's analysis.py actually contains (DTW, changepoint detection, LLM diagnosis) before committing the dashboard anomaly layer to it.
Pin down evaluation-function weights for Component A: how to trade off total wait time vs. max queue penalty vs. emergency delay penalty.
Decide rollout parameters: K (ticks per rollout) and M (rollouts per action) — bounded by how fast the existing simulator runs.
Map team members to workstreams (sim/agent/environment, CV, dashboard, hardware) based on who is strongest where.
Confirm Forsaken-Apex vehicle detection model performance on the crossing camera angle before committing to live vehicle counting for the MVP.
Add remaining temperature range entries to TEMP_RANGES in weather_engine.py — the current spec has representative entries, all combinations need filling.
Lunar-calendar holidays (Eid al-Fitr, Eid al-Adha, Muharram) need year-specific dates resolved in holidays_dynamic.json for 2026.
Repositories
https://github.com/Codiosityy/QuarryFlow-Crossing (base project — extend this)
https://github.com/Codiosityy/Forsaken-Apex (CV / edge model)
https://github.com/Codiosityy/openenv-vendor-compliance (scenario / evaluation framework)
https://github.com/NexiSynapse/Critical-Mass (search infrastructure for Component A)
"Every other team built a gate sensor. We built a model of the human beings waiting behind the gate — and then we gave that model a calendar."