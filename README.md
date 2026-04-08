# 🚦 QuarryFlow-Crossing

**Adaptive Traffic Release Strategies for Railway Level Crossings**
A simulation and ML-driven framework to model vehicle queues at railway crossings and evaluate intelligent gate release policies.

---

# ✨ Overview

QuarryFlow-Crossing simulates **two-sided traffic queues** at railway level crossings and evaluates multiple **vehicle release strategies**, including:

* Free Flow Release
* Alternating Release
* Adaptive Burst Release
* ML-based Policy Selection (RandomForest)

The system models:

* Mixed vehicle types
* Queue accumulation
* Aggressive drivers
* Waiting time penalties
* Crossing clearance efficiency

---

# 🧠 System Architecture

```mermaid
flowchart TB
    A[Scenario Config] --> B[Traffic Generator]
    B --> C[Queue Builder]

    C --> D[Barrier Closed State]
    D --> E[Vehicle Accumulation]

    E --> F[Barrier Opens]

    F --> G{Policy Engine}
    G -->|Free Flow| H1[Release All]
    G -->|Alternating| H2[Side Switching]
    G -->|Adaptive| H3[ML Policy Selector]

    H3 --> RF[RandomForest Model]

    H1 --> I[Vehicle Discharge]
    H2 --> I
    H3 --> I

    I --> J[Metrics Collector]
    J --> K[Simulation Results]
    K --> L[Streamlit Dashboard]
```

---

# 🚦 Railway Crossing Simulation Flow

```mermaid
flowchart LR
    Start --> TrainIncoming
    TrainIncoming --> CloseGate
    CloseGate --> BuildQueues

    BuildQueues --> LeftQueue
    BuildQueues --> RightQueue

    LeftQueue --> Wait
    RightQueue --> Wait

    Wait --> TrainPasses
    TrainPasses --> GateOpens

    GateOpens --> PolicySelection
    PolicySelection --> VehicleRelease

    VehicleRelease --> ClearQueues
    ClearQueues --> End
```

---

# 🤖 Adaptive ML Policy Selection

```mermaid
flowchart TB
    A[Queue Length Left]
    B[Queue Length Right]
    C[Vehicle Mix]
    D[Aggressiveness Score]
    E[Wait Time]

    A --> F[Feature Vector]
    B --> F
    C --> F
    D --> F
    E --> F

    F --> RF[RandomForest Model]

    RF --> P1[Free Flow]
    RF --> P2[Alternating]
    RF --> P3[Adaptive Burst]

    P1 --> Output[Selected Policy]
    P2 --> Output
    P3 --> Output
```

---

# 🔁 Core Simulation Loop

```mermaid
flowchart TB
    Start --> Init
    Init --> SpawnVehicles
    SpawnVehicles --> UpdateQueues
    UpdateQueues --> CheckBarrier

    CheckBarrier -->|Closed| Accumulate
    CheckBarrier -->|Open| ApplyPolicy

    Accumulate --> TimeStep
    ApplyPolicy --> ReleaseVehicles

    ReleaseVehicles --> UpdateMetrics
    UpdateMetrics --> TimeStep

    TimeStep --> Continue{More Time?}
    Continue -->|Yes| SpawnVehicles
    Continue -->|No| End
```

---

# 📊 Policy Comparison Pipeline

```mermaid
flowchart LR
    Scenario --> RunFreeFlow
    Scenario --> RunAlternating
    Scenario --> RunAdaptive

    RunFreeFlow --> Metrics
    RunAlternating --> Metrics
    RunAdaptive --> Metrics

    Metrics --> Compare
    Compare --> Charts
    Charts --> Dashboard
```

---

# 🎛️ Streamlit Dashboard Architecture

```mermaid
flowchart TB
    User --> UI[Streamlit UI]

    UI --> LoadScenario
    UI --> SelectPolicy
    UI --> RunSimulation

    RunSimulation --> Engine
    Engine --> Results

    Results --> Charts
    Results --> Tables
    Results --> KPIs

    Charts --> UI
    Tables --> UI
    KPIs --> UI
```

---

# 🧩 Vehicle Release Strategy Logic

```mermaid
flowchart TB
    GateOpen --> CheckPolicy

    CheckPolicy --> FreeFlow
    CheckPolicy --> Alternating
    CheckPolicy --> Adaptive

    FreeFlow --> ReleaseAll

    Alternating --> LeftSide
    Alternating --> RightSide

    Adaptive --> EvaluateQueues
    EvaluateQueues --> BurstLeft
    EvaluateQueues --> BurstRight

    ReleaseAll --> Exit
    LeftSide --> Exit
    RightSide --> Exit
    BurstLeft --> Exit
    BurstRight --> Exit
```

---

# 📁 Project Structure

```
QuarryFlow-Crossing
│
├── src/
│   ├── simulation/
│   ├── policies/
│   ├── models/
│   └── metrics/
│
├── scripts/
│   ├── train_model.py
│   ├── compare_policies.py
│   └── generate_data.py
│
├── app/
│   └── streamlit_app.py
│
├── tests/
│
└── README.md
```

---

# ⚙️ Installation

```bash
git clone https://github.com/Codiosityy/QuarryFlow-Crossing
cd QuarryFlow-Crossing
pip install -r requirements.txt
```

---

# ▶️ Run Simulation

```bash
python scripts/compare_policies.py
```

---

# 🤖 Train ML Policy Model

```bash
python scripts/train_model.py
```

---

# 📊 Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

# 📈 Metrics Evaluated

* Average wait time
* Queue length
* Clearance time
* Throughput
* Fairness index
* Starvation probability

---

# 🧪 Policies Implemented

| Policy         | Description                       |
| -------------- | --------------------------------- |
| Free Flow      | All vehicles released immediately |
| Alternating    | Left-right switching              |
| Adaptive Burst | Larger queue priority             |
| ML Adaptive    | RandomForest decision             |

---

# 🎯 Use Cases

* Railway crossing optimization
* Traffic signal research
* Smart city simulation
* Reinforcement learning experiments
* Transport policy evaluation

---

# 🧠 Future Work

* Reinforcement learning policy
* Multi-crossing simulation
* Real-world dataset integration
* Live traffic API support
* GPU simulation engine

---

# 📜 License

MIT License

---

# 👨‍💻 Author

**Codiosityy**

---

# ⭐ Star the repo if you find it useful
