# Citadel: Build the Algorithmic Black Box

This repository contains a high-fidelity market microstructure simulator and a custom Reinforcement Learning (RL) environment built for the WiDS 2026 Citadel Challenge.

The project implements an event-driven Limit Order Book (LOB) engine and formalizes the trading problem as a Markov Decision Process (MDP) with a specialized risk-adjusted reward function. This infrastructure serves as the foundation for training automated trading agents.

## Project Overview

The goal of this project is to create a realistic "Black Box" simulator where an agent can learn to trade against background market participants. Unlike static backtesting, this system models latency, liquidity constraints, and market impact.

**Key Features:**

* **Event-Driven Architecture:** A custom event loop (`EventLoop`) handles asynchronous actions, simulating processing delays and network latency.
* **Matching Engine:** A price-time priority matching engine that maintains a functional limit order book.
* **Risk Management:** A sophisticated reward engineering module that penalizes drawdown and inventory accumulation to prevent reckless behavior.
* **Gymnasium Environment:** A fully compliant RL environment ready for integration with modern reinforcement learning libraries.

## Repository Structure

The codebase is organized into modular components:

```text
MarketSim/
├── agents/
│   └── agents.py           # Background market participants (Market Makers, Noise Traders)
├── analytics/              # Tools for analyzing simulation performance
│   ├── metrics.py
│   ├── plots.py
│   ├── snapshots.py
│   └── tape.py             # Trade recording system
├── engine/                 # Core simulation logic
│   ├── event_loop.py       # Asynchronous event scheduler
│   ├── matching_engine.py  # LOB data structure and matching logic
│   └── order.py            # Order class definition
├── environment/            # The RL Interface (MDP)
│   ├── check_env.py        # Script to verify Gymnasium API compliance
│   └── market_environment.py # Custom Gymnasium Environment
├── tests/                  # Validation scripts
│   ├── verifyday2.py       # Verifies environment stability and observation space
│   └── verifyday3.py       # Verifies risk-adjusted reward logic
└── run_simulation.py       # Entry point for running the base market simulation

```

## Installation

This project is built using Python. Ensure you have the necessary dependencies installed:

```bash
pip install gymnasium shimmy numpy pandas

```

## Usage

### 1. Run the Market Simulation

To observe the market engine in action with background agents (Market Makers and Noise Traders) but without an RL agent:

```bash
python MarketSim/run_simulation.py

```

### 2. Verify Environment Stability (Day 2)

Run the environment verification script to ensure the Gymnasium interface works correctly, the observation space is normalized, and the simulation remains stable over long episodes:

```bash
python MarketSim/tests/verifyday2.py

```

### 3. Verify Reward Engineering (Day 3)

Run the reward verification script to confirm that the Risk-Adjusted Reward function is active. This test demonstrates that the environment penalizes the agent for high inventory and drawdowns, even if the PnL is positive.

```bash
python MarketSim/tests/verifyday3.py

```

### 4. API Compliance Check

To ensure the environment is fully compatible with standard RL libraries:

```bash
python MarketSim/environment/check_env.py

```

## Methodology

### The Engine

The `MatchingEngine` implements a standard Order Book model. It supports:

* **Limit Orders:** Passive orders that provide liquidity.
* **Market Orders:** Aggressive orders that remove liquidity.
* **Background Agents:** Autonomous agents that populate the book to create a realistic trading landscape.

### The MDP Formulation (Day 3)

The `GymTradingEnvironment` defines the Reinforcement Learning problem:

* **State Space:** A 5-dimensional normalized vector:
1. Relative Bid Price
2. Relative Ask Price
3. Relative Spread
4. Normalized Inventory
5. Normalized Cash Balance


* **Action Space:** A discrete set of actions (0: Hold, 1: Buy, 2: Sell) to ensure deterministic behavior.
* **Reward Function:**
The reward is calculated as:
`Reward = PnL - (Risk_Aversion * (Drawdown_Penalty + Inventory_Penalty))`
This function ensures that the agent learns to balance profit generation with capital preservation.

## License

This project is open-source.
