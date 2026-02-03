Here is the clean, emoji-free version of the humanized `README.md`.

---

# Citadel: Build the Algorithmic Black Box

Hi! This is my submission for the **WiDS 2026 Citadel Challenge**.

The goal was to build a realistic market simulator from scratch (no simple backtesting on static data!) and train a Reinforcement Learning agent to trade profitably without blowing up the account.

I successfully completed **Phase I (Days 1–5)**, building the "Black Box" engine and proving that a PPO agent can learn to navigate the order book.

## What's in the Box?

Instead of just replaying historical prices, I built an **Event-Driven Simulator**. This means the market reacts to the agent, and orders take time to travel (simulated latency).

* **The Engine:** A custom Limit Order Book (LOB) matching engine that respects price-time priority.
* **The Environment:** A `Gymnasium` environment that feeds normalized market data to the AI.
* **The Agent:** A **PPO (Proximal Policy Optimization)** model that learns to trade against background Market Makers and Noise Traders.
* **Risk Management:** I engineered a custom reward function so the agent doesn't just gamble. It gets penalized for high drawdowns and holding too much inventory.

## Project Structure

Here is how I organized the code:

* `MarketSim/engine/`: The core logic. Contains the `MatchingEngine` and `EventLoop` (simulates time/delays).
* `MarketSim/environment/`: The bridge between the sim and the AI. Contains `GymTradingEnvironment`.
* `MarketSim/agents/`: The "NPCs" of the market (Noise Traders & Market Makers).
* `train_day5.py`: The main script to train the PPO agent.

## How to Run It

I developed this using Python 3.11. Here is how to get it running on your machine:

**1. Clone the repo and install dependencies:**

```bash
git clone https://github.com/your-username/algorithmic-black-box.git
cd algorithmic-black-box
pip install gymnasium "stable-baselines3[extra]" shimmy numpy pandas

```

**2. Run the "Sanity Check" Training:**
I created a script that trains the agent for 50k steps to prove it learns.

```bash
python train_day5.py

```

*You should see the `ep_rew_mean` (mean reward) creeping up in the logs!*

**3. Verify the Environment:**
If you want to check if the Gym environment is compliant (no bugs/crashes):

```bash
python MarketSim/environment/check_env.py

```

## Key Learnings & Methodology

### 1. The "Gambler" Problem

My biggest challenge was that the initial agent would just buy max quantity and hope for the best. To fix this, I implemented a **Risk-Adjusted Reward**:



This forces the agent to keep its inventory near zero and exit losing trades quickly.

### 2. Handling Latency

Real trading isn't instant. I added an **Event Loop** that adds a small delay (e.g., 50ms) to every order. The agent has to predict where the price *will be* when its order arrives, not just where it is now.

### 3. Normalization is Key

Feeding raw prices (like `150.00`) to a neural net usually breaks it. I converted everything to relative values (e.g., "Price is 0.05% above the mid").

## Future Work

If I had more time, I would:

* Switch from Discrete actions (Buy/Sell Fixed Amount) to Continuous ones (choose exact price/qty).
* Add a multi-agent setup where multiple RL agents compete against each other.
* Visualize the order book depth in real-time with a dashboard.

---

*Built for the WiDS 2026 Datathon.*
