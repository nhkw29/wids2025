from environment.market_environment import GymTradingEnvironment

env = GymTradingEnvironment()
env.reset()

print("--- Day 3 Reward Verification ---")

# 1. Force a BUY to build inventory (create risk)
obs, reward, terminated, _, info = env.step(1) 

print(f"Action: BUY | Inventory: {info['inventory']}")
print(f"PnL Component: {info['reward_pnl_component']:.4f}")
print(f"Risk Penalty:  {info['reward_penalty_component']:.4f}")
print(f"Total Reward:  {reward:.4f}")

if info['reward_penalty_component'] > 0:
    print("SUCCESS: Risk Penalty is active.")
    print("The agent is now being punished for holding inventory/drawdown.")
else:
    print("FAILURE: Risk Penalty is zero. Check calculations.")