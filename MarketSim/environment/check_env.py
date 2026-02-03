from environment.market_environment import GymTradingEnvironment
from stable_baselines3.common.env_checker import check_env

env = GymTradingEnvironment()

print("Checking environment compliance...")
check_env(env)
print("Environment is compliant!")

obs, _ = env.reset()
done = False
total_reward = 0

print("\nRunning random simulation...")
while not done:
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Simulation finished. Total Reward: {total_reward:.2f}")
print(f"Final Observation: {obs}")