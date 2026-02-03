from environment.market_environment import GymTradingEnvironment
from stable_baselines3.common.env_checker import check_env

# 1. Initialize
env = GymTradingEnvironment()

# 2. Run Official Check
# If this prints nothing or just warnings, you passed. 
# If it crashes, post the error.
print("Running Gymnasium Compliance Check...")
check_env(env)
print("Environment Compliance Passed!")

# 3. Test a quick loop
obs, _ = env.reset()
done = False
steps = 0
while not done:
    action = env.action_space.sample() # Random action
    obs, reward, terminated, truncated, _ = env.step(action)
    steps += 1
    if steps >= 100: break


print(f"Simulation Loop Passed ({steps} steps). Last Reward: {reward:.4f}")
