import gymnasium as gym
from lob_env import LobEnv
from baseline_agent import FixedSpreadAgent
import numpy as np

env = LobEnv(poisson_rate=20)
agent = FixedSpreadAgent(env.action_space)

obs, _ = env.reset()
pnl_history = []

for _ in range(env.ep_len):
    action = agent.act(obs)
    obs, reward, done, trunc, info = env.step(action)
    pnl_history.append(info["pnl"])
    if done:
        break

print("Final P&L:", pnl_history[-1])
print("Inventory :", obs[2])

import matplotlib.pyplot as plt
plt.plot(pnl_history)
plt.title("Baseline equity curve")
plt.xlabel("Step")
plt.ylabel("PnL")
plt.show()
