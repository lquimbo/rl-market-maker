# RL Market‑Making Engine (WIP) 🏦🤖

Train a reinforcement‑learning agent to post bid/ask quotes in a simulated
limit‑order book and maximise P&L while controlling inventory risk.

| Phase                | Status | ETA        |
|----------------------|:------:|-----------:|
| Project scaffold     | ✔      | 21 June 2025|
| LOB simulator        | ✔      | 07 Jul 2025 |
| Baseline agent&Upload| ✔      | 21 Jul 2025 |
| Double‑DQN training  | ☐      | 28 Jul 2025 |
| Evaluation & plots   | ☐      | 30 Jul 2025 |

> **Note:** Repo under active development; core environment and baseline agent
> will be committed by **30 Jul 2025**.

## Quick Start

```bash
git clone https://github.com/lquimbo/rl-market-maker.git
conda env create -f environment.yml     # env file WIP
python src/agent_training_stub.py

### code stubs

Open a new file **`src\lob_env_stub.py`** :

```python
"""
Gym‑compatible limit‑order‑book environment (stub).
Todo by 30 Jul 2025:
  • state: top‑5 bid/ask levels + inventory
  • actions: ±2‑tick skew
  • reward: realized spread − κ * inventory²
"""

class LobEnv:
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

def main():
    print("Training module coming soon...")

if __name__ == "__main__":
    main()

