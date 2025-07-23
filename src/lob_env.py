import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LobEnv(gym.Env):
    """
    Synthetic limit‑order‑book environment.
    State  : [best_bid, best_ask, inventory]
    Action : {-2, -1, 0, +1, +2}  (skew in ticks from mid)
    Reward : realized PnL – kappa * inventory^2
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        poisson_rate=20,
        tick_size=0.01,
        max_inventory=50,
        kappa=1e-4,
        episode_length=1_000,
    ):
        super().__init__()
        self.tick = tick_size
        self.poisson = poisson_rate
        self.max_inv = max_inventory
        self.kappa = kappa
        self.ep_len = episode_length

        # Action: discrete skew
        self.action_space = spaces.Discrete(5)  # maps to [-2,-1,0,+1,+2] ticks
        # Observation: best_bid, best_ask, inventory
        low  = np.array([0.0, 0.0, -self.max_inv], dtype=np.float32)
        high = np.array([1e6, 1e6,  self.max_inv], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.reset()

    def _simulate_mid_move(self):
        """Random walk mid‑price; Poisson events per step."""
        n = np.random.poisson(self.poisson)
        jump = (np.random.choice([-1, +1], size=n).sum()) * self.tick
        self.mid_price = max(self.tick, self.mid_price + jump)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.mid_price = 100.0  # arbitrary start
        self.inventory = 0
        self.cash = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        best_bid = self.mid_price - 0.5 * self.tick
        best_ask = self.mid_price + 0.5 * self.tick
        return np.array([best_bid, best_ask, self.inventory], dtype=np.float32)

    def step(self, action_idx):
        """
        Execute market‑maker quotes for one time step.
        action_idx ∈ {0..4} maps to skew ticks {-2..+2}
        """
        self.step_idx += 1
        done = self.step_idx >= self.ep_len

        skew_ticks = action_idx - 2  # map 0→-2, 2→0, 4→+2
        bid_quote = self.mid_price - self.tick + skew_ticks * self.tick
        ask_quote = self.mid_price + self.tick + skew_ticks * self.tick

        # Random counter‑party flow: bernoulli for hit/lift
        if np.random.rand() < 0.5:
            # hit our bid
            self.inventory += 1
            self.cash -= bid_quote
        if np.random.rand() < 0.5:
            # lift our ask
            self.inventory -= 1
            self.cash += ask_quote

        # Simulate underlying mid‑price move
        self._simulate_mid_move()

        # Mark‑to‑market PnL
        pnl = self.cash + self.inventory * self.mid_price
        reward = pnl - self.kappa * (self.inventory ** 2)

        obs = self._get_obs()
        info = {"pnl": pnl, "inventory": self.inventory}
        return obs, reward, done, False, info