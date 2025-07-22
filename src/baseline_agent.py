import numpy as np

class FixedSpreadAgent:
    """
    Always selects action '2'  (index 2 → skew 0 ticks).
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return 2  # center quotes
