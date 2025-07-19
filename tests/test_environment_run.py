import unittest
import numpy as np
import config
from environment import GearEnv

class TestEnvironmentRun(unittest.TestCase):
    def test_long_run(self):
        """Test running the environment for many steps to check for hangs."""
        env = GearEnv(verbose=0)
        state, _ = env.reset()
        
        for i in range(2000):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                state, _ = env.reset()
            if (i + 1) % 100 == 0:
                print(f"Completed step {i + 1}")

if __name__ == '__main__':
    unittest.main()
