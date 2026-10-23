import unittest

import torch

from exit_env import ExitConfig, RotatedEnv


class RotationTests(unittest.TestCase):
    def test_rotation_preserves_episode_and_closed_cordon(self):
        torch.set_num_threads(2)
        cfg = ExitConfig(T=8)
        a, b = RotatedEnv(cfg, 2, 41), RotatedEnv(cfg, 2, 41, turns=1)
        a.reset("delayed")
        b.reset("delayed")
        self.assertTrue(torch.equal(torch.rot90(a.walls, 1, (1, 2)), b.walls))
        self.assertTrue(torch.equal(torch.rot90(a.blocked, 1, (1, 2)), b.blocked))
        expected = torch.stack((cfg.H-1-a.pc[..., 1], a.pc[..., 0]), -1)
        self.assertTrue(torch.equal(expected, b.pc))
        self.assertTrue(torch.equal(a.first, b.first))
        self.assertTrue(torch.equal(a.accept, b.accept))
        self.assertTrue(torch.equal(a.priorities, b.priorities))
        c = RotatedEnv(cfg, 2, 41, turns=4)
        c.reset("delayed")
        self.assertTrue(torch.equal(a.obs(), c.obs()))


if __name__ == "__main__":
    unittest.main()
