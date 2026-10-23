import unittest

import torch

from exit_env import ExitConfig, ExitEnv
from train_exit import make_policy, rotate_observation


class ExitV2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg = ExitConfig(T=12, visible_contacts=False, random_orientation=True,
                              exit_target="cordon", outside_hold=3)

    def test_future_contacts_are_not_in_observation(self):
        env = ExitEnv(self.cfg, 2, 11)
        obs = env.reset("sealed")
        local = obs[..., :-8].reshape(2, self.cfg.n_crowd, -1, 5)
        self.assertEqual(local[..., 3].count_nonzero().item(), 0)
        self.assertGreater(local[..., 1].count_nonzero().item(), 0)

    def test_policy_has_no_fixed_direction_under_rotation(self):
        env = ExitEnv(self.cfg, 2, 17)
        obs = env.reset("sealed")
        # Controle aussi les vecteurs de message, absents au reset.
        obs[..., -4:-2] = torch.rand_like(obs[..., -4:-2]) - .5
        net = make_policy(self.cfg, "c4")
        with torch.no_grad():
            logits, value = net(obs)
            rotated, rotated_value = net(rotate_observation(obs, self.cfg.obs_r, self.cfg.H))
        permutation = torch.tensor([0, 3, 4, 2, 1, 5, 6])
        torch.testing.assert_close(logits, rotated[..., permutation], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(value, rotated_value, atol=1e-6, rtol=1e-5)

    def test_three_consecutive_outside_steps_reward_once(self):
        env = ExitEnv(self.cfg, 2, 12)
        env.reset("sealed")
        original = env.pc[:, 0].clone()
        outside = []
        for b in range(env.B):
            candidates = env.gates[b, 0] + env.dirs[1:]
            valid = ~env.walls[b, candidates[:, 0], candidates[:, 1]]
            valid &= (candidates-self.cfg.H//2).abs().max(-1).values > self.cfg.ring
            outside.append(candidates[valid][0])
        outside = torch.stack(outside)
        stay = torch.zeros_like(env.alive, dtype=torch.long)
        env.pc[:, 0] = outside
        env.step(stay)
        env.step(stay)
        self.assertTrue(env.alive[:, 0].all())
        env.pc[:, 0] = original
        env.step(stay)
        self.assertTrue((env.outside_streak[:, 0] == 0).all())
        env.pc[:, 0] = outside
        env.step(stay)
        env.step(stay)
        _, reward, _, info = env.step(stay)
        self.assertTrue(info["escaped"][:, 0].all())
        self.assertTrue((reward[:, 0] > 0).all())
        _, reward, _, info = env.step(stay)
        self.assertFalse(info["escaped"][:, 0].any())
        self.assertTrue((reward[:, 0] == 0).all())

    def test_sealed_v2_cannot_escape(self):
        env = ExitEnv(self.cfg, 2, 13)
        env.reset("sealed")
        for _ in range(self.cfg.T):
            env.step(torch.randint(7, env.alive.shape))
        self.assertTrue(env.alive.all())


if __name__ == "__main__":
    unittest.main()
