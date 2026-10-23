import unittest

import torch

from env import distance_map
from exit_env import ExitConfig, ExitEnv


class ClosedNasseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.cfg = ExitConfig(T=24, first_open_min=3, first_open_max=3, response_min=2, response_max=2,
                              response_probability_min=1, response_probability_max=1, comm_drop=0)
        self.env = ExitEnv(self.cfg, B=2, seed=7)

    def test_initially_closed_every_scenario(self):
        for scenario in ("sealed", "delayed", "responsive", "relocating"):
            obs = self.env.reset(scenario)
            self.assertFalse(self.env.opened.any())
            self.assertTrue(torch.isfinite(obs).all())
            self.assertEqual(obs.shape, (2, self.cfg.n_crowd, self.env.obs_dim))

    def test_sealed_cannot_be_escaped_by_random_actions(self):
        e = self.env
        e.reset("sealed")
        for _ in range(e.cfg.T):
            e.step(torch.randint(7, e.alive.shape))
        self.assertTrue(e.alive.all())
        self.assertFalse(e.crossed.any())
        self.assertFalse(e.opened.any())

    def place_near_gate(self):
        e = self.env
        for b in range(e.B):
            gate = e.gates[b, 0]
            positions = gate + e.dirs[1:]
            valid = ~e.walls[b, positions[:, 0], positions[:, 1]]
            valid &= (positions - e.cfg.H//2).abs().max(-1).values < e.cfg.ring
            e.pc[b, 0] = positions[valid][0]

    def test_request_delay_and_no_spontaneous_response(self):
        e = self.env
        e.reset("responsive")
        self.place_near_gate()
        stay = torch.zeros_like(e.alive, dtype=torch.long)
        e.step(stay)
        self.assertFalse(e.opened.any())
        request = stay.clone()
        request[:, 0] = 5
        e.step(request)
        self.assertFalse(e.opened.any())
        e.step(stay)
        self.assertTrue(e.opened[:, 0].all())

    def test_refusal_remains_possible(self):
        self.cfg.response_probability_min = self.cfg.response_probability_max = 0
        e = self.env
        e.reset("responsive")
        self.place_near_gate()
        for _ in range(12):
            e.step(torch.full_like(e.alive, 5, dtype=torch.long))
        self.assertFalse(e.opened.any())

    def test_no_hidden_scenario_in_observation(self):
        a, b = ExitEnv(self.cfg, 2, 19), ExitEnv(self.cfg, 2, 19)
        self.assertTrue(torch.equal(a.reset("sealed"), b.reset("delayed")))

    def test_gate_capacity_and_no_teleport(self):
        e = self.env
        e.reset("delayed")
        self.place_near_gate()
        e.pc[:, 1:3] = e.pc[:, :1]
        e.t = 3
        e._barriers()
        actions = torch.zeros_like(e.alive, dtype=torch.long)
        for b in range(e.B):
            direction = e.gates[b, 0] - e.pc[b, 0]
            action = (e.dirs == direction).all(-1).nonzero()[0, 0]
            actions[b, :3] = action
        before = e.pc.clone()
        e.step(actions)
        self.assertTrue(((e.pc - before).abs().sum(-1) <= 1).all())
        self.assertTrue(((e.pc[:, :3] == e.gates[:, :1]).all(-1).sum(1) == 1).all())

    def test_broadcast_arrives_after_delay_only(self):
        e = self.env
        e.reset("delayed")
        self.place_near_gate()
        e.t = 3
        e._barriers()
        actions = torch.zeros_like(e.alive, dtype=torch.long)
        actions[:, 0] = 6
        e.step(actions)
        self.assertTrue((e.message_seen < 0).all())
        e.step(torch.zeros_like(actions))
        self.assertTrue((e.message_seen < 0).all())
        e.step(torch.zeros_like(actions))
        self.assertTrue((e.message_seen == 3).all())

    def test_delayed_opening_really_connects_to_outside(self):
        e = self.env
        e.reset("delayed")
        e.t = 3
        e._barriers()
        dist = distance_map(~e.walls & ~e.blocked, e.border.expand_as(e.walls))
        self.assertTrue(torch.isfinite(e.at(dist, e.pc)).all())


if __name__ == "__main__":
    unittest.main()
