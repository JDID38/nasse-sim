"""Regression : une nasse mobile ne disparait pas du fait de sa fermeture."""
import unittest

import torch

from env import Config, NasseEnv
from police_script import ScriptedPolice


class MobileContainmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        cfg = Config(H=25, n_crowd=6, n_police=4, demo_radius=2,
                     fermeture="progressive", presence_mode="rassemblement")
        self.env = e = NasseEnv(cfg, 2, 0)
        e.reset()
        self.police = p = ScriptedPolice(e, (2,), pursuit_k=1)
        p.reset()
        # Couloir a deux sorties, regroupement exterieur a l'anneau initial.
        e.walls[:] = True
        e.walls[:, 4, :] = False
        e.pc[:] = torch.tensor([4, 6])
        e.alive[:] = True
        p.squad[:] = True
        p.unit_cut[:] = -1
        p.squad_target[:] = -1
        e.pp[:] = torch.tensor([4, 20])
        e.police_block[:] = False
        e.refresh()
        p._pursue()
        has = (p.squad_target >= 0).all(-1)
        e.pp[has] = p.squad_target[has]
        p._update_env()
        e.refresh()

    def test_closed_cordon_is_retained_without_presence_reward(self):
        e, p = self.env, self.police
        self.assertTrue(e.enclosed.all())
        self.assertFalse(e.gathered.any())
        before = p.squad_target.clone()
        for _ in range(3):
            p._pursue()
            p._update_env()
            e.refresh()
            self.assertTrue(torch.equal(before, p.squad_target))
            self.assertTrue(e.enclosed.all())
            self.assertFalse(e.gathered.any())

    def test_empty_cordon_releases_only_its_own_batch(self):
        e, p = self.env, self.police
        before = p.squad_target[1].clone()
        e.alive[0] = False
        e.refresh()
        p._pursue()
        p._update_env()
        self.assertTrue((p.squad_target[0] == -1).all())
        self.assertFalse(e.police_block[0].any())
        self.assertTrue(torch.equal(before, p.squad_target[1]))
        self.assertTrue(e.police_block[1].any())


if __name__ == "__main__":
    unittest.main()
