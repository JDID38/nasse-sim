"""Police scriptee : sert d'adversaire competent et de reference de calibrage.

Elle vise les "points de coupure" : les cases de rue sur l'anneau juste a
l'exterieur de la zone de manif (distance demo_radius + 2 du centre). Les tenir
toutes ferme la zone. Chaque unite prend le point libre le plus proche, puis y va
par le plus court chemin.

Mode "progressive" (cfg.fermeture) : comme decrit par l'Observatoire parisien des
libertes publiques, le dispositif est deja en place autour de la manif et laisse
passer. Puis les sorties se ferment une a une, a des moments tires au hasard
(premiere fermeture, puis fenetre de fermeture). Une fois tout ferme, un seul point
laisse sortir au compte-gouttes.
"""
import torch

from env import distance_map


def moves_toward(env, pos, dist):
    H = env.cfg.H
    cand = (pos[:, :, None, :] + env.dirs[None, None]).clamp(0, H - 1)
    d = dist[env.bidx[:, None, None], torch.arange(pos.shape[1])[None, :, None], cand[..., 0], cand[..., 1]]
    return d.argmin(-1)


class ScriptedPolice:
    """ring_extra : distances possibles de l'anneau au-dela de la zone de manif,
    tirees au hasard pour chaque partie (varier la tactique). Par defaut : 2."""

    def __init__(self, env, ring_extra=(2,), seed=0, first=(0.2, 0.4), window=(0.15, 0.35),
                 pursuit_k=3, decide_every=3, pursuit_min=4):
        """first, window : premiere fermeture et duree de la fermeture complete,
        en fraction de T (mode progressif). window=(0, 0) : tout ferme d'un coup.

        En mode rassemblement (be water), les unites qui ne tiennent pas l'anneau forment
        une reserve mobile : tous les decide_every tours, elle repere jusqu'a pursuit_k
        regroupements d'au moins pursuit_min groupes hors de l'anneau et va les encercler.
        Une unite de reserve ne bloque qu'une fois en position."""
        self.env, self.ring_extra = env, ring_extra
        self.first, self.window = first, window
        self.pursuit_k, self.decide_every, self.pursuit_min = pursuit_k, decide_every, pursuit_min
        self.rng = torch.Generator(device=env.dev).manual_seed(seed)

    def reset(self):
        env, H = self.env, self.env.cfg.H
        B, Np = env.B, env.cfg.n_police
        extra = torch.tensor(self.ring_extra)
        ring = env.cfg.demo_radius + extra[torch.randint(len(extra), (B,), generator=self.rng)]
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(H), indexing="ij")
        cheb = torch.maximum((yy - env.center).abs(), (xx - env.center).abs())
        cut = (cheb[None] == ring[:, None, None]) & ~env.walls  # (B,H,W)
        self.n_cuts = cut.flatten(1).sum(1)
        # chemins par l'exterieur : traverser la manif bloquerait l'unite derriere la foule
        street = ~env.walls & (cheb[None] >= ring[:, None, None])
        targets = torch.full((B, Np, H, H), float("inf"))
        K = int(self.n_cuts.max()) if B else 0
        self.cut_cells = torch.zeros(B, max(K, 1), 2, dtype=torch.long)
        self.unit_cut = torch.full((B, Np), -1, dtype=torch.long)
        primary = torch.zeros(B, Np, dtype=torch.bool)
        for b in range(B):
            cells = cut[b].nonzero().tolist()
            if not cells:
                continue
            masks = torch.zeros(len(cells), H, H, dtype=torch.bool)
            for i, (y, x) in enumerate(cells):
                masks[i, y, x] = True
            self.cut_cells[b, :len(cells)] = torch.tensor(cells)
            dm = distance_map(street[b].expand(len(cells), -1, -1), masks)  # (K,H,W)
            pp = env.pp[b]
            d = dm[:, pp[:, 0], pp[:, 1]].T.clone()  # (Np,K)
            # affectation gloutonne : la paire unite/point la plus proche d'abord
            d0 = d.clone()
            for _ in range(min(Np, len(cells))):
                i = int(d.argmin())
                p, k = divmod(i, len(cells))
                if d[p, k] == float("inf"):
                    break
                targets[b, p] = dm[k]
                self.unit_cut[b, p] = k
                primary[b, p] = True
                d[p, :] = float("inf")
                d[:, k] = float("inf")
            # unites en trop : renfort sur le point le plus proche, pas de barrage au hasard
            # (en mode rassemblement, elles forment la reserve mobile)
            free_units = torch.isinf(targets[b].flatten(1).min(1).values)
            if env.cfg.presence_mode == "rassemblement":
                continue
            for p in free_units.nonzero().flatten().tolist():
                k = int(d0[p].argmin())
                if d0[p, k] < float("inf"):
                    targets[b, p] = dm[k]
                    self.unit_cut[b, p] = k
        self.dist = targets
        self.ring = ring
        self.squad = (self.unit_cut < 0) & (env.cfg.presence_mode == "rassemblement")
        self.squad_target = torch.full((B, Np, 2), -1, dtype=torch.long)
        self.progressive = env.cfg.fermeture == "progressive"
        if self.progressive:
            T = env.cfg.T
            # chaque point a deja son unite, en place des le debut
            pos = self.cut_cells[env.bidx[:, None], self.unit_cut.clamp(min=0)]
            env.pp = torch.where(primary[..., None], pos, env.pp)
            u = lambda lo, hi, shape: lo + (hi - lo) * torch.rand(shape, generator=self.rng)
            t0 = u(*self.first, (B, 1)) * T
            w = u(*self.window, (B, 1)) * T
            ct = t0 + w * torch.rand(B, max(K, 1), generator=self.rng)
            ct[torch.arange(B), 0] = t0[:, 0]  # un des points ferme pile a t0
            ct = ct.gather(1, torch.argsort(torch.rand(B, max(K, 1), generator=self.rng), 1))
            valid = torch.arange(max(K, 1))[None] < self.n_cuts[:, None]
            self.close_t = torch.where(valid, ct.round(), torch.full_like(ct, float("inf")))
            self.full_t = torch.where(valid, self.close_t, torch.zeros_like(ct)).max(1).values
            self.first_t = self.close_t.min(1).values
            # le point de sortie au compte-gouttes, une fois tout ferme
            pick = (torch.rand(B, generator=self.rng) * self.n_cuts.clamp(min=1)).long()
            env.filter_cell = self.cut_cells[torch.arange(B), pick]
            self._update_env()

    def _update_env(self):
        env = self.env
        t = env.t
        closed = env.t >= self.close_t  # (B,K)
        k = self.unit_cut.clamp(min=0)
        env.police_block = (self.unit_cut >= 0) & closed.gather(1, k)
        if self.squad.any():
            # une unite de reserve bloque une fois arrivee sur sa case cible
            arrived = (env.pp == self.squad_target).all(-1) & self.squad
            env.police_block = env.police_block | arrived
        n = self.n_cuts.clamp(min=1).float()
        env.frac_open = 1 - (closed & torch.isfinite(self.close_t)).sum(1).float() / n
        env.first_close = torch.where(t >= self.first_t, self.first_t, torch.full_like(self.first_t, float("inf")))
        env.drip_on = (t >= self.full_t) & (self.n_cuts > 0)

    def _pursue(self):
        """Envoie la reserve encercler jusqu'a pursuit_k regroupements hors de l'anneau.
        Tout est vectorise sur les parties : seule la boucle sur les cases a tenir reste."""
        env, H = self.env, self.env.cfg.H
        B, Np, dev = env.B, env.cfg.n_police, env.dev
        yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(H, device=dev), indexing="ij")
        cheb_c = torch.maximum((yy - env.center).abs(), (xx - env.center).abs())
        # La recompense exclut les groupes nasses, pas l'observation policiere :
        # reutiliser gather_grid ferait abandonner le cordon des sa fermeture.
        nb = env.box(env.occ(env.pc, env.alive), env.cfg.gather_r)
        nb = nb.masked_fill(env.edge, 0)
        nb = nb.masked_fill(cheb_c[None] <= self.ring[:, None, None], 0)
        d_p = env.cfg.gather_r + 2  # anneau de l'encerclement autour du regroupement
        bidx = torch.arange(B, device=dev)
        # 1. les pursuit_k plus gros regroupements, avec suppression autour de chacun
        rings, counts = [], []
        for _ in range(self.pursuit_k):
            v, i = nb.flatten(1).max(1)
            cy, cx = i // H, i % H
            dc = torch.maximum((yy[None] - cy[:, None, None]).abs(), (xx[None] - cx[:, None, None]).abs())
            ok = (v >= self.pursuit_min)[:, None, None]
            ring_c = (dc == d_p) & ~env.walls & ok
            rings.append(ring_c.flatten(1))
            counts.append(ring_c.flatten(1).sum(1))
            nb = nb.masked_fill(dc <= 2 * d_p, 0)
        # cases de chaque anneau, rangees en tete (tri stable du masque)
        order = [torch.sort((~r).to(torch.int8), dim=1, stable=True).indices for r in rings]
        # 2. affectation gloutonne : on alterne les regroupements, case par case, et chaque
        #    case prend l'unite de reserve libre la plus proche (distance de Manhattan)
        free = self.squad.clone()
        new_t = torch.full((B, Np, 2), -1, dtype=torch.long, device=dev)
        py, px = env.pp[..., 0], env.pp[..., 1]
        steps = int(max(int(c.max()) for c in counts)) if counts else 0
        for rank in range(steps):
            for j in range(len(rings)):
                valid = (rank < counts[j]) & free.any(1)
                if not valid.any():
                    continue
                cell = order[j][:, min(rank, H * H - 1)]
                y, x = cell // H, cell % H
                d = (py - y[:, None]).abs() + (px - x[:, None]).abs()
                d = d.masked_fill(~free, 10 ** 6)
                u = d.argmin(1)
                pick = valid
                new_t[bidx[pick], u[pick]] = torch.stack([y, x], -1)[pick]
                free[bidx[pick], u[pick]] = False
        # 3. chemins : seulement pour les unites dont la cible a change
        changed = (new_t != self.squad_target).any(-1) & self.squad
        self.squad_target = new_t
        has = (new_t >= 0).all(-1)
        upd = changed & has
        bi, ui = upd.nonzero(as_tuple=True)
        if len(bi):
            street = ~env.walls & ~(env.occ(env.pc, env.alive) > 0)
            masks = torch.zeros(len(bi), H, H, dtype=torch.bool, device=dev)
            t = new_t[bi, ui]
            masks[torch.arange(len(bi), device=dev), t[:, 0], t[:, 1]] = True
            self.dist[bi, ui] = distance_map(street[bi] | masks, masks)
        self.dist[self.squad & ~has] = float("inf")

    def act(self, op=None):
        if self.progressive:
            self._update_env()
        if self.squad.any() and self.env.t > 0 and self.env.t % self.decide_every == 0:
            self._pursue()
            self._update_env()
        a = moves_toward(self.env, self.env.pp, self.dist)
        here = self.dist[self.env.bidx[:, None], torch.arange(self.env.cfg.n_police)[None], self.env.pp[..., 0], self.env.pp[..., 1]]
        a[(here == 0) | (here == float("inf"))] = 0
        return a
