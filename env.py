"""Environnement grille : police contre foule, en lots (plusieurs parties en parallele).

Le quartier est une grille de rues (une rue toutes les `spacing` cases), avec des
troncons supprimes au hasard pour creer des impasses. Le bord de la carte = sorties.

Foule : `n_crowd` groupes. Ils gagnent un peu a chaque tour passe dans la zone de
manif (le carre autour du centre), perdent si leur case est trop dense (risque
d'ecrasement) et perdent beaucoup s'ils sont encercles a la fin.
Police : `n_police` unites. Une case occupee par la police bloque la foule.
La police gagne pour chaque groupe encercle a la fin.

Encercle = depuis sa case, le groupe ne peut plus rejoindre le bord sans passer
par une case tenue par la police.

Volontairement absent : tout contact physique avec les lignes. La foule ne peut
que se deplacer, attendre, se regrouper ou se disperser.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# immobile, haut, bas, gauche, droite
DIRS = torch.tensor([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])


@dataclass
class Config:
    carte: str = "grille"  # "grille" (quartier invente) ou "paris" (rues OSM)
    # "arrivee" : la police arrive de loin et boucle ; "progressive" : elle est deja
    # en place, laisse passer, puis ferme les sorties une a une (observatoire LDH)
    fermeture: str = "arrivee"
    H: int = 25
    spacing: int = 4
    n_crowd: int = 24
    n_police: int = 12
    T: int = 48
    demo_radius: int = 4
    police_min_dist: int = 9
    cap: int = 3
    obs_r: int = 7
    remove_frac: float = 0.15
    presence_reward: float = 0.05
    density_coef: float = 0.1
    kettle_penalty: float = 10.0
    # potentiel = -shaping_coef * distance a la sortie libre la plus proche (plafonnee)
    shaping_coef: float = 0.05
    shaping_cap: int = 40
    gamma: float = 0.99
    police_kettle_reward: float = 0.25
    police_escape_penalty: float = 0.05
    police_enclose_reward: float = 0.005
    # cout du temps passe nasse (par tour), et passage par le filtre (fouille, controle)
    enclosed_cost: float = 0.0
    filter_cost: float = 0.5
    drip_every: int = 3
    # "zone" : la manif compte sur la place de depart ; "rassemblement" : partout ou au
    # moins gather_min groupes sont reunis (rayon gather_r), hors bord de carte (be water)
    presence_mode: str = "zone"
    gather_r: int = 2
    gather_min: int = 6
    edge_margin: int = 3
    # cohesion (modele ESIM) : un groupe isole paie, d'autant plus que la menace monte
    cohesion_coef: float = 0.0
    iso_min: int = 2
    # retard de l'info qui circule (sorties fermees, plus gros rassemblement), en tours
    comm_delay: int = 0
    # echelle : personnes par groupe ; cap groupes par case de 100 m2 = cap*130/100 pers/m2
    group_size: int = 130


def paris_config(**kw):
    """Reglages pour les vrais quartiers : 1 case = 10 m, 1 tour = environ 10 s de marche."""
    base = dict(carte="paris", H=65, demo_radius=6, police_min_dist=12, T=96,
                shaping_cap=80, n_crowd=48, n_police=70)
    base.update(kw)
    return Config(**base)


def map_source(cfg):
    """Fonction rng -> carte (True = mur) pour l'entrainement."""
    if cfg.carte == "grille":
        return lambda rng: make_map(cfg, rng)
    from osm_map import CropSampler
    sampler = CropSampler(cfg.H, cfg.demo_radius)
    c = cfg.H // 2

    def draw(rng):
        while True:
            street = sampler(rng)
            if street[c, c] and _connected_to_border(street, (c, c)):
                return ~street
    return draw


def _connected_to_border(street, start):
    H = street.shape[0]
    seen = np.zeros_like(street)
    stack = [start]
    seen[start] = True
    while stack:
        y, x = stack.pop()
        if y in (0, H - 1) or x in (0, H - 1):
            return True
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < H and street[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    return False


def make_map(cfg, rng):
    """Renvoie un tableau booleen : True = mur (pas une rue)."""
    H, s = cfg.H, cfg.spacing
    c = H // 2
    while True:
        street = np.zeros((H, H), bool)
        street[::s, :] = True
        street[:, ::s] = True
        for r in range(0, H, s):
            for x in range(0, H - 1, s):
                if rng.random() < cfg.remove_frac:
                    street[r, x + 1:x + s] = False
        for x in range(0, H, s):
            for r in range(0, H - 1, s):
                if rng.random() < cfg.remove_frac:
                    street[r + 1:r + s, x] = False
        if street[c, c] and _connected_to_border(street, (c, c)):
            return ~street


def shift4(m):
    """Cases voisines (4 directions) d'un masque (B,H,W)."""
    out = torch.zeros_like(m)
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    out[:, :, 1:] |= m[:, :, :-1]
    out[:, :, :-1] |= m[:, :, 1:]
    return out


def distance_map(free, targets, max_iter=200):
    """Distance en pas (par les cases libres) jusqu'aux cases cibles. inf si inatteignable."""
    dist = torch.full(free.shape, float("inf"))
    reached = targets & free
    dist[reached] = 0
    for k in range(1, max_iter):
        new = shift4(reached) & free & ~reached
        if not new.any():
            break
        dist[new] = k
        reached |= new
    return dist


class NasseEnv:
    def __init__(self, cfg: Config, B: int, seed: int = 0, map_fn=None):
        self.cfg, self.B = cfg, B
        self.rng = np.random.default_rng(seed)
        self.map_fn = map_fn or map_source(cfg)
        # tout suit le peripherique par defaut (torch.set_default_device) : CPU ou carte graphique
        self.dev = torch.get_default_device()
        self.dirs = DIRS.to(self.dev)
        H = cfg.H
        self.border = torch.zeros(H, H, dtype=torch.bool)
        self.border[0, :] = self.border[-1, :] = True
        self.border[:, 0] = self.border[:, -1] = True
        self.center = H // 2
        self.bidx = torch.arange(B)
        k = 2 * cfg.obs_r + 1
        self.win = torch.arange(k)
        self.obs_dim = k * k * 4 + 11
        m = cfg.edge_margin
        self.edge = torch.zeros(cfg.H, cfg.H, dtype=torch.bool, device=self.dev)
        self.edge[:m] = self.edge[-m:] = True
        self.edge[:, :m] = self.edge[:, -m:] = True
    # ---------- utilitaires ----------
    def _cheb(self, pos):
        return (pos - self.center).abs().max(-1).values

    def occ(self, pos, mask=None):
        B, H = self.B, self.cfg.H
        flat = torch.zeros(B, H * H)
        idx = pos[..., 0] * H + pos[..., 1]
        w = torch.ones(idx.shape) if mask is None else mask.float()
        flat.scatter_add_(1, idx, w)
        return flat.view(B, H, H)

    def at(self, grid, pos):
        """Valeur de grid (B,H,W) a chaque position (B,N,2)."""
        return grid[self.bidx[:, None], pos[..., 0], pos[..., 1]]

    def _sample_cells(self, street, cond, n):
        ys, xs = np.nonzero(street & cond)
        pick = self.rng.choice(len(ys), size=n, replace=len(ys) < n)
        return np.stack([ys[pick], xs[pick]], -1)

    # ---------- partie ----------
    def reset(self, walls=None):
        cfg, H = self.cfg, self.cfg.H
        if walls is None:
            walls = np.stack([self.map_fn(self.rng) for _ in range(self.B)])
        yy, xx = np.mgrid[:H, :H]
        cheb = np.maximum(abs(yy - self.center), abs(xx - self.center))
        pc, pp = [], []
        for w in walls:
            street = ~w
            pc.append(self._sample_cells(street, cheb <= cfg.demo_radius, cfg.n_crowd))
            ring = (cheb >= cfg.police_min_dist) & (cheb < H // 2)
            pp.append(self._sample_cells(street, ring, cfg.n_police))
        self.walls = torch.from_numpy(walls).to(self.dev)
        self.pc = torch.from_numpy(np.stack(pc)).long().to(self.dev)
        self.pp = torch.from_numpy(np.stack(pp)).long().to(self.dev)
        self.alive = torch.ones(self.B, cfg.n_crowd, dtype=torch.bool)
        self.t = 0
        # pilotes par la police scriptee (mode progressif) ; par defaut tout bloque
        self.police_block = torch.ones(self.B, cfg.n_police, dtype=torch.bool)
        self.frac_open = torch.ones(self.B)
        self.first_close = torch.full((self.B,), float("inf"))
        self.drip_on = torch.zeros(self.B, dtype=torch.bool)
        self.filter_cell = torch.zeros(self.B, 2, dtype=torch.long)
        self._update_exit_dist()
        self._update_gather()
        self._hist = []
        self._push_hist()
        return self.obs()

    def box(self, grid, r):
        """Somme sur le carre de rayon r autour de chaque case."""
        k = 2 * r + 1
        return F.avg_pool2d(grid[:, None].float(), k, 1, r, count_include_pad=True)[:, 0] * (k * k)

    def _update_gather(self):
        """Rassemblements : groupes libres (non nasses) reunis, hors bord de carte."""
        cfg = self.cfg
        free_groups = self.alive & ~self.enclosed
        nb = self.box(self.occ(self.pc, free_groups), cfg.gather_r)
        nb = nb.masked_fill(self.edge, 0)
        self.gather_grid = nb
        self.gathered = free_groups & (self.at(nb, self.pc) >= cfg.gather_min)
        flat = nb.flatten(1)
        v, i = flat.max(1)
        self.gather_center = torch.stack([i // cfg.H, i % cfg.H], -1)
        self.gather_size = v
        # nombre de rassemblements distincts : zones connexes ou le seuil est atteint
        mask = nb >= cfg.gather_min
        lab = torch.where(mask, torch.arange(1, cfg.H * cfg.H + 1, device=self.dev).view(cfg.H, cfg.H).float(),
                          torch.zeros((), device=self.dev))
        for _ in range(2 * cfg.H):
            new = torch.where(mask, F.max_pool2d(lab[:, None], 3, 1, 1)[:, 0], lab)
            if torch.equal(new, lab):
                break
            lab = new
        peak = mask & (lab == torch.arange(1, cfg.H * cfg.H + 1, device=self.dev).view(cfg.H, cfg.H).float())
        self.n_gatherings = peak.flatten(1).sum(1).float()

    def _push_hist(self):
        self._hist.append((self.frac_open.clone(), self.first_close.clone(),
                           self.gather_center.clone(), self.gather_size.clone()))
        if len(self._hist) > self.cfg.comm_delay + 1:
            self._hist.pop(0)

    def refresh(self):
        """A appeler si la police a ete repositionnee apres reset (mode progressif)."""
        self._update_exit_dist()
        self._update_gather()
        self._hist = []
        self._push_hist()
        return self.obs()

    def occ_block(self):
        """Cases tenues par une unite qui bloque vraiment (pas un cordon qui laisse passer)."""
        return self.occ(self.pp, self.police_block)

    def _filter_zone(self):
        """Cases d'ou l'on peut passer le filtre : a cote du point de sortie au compte-gouttes."""
        z = torch.zeros(self.B, self.cfg.H, self.cfg.H, dtype=torch.bool)
        b = self.drip_on.nonzero().flatten()
        if len(b):
            z[b, self.filter_cell[b, 0], self.filter_cell[b, 1]] = True
            z = shift4(z) | z
        return z

    def _update_exit_dist(self):
        """Distance de chaque groupe a la sortie libre la plus proche (inf = encercle)."""
        free = ~self.walls & ~(self.occ_block() > 0)
        border = self.border.expand(self.B, -1, -1)
        d_border = distance_map(free, border)
        self.enclosed = self.alive & torch.isinf(self.at(d_border, self.pc))
        # la distance guide aussi vers le filtre quand il est la seule issue
        self.dist_grid = distance_map(free, border | self._filter_zone()) if self.drip_on.any() else d_border
        self.exit_dist = self.at(self.dist_grid, self.pc)

    def _potential(self):
        cfg = self.cfg
        d = self.exit_dist.clamp(max=cfg.shaping_cap)
        return -cfg.shaping_coef * d * self.alive

    def obs(self):
        cfg, r = self.cfg, self.cfg.obs_r
        # un cordon qui laisse passer se voit, mais moins qu'un cordon ferme
        occ_p = self.occ_block() + 0.3 * self.occ(self.pp, ~self.police_block)
        occ_c = self.occ(self.pc, self.alive)
        chans = torch.stack([
            self.walls.float(),
            occ_p / 3,
            occ_c / 6,
            self.border.float().expand(self.B, -1, -1),
        ], -1)  # (B,H,W,C)
        padded = torch.zeros(self.B, cfg.H + 2 * r, cfg.H + 2 * r, 4)
        padded[..., 0] = 1.0  # hors carte = mur
        padded[:, r:-r, r:-r] = chans

        def window(pos):
            ys = pos[..., 0, None, None] + self.win[None, None, :, None]
            xs = pos[..., 1, None, None] + self.win[None, None, None, :]
            w = padded[self.bidx[:, None, None, None], ys, xs]  # (B,N,k,k,C)
            return w.flatten(2)

        def scalars(pos, alive):
            half = cfg.H / 2
            # l'info qui circule arrive avec comm_delay tours de retard
            fo, fc, gc, gs = self._hist[0]
            return torch.stack([
                torch.full(pos.shape[:2], self.t / cfg.T),
                (pos[..., 0] - self.center) / half,
                (pos[..., 1] - self.center) / half,
                (self._cheb(pos) <= cfg.demo_radius).float(),
                alive.float(),
                (self.at(self.dist_grid, pos).clamp(max=cfg.shaping_cap) / cfg.shaping_cap),
                # ce qui circule par telephone : part des sorties encore ouvertes,
                # temps ecoule depuis la premiere fermeture
                fo[:, None].expand(pos.shape[:2]),
                ((self.t - fc) / cfg.T).clamp(0, 1)[:, None].expand(pos.shape[:2]),
                # ou se trouve le plus gros rassemblement, et sa taille
                (gc[:, None, 0] - pos[..., 0]) / half,
                (gc[:, None, 1] - pos[..., 1]) / half,
                (gs / cfg.n_crowd)[:, None].expand(pos.shape[:2]),
            ], -1)

        oc = torch.cat([window(self.pc), scalars(self.pc, self.alive)], -1)
        op = torch.cat([window(self.pp), scalars(self.pp, torch.ones(self.pp.shape[:2], dtype=torch.bool))], -1)
        return oc, op

    def kettled(self):
        free = ~self.walls & ~(self.occ_block() > 0)
        reach = self.border & free
        while True:
            new = shift4(reach) & free & ~reach
            if not new.any():
                break
            reach |= new
        return self.alive & ~self.at(reach, self.pc)

    def step(self, ac, ap):
        cfg, H = self.cfg, self.cfg.H
        phi0 = self._potential()
        # la police bouge d'abord ; elle ne peut pas entrer dans une case occupee par la foule
        occ_c = self.occ(self.pc, self.alive)
        prop = (self.pp + self.dirs[ap]).clamp(0, H - 1)
        ok = ~self.at(self.walls, prop) & ~(self.at(occ_c, prop) > 0)
        self.pp = torch.where(ok[..., None], prop, self.pp)
        occ_p = self.occ_block()
        # puis la foule ; elle ne peut pas entrer dans une case tenue par la police
        prop = (self.pc + self.dirs[ac]).clamp(0, H - 1)
        ok = ~self.at(self.walls, prop) & ~(self.at(occ_p, prop) > 0) & self.alive
        self.pc = torch.where(ok[..., None], prop, self.pc)

        escaped = self.alive & self.at(self.border.expand(self.B, -1, -1), self.pc)
        self.alive = self.alive & ~escaped
        # compte-gouttes : tous les drip_every tours, un groupe colle au filtre passe
        filtered = torch.zeros_like(self.alive)
        if self.drip_on.any() and self.t % cfg.drip_every == 0:
            near = self.alive & self.at(self._filter_zone(), self.pc)
            first = near.float().argmax(1)
            pick = near.any(1)
            filtered[pick.nonzero().flatten(), first[pick]] = True
            self.alive = self.alive & ~filtered
        occ_c = self.occ(self.pc, self.alive)
        over = (self.at(occ_c, self.pc) - cfg.cap).clamp(min=0) * self.alive
        in_demo = (self._cheb(self.pc) <= cfg.demo_radius) & self.alive

        rc = -cfg.density_coef * over - cfg.filter_cost * filtered
        rp = (-cfg.police_escape_penalty * escaped.sum(1, keepdim=True).float()).expand(-1, cfg.n_police).clone()
        self._update_exit_dist()
        enclosed = self.enclosed
        self._update_gather()
        # une nasse n'est plus une manif : la presence ne compte plus une fois encercle
        tenu = self.gathered if cfg.presence_mode == "rassemblement" else in_demo & ~enclosed
        rc = rc + cfg.presence_reward * tenu - cfg.enclosed_cost * enclosed
        if cfg.cohesion_coef:
            others = self.at(self.box(self.occ(self.pc, self.alive), cfg.gather_r), self.pc) - 1
            isolated = self.alive & ~enclosed & (others < cfg.iso_min)
            threat = 1 - self.frac_open
            rc = rc - cfg.cohesion_coef * isolated * (1 + 2 * threat[:, None])
        else:
            isolated = torch.zeros_like(enclosed)
        # faconnage par potentiel (ne change pas la meilleure strategie) : un signal
        # des que la nasse se ferme au lieu d'attendre la fin de la partie
        rc = rc + cfg.gamma * self._potential() - phi0
        rp = rp + cfg.police_enclose_reward * enclosed.sum(1, keepdim=True).float()
        self.t += 1
        done = self.t >= cfg.T
        info = {"escaped": escaped | filtered, "filtered": filtered, "over": over, "in_demo": in_demo,
                "enclosed": enclosed, "tenu": tenu, "isolated": isolated, "n_gatherings": self.n_gatherings}
        if done:
            k = enclosed
            rc = rc - cfg.kettle_penalty * k
            rp = rp + cfg.police_kettle_reward * k.sum(1, keepdim=True).float()
            info["kettled"] = k
        self._push_hist()
        return self.obs(), rc, rp, done, info
