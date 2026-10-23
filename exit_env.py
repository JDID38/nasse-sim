"""Sortie non violente depuis une nasse fermee. Hypotheses, pas calibration terrain.

Actions : attendre, N/S/O/E, demander un passage, partager une ouverture vue.
Les lignes fermees sont infranchissables ; aucun contact offensif n'est modelise.
"""
from dataclasses import dataclass

import numpy as np
import torch

from env import Config, DIRS, distance_map, map_source

SCENARIOS = ("sealed", "delayed", "responsive", "relocating")


@dataclass
class ExitConfig:
    carte: str = "grille"
    H: int = 25
    n_crowd: int = 24
    T: int = 144
    ring: int = 8
    spawn_radius: int = 4
    obs_r: int = 7
    gate_count: int = 3
    cap: int = 3
    gate_capacity: int = 1
    first_open_min: int = 12
    first_open_max: int = 48
    open_min: int = 8
    open_max: int = 24
    period_min: int = 32
    period_max: int = 56
    response_min: int = 3
    response_max: int = 12
    response_probability_min: float = 0.1
    response_probability_max: float = 0.7
    comm_delay: int = 3
    comm_drop: float = 0.2
    message_ttl: int = 18
    exit_reward: float = 8.0
    team_exit_reward: float = 2.0
    waiting_cost: float = 0.02
    density_cost: float = 0.25
    blocked_cost: float = 0.02
    request_cost: float = 0.002
    remaining_cost: float = 4.0
    visible_contacts: bool = True  # Compatibilite des checkpoints v1.
    random_orientation: bool = False
    exit_target: str = "map_border"
    outside_hold: int = 3


def component(mask, start):
    seen = np.zeros_like(mask)
    stack = [start]
    seen[start] = True
    H = len(mask)
    while stack:
        y, x = stack.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < H and mask[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    return seen


def neighbors(mask):
    result = np.zeros_like(mask)
    result[1:] |= mask[:-1]
    result[:-1] |= mask[1:]
    result[:, 1:] |= mask[:, :-1]
    result[:, :-1] |= mask[:, 1:]
    return result


class ExitEnv:
    def __init__(self, cfg=None, B=32, seed=0):
        self.cfg = cfg or ExitConfig()
        c = self.cfg
        if c.exit_target not in ("map_border", "cordon") or c.outside_hold < 1:
            raise ValueError("Objectif de sortie ou duree invalide")
        if not (0 < c.spawn_radius < c.ring < c.H // 2):
            raise ValueError("Il faut spawn_radius < ring < H/2")
        if min(c.gate_count, c.gate_capacity, c.comm_delay, c.first_open_min, c.response_min) < 1:
            raise ValueError("Capacites et delais doivent etre strictement positifs")
        self.B, self.dev = B, torch.get_default_device()
        self.rng = np.random.default_rng(seed)
        self.bidx = torch.arange(B)
        self.dirs = DIRS.to(self.dev)
        self.map_fn = map_source(Config(carte=c.carte, H=c.H, demo_radius=c.spawn_radius))
        self.obs_dim = (2 * c.obs_r + 1) ** 2 * 5 + 8
        yy, xx = np.mgrid[:c.H, :c.H]
        self.cheb_np = np.maximum(abs(yy - c.H // 2), abs(xx - c.H // 2))
        self.border = torch.zeros(c.H, c.H, dtype=torch.bool)
        self.border[[0, -1], :] = True
        self.border[:, [0, -1]] = True
        self.cache = {}

    def geometry(self, walls):
        key = walls.tobytes()
        if key in self.cache:
            return self.cache[key]
        c = self.cfg
        street = ~walls
        cut = street & (self.cheb_np == c.ring)
        inner = street & (self.cheb_np < c.ring)
        choices = np.argwhere(inner & (self.cheb_np <= c.spawn_radius))
        if not len(choices):
            raise ValueError("Pas de zone de depart")
        start = tuple(choices[np.argmin(np.abs(choices - c.H // 2).sum(1))])
        interior = component(inner, start)
        # Composantes exterieures reliees au bord de la carte.
        exterior = street & (self.cheb_np > c.ring)
        edge = exterior.copy()
        edge[1:-1, 1:-1] = False
        reachable = np.zeros_like(exterior)
        for point in np.argwhere(edge):
            point = tuple(point)
            if not reachable[point]:
                reachable |= component(exterior, point)
        gates = np.argwhere(cut & neighbors(interior) & neighbors(reachable))
        spawn = np.argwhere(interior & (self.cheb_np <= c.spawn_radius))
        if not len(gates) or not len(spawn):
            raise ValueError("Aucun passage a une case reliant interieur et exterieur")
        result = cut, gates, spawn
        if len(self.cache) < 256:
            self.cache[key] = result
        return result

    def at(self, grid, pos):
        return grid[self.bidx[:, None], pos[..., 0], pos[..., 1]]

    def occupancy(self):
        c = self.cfg
        flat = torch.zeros(self.B, c.H * c.H)
        flat.scatter_add_(1, self.pc[..., 0] * c.H + self.pc[..., 1], self.alive.float())
        return flat.reshape(self.B, c.H, c.H)

    def reset(self, scenario=None, walls=None):
        c, rng = self.cfg, self.rng
        maps, cuts, gates, positions = [], [], [], []
        for b in range(self.B):
            for attempt in range(100):
                wall = np.asarray(walls[b] if walls is not None else self.map_fn(rng), dtype=bool)
                if walls is None and c.random_orientation:
                    wall = np.rot90(wall, int(rng.integers(4))).copy()
                try:
                    cut, possible, spawn = self.geometry(wall)
                    break
                except ValueError:
                    if walls is not None:
                        raise
            else:
                raise RuntimeError("Impossible de generer une nasse valide apres 100 essais")
            selected = possible[rng.choice(len(possible), min(c.gate_count, len(possible)), replace=False)]
            padded = np.full((c.gate_count, 2), -1)
            padded[:len(selected)] = selected
            gates.append(padded)
            positions.append(spawn[rng.choice(len(spawn), c.n_crowd, replace=len(spawn) < c.n_crowd)])
            maps.append(wall)
            cuts.append(cut)
        self.walls = torch.as_tensor(np.stack(maps), device=self.dev)
        self.cut = torch.as_tensor(np.stack(cuts), device=self.dev)
        self.gates = torch.as_tensor(np.stack(gates), device=self.dev, dtype=torch.long)
        self.valid_gates = (self.gates >= 0).all(-1)
        self.pc = torch.as_tensor(np.stack(positions), device=self.dev, dtype=torch.long)
        self.alive = torch.ones(self.B, c.n_crowd, dtype=torch.bool)
        self.t = 0
        kinds = rng.integers(0, len(SCENARIOS), self.B) if scenario is None else np.full(self.B, SCENARIOS.index(scenario))
        self.kinds = torch.as_tensor(kinds, device=self.dev)
        def integers(lo, hi, shape):
            return torch.as_tensor(rng.integers(lo, hi + 1, shape), device=self.dev)
        self.first = integers(c.first_open_min, c.first_open_max, (self.B,))
        self.duration = integers(c.open_min, c.open_max, (self.B,))
        self.period = integers(c.period_min, c.period_max, (self.B,))
        self.response_delay = integers(c.response_min, c.response_max, (self.B,))
        probability = rng.uniform(c.response_probability_min, c.response_probability_max, (1, self.B, 1))
        # Tirages predetermines : le futur et le scenario ne figurent jamais dans obs().
        self.accept = torch.as_tensor(rng.random((c.T, self.B, c.gate_count)) < probability, device=self.dev)
        self.delivery = torch.as_tensor(rng.random((c.T, self.B)) >= c.comm_drop, device=self.dev)
        self.priorities = torch.as_tensor(rng.random((c.T, self.B, c.n_crowd)), device=self.dev)
        self.due = torch.full((self.B, c.gate_count), -1, dtype=torch.long)
        self.request_until = torch.full_like(self.due, -1)
        self.pending_accept = torch.zeros_like(self.valid_gates)
        self.open_until = torch.zeros_like(self.due)
        self.message = torch.zeros(self.B, 2, dtype=torch.long)
        self.message_seen = torch.full((self.B,), -10000, dtype=torch.long)
        self.messages = []
        self.gate_grid = torch.zeros_like(self.walls)
        bi, gi = self.valid_gates.nonzero(as_tuple=True)
        p = self.gates[bi, gi]
        self.gate_grid[bi, p[:, 0], p[:, 1]] = True
        self.crossed = torch.zeros_like(self.alive)
        self.outside_streak = torch.zeros_like(self.pc[..., 0])
        self._barriers()
        # Invariant : tout le monde commence enferme, sans filtration ni sortie magique.
        d = distance_map(~self.walls & ~self.blocked, self.border.expand_as(self.walls), max_iter=c.H * c.H)
        if not torch.isinf(self.at(d, self.pc)).all():
            raise RuntimeError("Episode invalide : depart non enferme")
        return self.obs()

    def _barriers(self):
        c = self.cfg
        t = self.t
        delayed = (self.kinds == 1) & (t >= self.first)
        cycle = ((t - self.first).clamp(min=0) // self.period)
        index = cycle % self.valid_gates.sum(1).clamp(min=1)
        relocating = (self.kinds == 3) & (t >= self.first) & ((t - self.first) % self.period < self.duration)
        opened = (torch.arange(c.gate_count)[None] == 0) & delayed[:, None]
        opened |= (torch.arange(c.gate_count)[None] == index[:, None]) & relocating[:, None]
        opened |= (self.kinds == 2)[:, None] & (t < self.open_until)
        self.opened = opened & self.valid_gates
        self.blocked = self.cut.clone()
        bi, gi = self.opened.nonzero(as_tuple=True)
        p = self.gates[bi, gi]
        self.blocked[bi, p[:, 0], p[:, 1]] = False

    def obs(self):
        c, r = self.cfg, self.cfg.obs_r
        channels = torch.stack((self.walls.float(), self.blocked.float(), self.occupancy() / c.cap,
                                self.gate_grid.float() * c.visible_contacts, self.border.expand_as(self.walls).float()), -1)
        padded = torch.zeros(self.B, c.H + 2*r, c.H + 2*r, 5)
        padded[..., 0] = 1
        padded[:, r:r+c.H, r:r+c.H] = channels
        win = torch.arange(2*r+1)
        ys = self.pc[..., 0, None, None] + win[None, None, :, None]
        xs = self.pc[..., 1, None, None] + win[None, None, None, :]
        local = padded[self.bidx[:, None, None, None], ys, xs].flatten(2)
        age = self.t - self.message_seen
        valid = age <= c.message_ttl
        rel = (self.message[:, None] - self.pc).float() / c.H
        rel = rel * valid[:, None, None]
        scalars = torch.stack((torch.full_like(self.alive, self.t / c.T, dtype=torch.float),
            self.pc[..., 0] / c.H, self.pc[..., 1] / c.H, self.alive.float(),
            rel[..., 0], rel[..., 1], valid[:, None].expand_as(self.alive).float(),
            (age / c.message_ttl).clamp(0, 1)[:, None].expand_as(self.alive)), -1)
        return torch.cat((local, scalars), -1)

    def step(self, actions):
        c = self.cfg
        if self.t >= c.T:
            raise RuntimeError("Episode termine")
        if actions.shape != self.alive.shape or (actions < 0).any() or (actions > 6).any():
            raise ValueError("Action invalide")
        active = self.alive.clone()
        delta = (self.pc[:, :, None] - self.gates[:, None]).abs()
        near = delta.sum(-1) <= 1
        requests = ((actions == 5) & active)[:, :, None] & near
        request = requests.any(1) & self.valid_gates & (self.kinds == 2)[:, None]
        request &= (self.t >= self.request_until) & (self.due < 0) & ~self.opened
        self.due = torch.where(request, self.t + self.response_delay[:, None], self.due)
        self.pending_accept = torch.where(request, self.accept[self.t], self.pending_accept)
        # Partage uniquement une ouverture actuellement visible, avec pertes et retard.
        visible = (delta.max(-1).values <= c.obs_r) & self.opened[:, None]
        sender = (actions == 6) & active & visible.any(-1)
        score = self.priorities[self.t].masked_fill(~sender, -1)
        who = score.argmax(1)
        choices = visible[self.bidx, who].float().argmax(1)
        message = self.gates[self.bidx, choices].clone()
        valid = sender.any(1) & self.delivery[self.t]
        self.messages.append((self.t + c.comm_delay, self.t, message, valid))
        move = torch.where(actions < 5, actions, 0)
        proposed = (self.pc + self.dirs[move]).clamp(0, c.H - 1)
        permitted = active & ~self.at(self.walls | self.blocked, proposed)
        # Une fermeture interdit de nouvelles entrees ; un groupe deja sur la case peut en sortir.
        entering = (proposed != self.pc).any(-1)
        contact = active & entering & self.at(self.blocked, proposed)
        for g in range(c.gate_count):
            candidates = permitted & entering & (proposed == self.gates[:, g, None]).all(-1)
            priority = self.priorities[self.t].masked_fill(~candidates, -1)
            selected = priority.argsort(1, descending=True).argsort(1) < c.gate_capacity
            permitted &= ~candidates | selected
        self.pc = torch.where(permitted[..., None], proposed, self.pc)
        outside = (self.pc - c.H // 2).abs().max(-1).values > c.ring
        self.crossed |= outside & active
        self.outside_streak = torch.where(outside & active, self.outside_streak + 1, 0)
        escaped = active & (self.outside_streak >= c.outside_hold if c.exit_target == "cordon"
                            else self.at(self.border.expand_as(self.walls), self.pc))
        self.alive &= ~escaped
        occupancy = self.occupancy()
        excess = (self.at(occupancy, self.pc) - c.cap).clamp(min=0) * self.alive
        reward = c.exit_reward * escaped.float() - c.waiting_cost * active - c.density_cost * excess
        reward -= c.blocked_cost * contact + c.request_cost * ((actions == 5) & active)
        reward += c.team_exit_reward * escaped.float().mean(1, keepdim=True) * active
        self.t += 1
        resolved = (self.due >= 0) & (self.t >= self.due)
        accepted = resolved & self.pending_accept
        self.open_until = torch.where(accepted, self.t + self.duration[:, None], self.open_until)
        self.request_until = torch.where(resolved, self.t + self.response_delay[:, None], self.request_until)
        self.due = torch.where(resolved, -1, self.due)
        while self.messages and self.messages[0][0] <= self.t:
            _, seen, pos, good = self.messages.pop(0)
            self.message = torch.where(good[:, None], pos, self.message)
            self.message_seen = torch.where(good, seen, self.message_seen)
        self._barriers()
        done = self.t == c.T
        if done:
            reward -= c.remaining_cost * self.alive
        info = {"escaped": escaped, "excess": excess, "blocked_attempt": contact,
                "request": (actions == 5) & active, "accepted": accepted,
                "max_density": occupancy.flatten(1).max(1).values, "crossed": self.crossed.clone()}
        return self.obs(), reward, done, info


class RotatedEnv(ExitEnv):
    """Tourne un episode entier pour une evaluation appariee des orientations."""
    def __init__(self, *args, turns=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.turns = turns

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        for _ in range(self.turns):
            for name in ("walls", "cut", "gate_grid"):
                setattr(self, name, torch.rot90(getattr(self, name), 1, (1, 2)))
            self.pc = torch.stack((self.cfg.H-1-self.pc[..., 1], self.pc[..., 0]), -1)
            rotated = torch.stack((self.cfg.H-1-self.gates[..., 1], self.gates[..., 0]), -1)
            self.gates = torch.where(self.valid_gates[..., None], rotated, self.gates)
        self._barriers()
        return self.obs()
