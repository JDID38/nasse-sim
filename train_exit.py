"""PPO : apprendre depuis une nasse fermee, sans reprise des objectifs be water."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch
from torch import nn

from exit_env import ExitConfig, ExitEnv
from train import Policy, act, cpu, gae, ppo_update

OBJECTIVE = "closed_nasse_exit_v1"
OBJECTIVE_V2 = "closed_nasse_exit_v2"


class ExitPolicy(Policy):
    def __init__(self, obs_dim):
        super().__init__(obs_dim)
        self.pi = nn.Linear(256, 7)


def rotate_observation(obs, radius, H):
    """Quart de tour antihoraire de toutes les donnees spatiales observees."""
    width = 2 * radius + 1
    local = obs[..., :-8].reshape(*obs.shape[:-1], width, width, 5)
    local = torch.rot90(local, 1, (-3, -2)).flatten(-3)
    s = obs[..., -8:].clone()
    s[..., 1], s[..., 2] = (H-1)/H - obs[..., -6], obs[..., -7]
    s[..., 4], s[..., 5] = -obs[..., -3], obs[..., -4]
    return torch.cat((local, s), -1)


class SymmetricExitPolicy(ExitPolicy):
    """Moyenne sur les quatre orientations : pas de direction privilegiee."""
    def __init__(self, obs_dim, radius, H):
        super().__init__(obs_dim)
        self.radius, self.H = radius, H

    def forward(self, obs):
        views = [obs]
        for _ in range(3):
            views.append(rotate_observation(views[-1], self.radius, self.H))
        logits, values = super().forward(torch.stack(views))
        permutation = torch.arange(7, device=obs.device)
        quarter = torch.tensor([0, 3, 4, 2, 1, 5, 6], device=obs.device)
        aligned = []
        for k in range(4):
            aligned.append(logits[k][..., permutation])
            permutation = quarter[permutation]
        return torch.stack(aligned).mean(0), values.mean(0)


def make_policy(cfg, architecture="mlp"):
    dim = (2*cfg.obs_r+1)**2*5+8
    if architecture == "c4":
        return SymmetricExitPolicy(dim, cfg.obs_r, cfg.H)
    if architecture == "mlp":
        return ExitPolicy(dim)
    raise ValueError(f"Architecture inconnue : {architecture}")


def rollout(env, policy, scenario=None, walls=None, record=False, collect=True):
    obs = env.reset(scenario=scenario, walls=walls)
    buf = {key: [] for key in ("obs", "actions", "logp", "value", "reward", "active")}
    B, N = env.alive.shape
    risk = torch.zeros(B)
    peak = torch.zeros(B)
    blocked = torch.zeros(B)
    requests = torch.zeros(B)
    accepted = torch.zeros(B)
    exit_times = torch.full((B, N), float(env.cfg.T))
    frames = []
    for _ in range(env.cfg.T):
        if record:
            frames.append({"t": env.t, "positions": env.pc[0].tolist(), "alive": env.alive[0].tolist(),
                           "opened": env.opened[0].tolist()})
        active = env.alive.clone()
        if isinstance(policy, nn.Module):
            actions, logp, value = act(policy, obs)
        else:
            actions = policy(env, obs)
            logp = value = torch.zeros_like(active, dtype=torch.float)
        if record:
            frames[-1]["actions"] = actions[0].tolist()
        if collect:
            for key, item in zip(buf, (obs, actions, logp, value, None, active)):
                if item is not None:
                    buf[key].append(item)
        obs, reward, done, info = env.step(actions)
        if collect:
            buf["reward"].append(reward)
        risk += info["excess"].sum(1) / N
        peak = torch.maximum(peak, info["max_density"])
        blocked += info["blocked_attempt"].sum(1) / N
        requests += info["request"].sum(1) / N
        accepted += info["accepted"].sum(1)
        exit_times[info["escaped"]] = env.t
    exited = (~env.alive).float().mean(1)
    outside_final = (env.pc - env.cfg.H // 2).abs().max(-1).values > env.cfg.ring
    stats = {"sortis": exited.mean().item(), "sortis_par_episode": exited.tolist(),
             "tours_jusqua_sortie_ou_fin": exit_times.mean().item(), "risque_densite_proxy": risk.mean().item(),
             "pic_groupes_par_case": peak.mean().item(), "tentatives_bloquees": blocked.mean().item(),
             "demandes_par_groupe": requests.mean().item(), "ouvertures_sur_demande": accepted.mean().item(),
             "franchi_cordon": env.crossed.float().mean().item(),
             "hors_cordon_fin": outside_final.float().mean().item(),
             "retour_interieur_apres_franchissement": (env.crossed & ~outside_final).float().mean().item(),
             "tous_sortis": (~env.alive.any(1)).float().mean().item()}
    replay = None
    if record:
        frames.append({"t": env.t, "positions": env.pc[0].tolist(), "alive": env.alive[0].tolist(),
                       "opened": env.opened[0].tolist()})
        replay = {"walls": env.walls[0].tolist(), "cut": env.cut[0].tolist(), "gates": env.gates[0].tolist(),
                  "scenario": int(env.kinds[0]), "frames": frames}
    return {key: torch.stack(items) for key, items in buf.items()} if collect else {}, stats, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default="runs_exit")
    parser.add_argument("--paris", action="store_true")
    parser.add_argument("--v2", action="store_true")
    args = parser.parse_args()
    if min(args.iters, args.B) < 1:
        parser.error("iters et B doivent etre positifs")
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    torch.manual_seed(args.seed)
    cfg = ExitConfig(carte="paris", H=65, n_crowd=48, ring=9, spawn_radius=6) if args.paris else ExitConfig()
    if args.v2:
        cfg = replace(cfg, visible_contacts=False, random_orientation=True, exit_target="cordon")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "ckpt_last.pt").exists():
        raise RuntimeError("Dossier contenant deja un checkpoint : choisir un nouvel --out")
    env = ExitEnv(cfg, args.B, args.seed)
    architecture = "c4" if args.v2 else "mlp"
    net = make_policy(cfg, architecture)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    history = []
    started = time.monotonic()
    for iteration in range(1, args.iters + 1):
        data, stats, _ = rollout(env, net)
        mask = data["active"]
        adv, ret = gae(data["reward"], data["value"], mask)
        ppo_update(net, optimizer, data["obs"][mask], data["actions"][mask], data["logp"][mask], adv[mask], ret[mask])
        stats.update(iteration=iteration, elapsed=round(time.monotonic() - started, 2))
        history.append(stats)
        if iteration == 1 or iteration % 10 == 0 or iteration == args.iters:
            print(json.dumps(stats), flush=True)
            checkpoint = {"objective": OBJECTIVE_V2 if args.v2 else OBJECTIVE, "architecture": architecture,
                          "crowd": cpu(net), "cfg": asdict(cfg), "iteration": iteration,
                          "seed": args.seed, "optimizer": optimizer.state_dict(), "B": args.B}
            temporary = out / "ckpt_last.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(out / "ckpt_last.pt")
            temp_log = out / "log.json.tmp"
            temp_log.write_text(json.dumps(history))
            temp_log.replace(out / "log.json")
            if iteration % 100 == 0 or iteration == args.iters:
                torch.save(checkpoint, out / f"ckpt_{iteration}.pt")


if __name__ == "__main__":
    main()
