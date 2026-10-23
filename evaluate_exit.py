"""Evaluation et replays : sorties depuis une nasse initialement fermee."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from env import distance_map
from exit_env import ExitConfig, ExitEnv, SCENARIOS
from osm_map import place_map
from train_exit import OBJECTIVE, OBJECTIVE_V2, make_policy, rollout


def baseline(kind):
    cache = {}
    def policy(env, obs):
        if kind == "wait":
            return torch.zeros_like(env.alive, dtype=torch.long)
        if kind == "random":
            return torch.randint(7, env.alive.shape)
        if kind != "informed_rule":
            raise ValueError(kind)
        c = env.cfg
        if "opened" not in cache or not torch.equal(cache["opened"], env.opened):
            maps = []
            for g in range(c.gate_count):
                target = torch.zeros_like(env.walls)
                b = env.valid_gates[:, g].nonzero().flatten()
                point = env.gates[b, g]
                target[b, point[:, 0], point[:, 1]] = True
                maps.append(distance_map(~env.walls & (~env.blocked | target), target))
            cache["maps"] = torch.stack(maps, 1)
            cache["border"] = distance_map(~env.walls & ~env.blocked, env.border.expand_as(env.walls))
            cache["opened"] = env.opened.clone()
        cand = (env.pc[:, :, None] + env.dirs).clamp(0, c.H - 1)
        here = cache["maps"][env.bidx[:, None, None], torch.arange(c.gate_count)[None, :, None],
                              env.pc[:, None, :, 0], env.pc[:, None, :, 1]]
        # Regle volontairement privilegiee : connait toutes les portes actuellement ouvertes.
        allowed = env.opened | ~env.opened.any(1, keepdim=True)
        here = here.masked_fill(~allowed[:, :, None], float("inf"))
        nearest = here.argmin(1)
        chosen = cache["maps"][env.bidx[:, None], nearest]
        d = chosen[env.bidx[:, None, None], torch.arange(c.n_crowd)[None, :, None], cand[..., 0], cand[..., 1]]
        actions = d.argmin(-1)
        outside = (env.pc - c.H//2).abs().max(-1).values >= c.ring
        db = cache["border"][env.bidx[:, None, None], cand[..., 0], cand[..., 1]]
        actions = torch.where(outside, db.argmin(-1), actions)
        near = (env.pc - env.gates[env.bidx[:, None], nearest]).abs().sum(-1) <= 1
        closed = ~env.opened[env.bidx[:, None], nearest]
        return torch.where(near & closed & ~outside, 5, actions)
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt")
    parser.add_argument("--B", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--paris", action="store_true")
    parser.add_argument("--places", nargs="+", default=["republique", "bastille", "nation"])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--policies", nargs="+", default=["ia", "wait", "random", "informed_rule"])
    parser.add_argument("--profile", choices=["nominal", "slow_refusal"], default="nominal")
    parser.add_argument("--out", default="exit_results/evaluation.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    cfg = ExitConfig(carte="paris", H=65, n_crowd=48, ring=9, spawn_radius=6) if args.paris else ExitConfig()
    checkpoint = None
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        if checkpoint.get("objective") not in (OBJECTIVE, OBJECTIVE_V2):
            raise ValueError("Ce checkpoint ne correspond pas a l'objectif sortie de nasse")
        cfg = ExitConfig(**checkpoint["cfg"])
    if args.profile == "slow_refusal":
        cfg = replace(cfg, first_open_min=60, first_open_max=90, response_min=18, response_max=30,
                      response_probability_min=0, response_probability_max=0.2, comm_drop=0.6)
    locations = args.places if cfg.carte == "paris" else ["grille"]
    report = {"objective": checkpoint["objective"] if checkpoint else OBJECTIVE, "cfg": asdict(cfg), "B": args.B, "seed": args.seed,
              "profile": args.profile, "checkpoint": args.ckpt,
              "checkpoint_sha256": hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest() if args.ckpt else None,
              "results": [], "episodes": [], "complete": False,
              "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                for name in ("exit_env.py", "train_exit.py", "evaluate_exit.py")},
              "note": "Risque de densite = proxy, pas probabilite de blessure. informed_rule connait la carte et les ouvertures actuelles."}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(json.dumps(report), encoding="utf-8")
        tmp.replace(output)
    for location in locations:
        walls = np.stack([~place_map(location, cfg.H)] * args.B) if cfg.carte == "paris" else None
        for scenario in args.scenarios:
            for kind in args.policies:
                if kind == "ia" and checkpoint is None:
                    continue
                torch.manual_seed(args.seed)
                env = ExitEnv(cfg, args.B, args.seed)
                if kind == "ia":
                    policy = make_policy(cfg, checkpoint.get("architecture", "mlp"))
                    policy.load_state_dict(checkpoint["crowd"])
                else:
                    policy = baseline(kind)
                _, stats, episode = rollout(env, policy, scenario, walls, record=True, collect=False)
                if scenario == "sealed" and stats["sortis"] != 0:
                    raise AssertionError("Sortie impossible observee : bug de simulation")
                stats.update(lieu=location, scenario=scenario, policy=kind)
                fractions = np.array(stats["sortis_par_episode"])
                stats["erreur_standard_entre_parties"] = float(fractions.std(ddof=1) / np.sqrt(args.B)) if args.B > 1 else None
                episode.update(lieu=location, policy=kind, scenario_name=scenario)
                report["results"].append(stats)
                report["episodes"].append(episode)
                save()
                print(json.dumps({k: v for k, v in stats.items() if k != "sortis_par_episode"}), flush=True)
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
