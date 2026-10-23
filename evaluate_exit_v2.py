"""Selection sur validation puis comparaison sur une graine de test reservee."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from exit_env import ExitConfig, RotatedEnv
from osm_map import place_map
from train import act
from train_exit import OBJECTIVE_V2, make_policy, rollout


def local_policy(kind):
    visits = None
    def policy(env, obs):
        nonlocal visits
        c = env.cfg
        B, N = env.pc.shape[:2]
        local = obs[..., :-8].reshape(B, N, 2*c.obs_r+1, 2*c.obs_r+1, 5)
        dirs = env.dirs
        neighbors = local[..., c.obs_r+dirs[:, 0], c.obs_r+dirs[:, 1], :]
        available = (neighbors[..., 0] < .5) & (neighbors[..., 1] < .5)
        # Les seules positions consultees sont celles des groupes eux-memes.
        positions = torch.round(obs[..., -7:-5] * c.H).long()
        if kind == "local_explore":
            if visits is None:
                visits = torch.zeros(B, N, c.H*c.H, device=obs.device)
            index = positions[..., 0]*c.H+positions[..., 1]
            visits.scatter_add_(2, index[..., None], env.alive.float()[..., None])
            candidates = (positions[..., None, :]+dirs).clamp(0, c.H-1)
            idx = candidates[..., 0]*c.H+candidates[..., 1]
            score = visits.gather(2, idx) + torch.rand(B, N, 5, device=obs.device)
            score[..., 0] += 1
        elif kind == "random_legal":
            score = torch.rand(B, N, 5, device=obs.device)
        else:
            raise ValueError(kind)
        # Sur une case qui vient de se refermer, attendre reste physiquement possible.
        available[..., 0] = True
        actions = score.masked_fill(~available, float("inf")).argmin(-1)
        near_police = (neighbors[..., 1] > .5).any(-1)
        ask = near_police & (torch.rand(B, N, device=obs.device) < .2)
        return torch.where(ask, 5, actions)
    return policy


def learned(net, mode):
    def policy(env, obs):
        actions, _, _ = act(net, obs)
        if mode == "no_sharing":
            actions = torch.where(actions == 6, 0, actions)
        return actions
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--out", default="campaign/evaluation_v2.json")
    parser.add_argument("--B", type=int, default=16)
    parser.add_argument("--validation-B", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    nets, configs = {}, {}
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["objective"] != OBJECTIVE_V2:
            raise ValueError("Evaluation reservee aux checkpoints v2")
        cfg = ExitConfig(**checkpoint["cfg"])
        net = make_policy(cfg, checkpoint["architecture"])
        net.load_state_dict(checkpoint["crowd"])
        net.eval()
        nets[path], configs[path] = net, cfg
    if any(asdict(cfg) != asdict(next(iter(configs.values()))) for cfg in configs.values()):
        raise ValueError("Les configurations des candidats doivent etre identiques")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(objective=OBJECTIVE_V2, success="3 consecutive steps outside cordon", validation_seed=2027,
        test_seed=2029, B=args.B, validation_B=args.validation_B, cfg=asdict(cfg),
        checkpoints={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in nets},
        sources={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                 ("exit_env.py", "train_exit.py", "evaluate_exit_v2.py")},
        validation=[], test=[], replays=[], complete=False)
    def save():
        temp = output.with_suffix(".tmp")
        temp.write_text(json.dumps(report))
        temp.replace(output)

    def evaluate(stage, path, profile, kind, B, seed):
        config = configs[path]
        if profile == "difficult":
            config = replace(config, first_open_min=60, first_open_max=90, response_min=18, response_max=30,
                             response_probability_min=0, response_probability_max=.2, comm_drop=.6)
        scenarios = ("delayed", "responsive", "relocating") if stage == "validation" else ("sealed", "delayed", "responsive", "relocating")
        for place in ("republique", "bastille", "nation"):
            walls = np.stack([~place_map(place, config.H)]*B)
            for turns in range(4):
                for scenario in scenarios:
                    torch.manual_seed(seed)
                    env = RotatedEnv(config, B, seed, turns=turns)
                    policy = learned(nets[path], kind) if kind in ("full", "no_sharing") else local_policy(kind)
                    record = stage == "test" and turns == 0 and kind in ("full", "local_explore")
                    _, stats, replay = rollout(env, policy, scenario, walls, collect=False, record=record)
                    if scenario == "sealed" and stats["sortis"] != 0:
                        raise AssertionError("Sortie observee sans ouverture")
                    stats.update(checkpoint=path, profile=profile, policy=kind, lieu=place, rotation=turns*90, scenario=scenario)
                    report[stage].append(stats)
                    if replay:
                        replay.update(profile=profile, policy=kind, lieu=place, scenario_name=scenario)
                        report["replays"].append(replay)
                    save()
            print(stage, Path(path).parent.name, profile, kind, place, flush=True)

    for path in nets:
        evaluate("validation", path, "nominal", "full", args.validation_B, report["validation_seed"])
    scores = {}
    for path in nets:
        rows = [r for r in report["validation"] if r["checkpoint"] == path]
        partitions = [np.mean([r["sortis"] for r in rows if r["lieu"] == place and r["rotation"] == rotation])
                      for place in ("republique", "bastille", "nation") for rotation in (0,90,180,270)]
        scores[path] = dict(worst_partition=float(min(partitions)), mean=float(np.mean([r["sortis"] for r in rows])))
    selected = max(scores, key=lambda p: (scores[p]["worst_partition"], scores[p]["mean"]))
    report.update(selected_checkpoint=selected, validation_scores=scores,
                  selection_rule="Maximize worst place/rotation validation mean; tie-break by overall mean")
    save()
    for profile in ("nominal", "difficult"):
        for kind in ("full", "no_sharing", "local_explore", "random_legal"):
            evaluate("test", selected, profile, kind, args.B, report["test_seed"])
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
