"""Audit des resultats v2 : reconstruction exacte des messages et illustrations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from exit_env import ExitConfig, ExitEnv
from dataclasses import replace
from osm_map import place_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/v2/campaign"))
    parser.add_argument("--out", type=Path, default=Path("results/analysis_v2"))
    args = parser.parse_args()
    root, out = args.root, args.out
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((root / "evaluation_v2.json").read_text())
    assert data["complete"]
    torch.set_num_threads(2)
    diagnostics = []
    for episode in data["replays"]:
        if episode["policy"] != "full":
            continue
        cfg = ExitConfig(**data["cfg"])
        if episode["profile"] == "difficult":
            cfg = replace(cfg, first_open_min=60, first_open_max=90, response_min=18, response_max=30,
                          response_probability_min=0, response_probability_max=.2, comm_drop=.6)
        env = ExitEnv(cfg, data["B"], data["test_seed"])
        walls = np.stack([~place_map(episode["lieu"], cfg.H)] * data["B"])
        env.reset(episode["scenario_name"], walls)
        first_open = first_message = None
        move_with_message = closer_with_message = delivered = 0
        previous_seen = -10000
        for index, frame in enumerate(episode["frames"]):
            assert env.pc[0].tolist() == frame["positions"], (episode["lieu"], index)
            assert env.alive[0].tolist() == frame["alive"]
            assert env.opened[0].tolist() == frame["opened"]
            if any(frame["opened"]) and first_open is None:
                first_open = frame["t"]
            valid = int(env.t-env.message_seen[0]) <= cfg.message_ttl
            frame["heard"] = env.message[0].tolist() if valid else None
            if valid and first_message is None:
                first_message = frame["t"]
            if int(env.message_seen[0]) != previous_seen:
                delivered += 1
                previous_seen = int(env.message_seen[0])
            if index == len(episode["frames"])-1:
                break
            actions = torch.zeros_like(env.alive, dtype=torch.long)
            actions[0] = torch.tensor(frame["actions"])
            before = env.pc[0].clone()
            alive = env.alive[0].clone()
            target = env.message[0].clone()
            env.step(actions)
            if valid:
                moved = alive & (before != env.pc[0]).any(-1)
                closer = (env.pc[0]-target).abs().sum(-1) < (before-target).abs().sum(-1)
                move_with_message += int(moved.sum())
                closer_with_message += int((moved & closer).sum())
        diagnostics.append(dict(profile=episode["profile"], lieu=episode["lieu"], scenario=episode["scenario_name"],
            first_open=first_open, first_message=first_message, delivered_messages=delivered,
            moves_with_message=move_with_message, moves_toward_message=closer_with_message,
            escaped=sum(not a for a in episode["frames"][-1]["alive"])))
    (out / "replay_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2), flush=True)

    profiles = ("nominal", "difficult")
    policies = ("full", "no_sharing", "local_explore", "random_legal")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, profile in zip(axes, profiles):
        rows = [r for r in data["test"] if r["profile"] == profile and r["scenario"] != "sealed"]
        scores = [100*np.mean([r["sortis"] for r in rows if r["policy"] == policy]) for policy in policies]
        bars = ax.bar(["Réseau", "Sans partage", "Exploration\nlocale", "Hasard\naccessible"], scores,
                      color=["#2563eb", "#60a5fa", "#64748b", "#cbd5e1"])
        ax.bar_label(bars, fmt="%.1f %%", padding=4)
        ax.set_ylim(0, 80)
        ax.set_title("Conditions standards" if profile == "nominal" else "Délais et refus accrus")
        ax.set_ylabel("Groupes sortis du cordon (%)")
    fig.suptitle("Test réservé : le partage d'une ouverture explique une grande partie du gain", fontsize=13)
    fig.supxlabel("Succès simulé : trois tours consécutifs dehors. Cas sans aucune ouverture exclus de ces moyennes.\nLe réseau produit davantage de densité que l'exploration locale ; ces scores ne mesurent pas la sécurité réelle.", fontsize=9)
    fig.savefig(out / "comparaison_v2.png", dpi=150)
    plt.close(fig)

    # Cas explicites de deux types, avec la regle de comparaison sur les memes conditions.
    selections = [("republique", "delayed"), ("bastille", "responsive")]
    picked = [[next(e for e in data["replays"] if e["profile"] == "nominal" and e["lieu"] == place
                    and e["scenario_name"] == scenario and e["policy"] == policy)
               for policy in ("full", "local_explore")] for place, scenario in selections]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.4), layout="constrained")
    def draw(t):
        for row, episodes in enumerate(picked):
            for col, ep in enumerate(episodes):
                ax = axes[row, col]
                ax.clear()
                frame = ep["frames"][t]
                grid = np.array(ep["walls"], dtype=int)
                grid[np.array(ep["cut"], dtype=bool)] = 2
                ax.imshow(grid, cmap=ListedColormap(["#f8fafc", "#cbd5e1", "#dc2626"]), vmin=0, vmax=2)
                # Aucun point futur ferme n'est dessine comme un repere.
                for gate, opened in zip(ep["gates"], frame["opened"]):
                    if opened:
                        ax.scatter(gate[1], gate[0], c="#22c55e", s=45, marker="s")
                pos, alive = np.array(frame["positions"]), np.array(frame["alive"])
                ax.scatter(pos[alive, 1], pos[alive, 0], s=12, color="#2563eb", alpha=.7)
                if frame.get("heard"):
                    y, x = frame["heard"]
                    ax.scatter(x, y, s=140, facecolors="none", edgecolors="#9333ea", linewidths=2)
                label = "Réseau" if col == 0 else "Exploration locale"
                mechanism = "ouverture tardive" if row == 0 else "demande de passage"
                ax.set_title(f"{label} · {ep['lieu']} · {mechanism}\nTour {t} · {int((~alive).sum())}/48 sortis", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlim(19, 46); ax.set_ylim(46, 19)
        fig.suptitle("Bleu : groupes • Rouge : cordon • Vert : ouverture\nCercle violet : position reçue par message, éventuellement périmée", fontsize=11)
        fig.supxlabel("Données cartographiques © les contributeurs OpenStreetMap — ODbL", fontsize=8)
    animation = FuncAnimation(fig, draw, frames=list(range(0, 145, 3)), interval=160)
    animation.save(out / "replays_v2.gif", writer=PillowWriter(fps=6), dpi=90)
    draw(96)
    fig.savefig(out / "replays_v2_t96.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
