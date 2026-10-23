"""Entrainement PPO en self-play : un reseau pour la foule, un pour la police.

Chaque camp partage un seul reseau entre tous ses agents (chaque groupe de la foule
voit sa propre fenetre locale et decide seul). Les deux camps apprennent en meme temps.

Adversaire mixte : la moitie des parties se joue contre la police scriptee
(police_script.py), l'autre contre la police apprise. Sans ca, la police apprise
s'effondrait et la foule apprenait juste a ne pas bouger.

Usage : python train.py --iters 300
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from env import Config, NasseEnv, paris_config
from police_script import ScriptedPolice


class Policy(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, 5)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def cpu(net):
    return {k: v.cpu() for k, v in net.state_dict().items()}


def load_policy(net, sd):
    """Charge un reseau, meme entraine avec moins d'entrees (anciennes observations) :
    les nouvelles entrees partent avec un poids nul, le reste est repris tel quel."""
    w = sd["body.0.weight"]
    d = net.body[0].weight.shape[1]
    if w.shape[1] < d:
        sd = dict(sd)
        sd["body.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], d - w.shape[1], device=w.device)], 1)
    dev = next(net.parameters()).device
    net.load_state_dict({k: v.to(dev) for k, v in sd.items()})


def act(net, obs):
    with torch.no_grad():
        logits, v = net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), v


def gae(rew, val, active, gamma=0.99, lam=0.95):
    """rew, val, active : (T,B,N). Un agent inactif (sorti) n'a plus de futur."""
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    last = torch.zeros_like(rew[0])
    for t in reversed(range(T)):
        if t == T - 1:
            nonterm, nv = torch.zeros_like(rew[0]), torch.zeros_like(rew[0])
        else:
            nonterm, nv = active[t + 1].float(), val[t + 1]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def ppo_update(net, opt, obs, act_, logp, adv, ret, epochs=4, mb=8192, clip=0.2, ent=0.01):
    n = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, mb):
            j = perm[i:i + mb]
            logits, v = net(obs[j])
            dist = torch.distributions.Categorical(logits=logits)
            ratio = (dist.log_prob(act_[j]) - logp[j]).exp()
            pg = -torch.min(ratio * adv[j], ratio.clamp(1 - clip, 1 + clip) * adv[j]).mean()
            loss = pg + 0.5 * (v - ret[j]).pow(2).mean() - ent * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()


def rollout(env, crowd, police, scripted, scr_mask):
    """scr_mask (B,) : True = cette partie se joue contre la police scriptee."""
    cfg = env.cfg
    env.reset()
    scripted.reset()
    oc, op = env.refresh()
    buf = {k: [] for k in ["oc", "ac", "lc", "vc", "rc", "mc", "op", "ap", "lp", "vp", "rp"]}
    leave_t = torch.full(env.alive.shape, float(cfg.T))
    tenus = bloques = 0.0
    for t in range(cfg.T):
        mc = env.alive.clone()
        ac, lc, vc = act(crowd, oc)
        ap, lp, vp = act(police, op)
        ap = torch.where(scr_mask[:, None], scripted.act(), ap)
        for k, v in zip(["oc", "ac", "lc", "vc", "mc", "op", "ap", "lp", "vp"], [oc, ac, lc, vc, mc, op, ap, lp, vp]):
            buf[k].append(v)
        (oc, op), rc, rp, done, info = env.step(ac, ap)
        tenus += info["tenu"].float().mean().item()
        bloques += info["enclosed"].float().mean().item()
        out_demo = ~info["in_demo"] & (leave_t == cfg.T)
        leave_t[out_demo] = t
        buf["rc"].append(rc)
        buf["rp"].append(rp)
    S = {k: torch.stack(v) for k, v in buf.items()}
    k = info["kettled"].float().mean(1)
    stats = {
        "encercles": k.mean().item(),
        "encercles_vs_script": k[scr_mask].mean().item(),
        "encercles_vs_ia": k[~scr_mask].mean().item(),
        "sortis": (~env.alive & ~info["kettled"]).float().mean().item(),
        "presence": torch.stack([m.float() for m in buf["mc"]]).mean().item(),
        "depart_moyen": leave_t.mean().item(),
        "densite": S["rc"].clamp(max=0).sum().item() / env.B,
        "tours_tenus": tenus,
        "tours_bloques": bloques,
    }
    return S, stats


# Mode "progressive" : la police est deja en place et ferme les sorties une a une.
# Le cout d'une nasse devient le temps bloque (par tour), plus le passage au filtre.
PROGRESSIVE = dict(fermeture="progressive", enclosed_cost=0.15, kettle_penalty=4.0)


# Be water : la manif compte la ou la foule se regroupe, l'info circule avec 30 s de
# retard, et la police dispose d'une reserve mobile qui encercle les regroupements.
BEWATER = dict(presence_mode="rassemblement", comm_delay=3)


def train(iters, B, out, seed, resume=None, paris=False, fermeture="arrivee", presence=None, threads=12,
          bewater=False, cohesion=0.0, n_police=None):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    kw = dict(PROGRESSIVE) if fermeture == "progressive" else {}
    if fermeture == "progressive":
        kw["T"] = 144 if paris else 72
    if presence is not None:
        kw["presence_reward"] = presence
    if bewater:
        kw.update(BEWATER, cohesion_coef=cohesion)
    if n_police is not None:
        kw["n_police"] = n_police
    elif bewater and paris:
        kw["n_police"] = 100  # anneau + reserve mobile (calibre sur Republique, 23/10)
    if paris:
        # vrais quartiers : uniquement la police scriptee, avec un anneau qui varie
        cfg = paris_config(**kw)
        env = NasseEnv(cfg, B, seed)
        scripted = ScriptedPolice(env, ring_extra=(2, 3, 4, 5), seed=seed, window=(0.0, 0.35))
        scr_mask = torch.ones(B, dtype=torch.bool)
    elif fermeture == "progressive":
        # la police apprise ne sait pas jouer ce dispositif : police scriptee partout,
        # avec des fermetures parfois d'un coup (fenetre nulle), parfois etalees
        cfg = Config(**kw)
        env = NasseEnv(cfg, B, seed)
        scripted = ScriptedPolice(env, seed=seed, window=(0.0, 0.35))
        scr_mask = torch.ones(B, dtype=torch.bool)
    else:
        cfg = Config(**kw)
        env = NasseEnv(cfg, B, seed)
        scripted = ScriptedPolice(env)
        scr_mask = torch.arange(B) < B // 2
    crowd, police = Policy(env.obs_dim), Policy(env.obs_dim)
    if resume:
        ck = torch.load(resume, map_location="cpu")
        load_policy(crowd, ck["crowd"])
        load_policy(police, ck["police"])
    oc_, op_ = torch.optim.Adam(crowd.parameters(), 3e-4), torch.optim.Adam(police.parameters(), 3e-4)
    out = Path(out)
    out.mkdir(exist_ok=True)
    log = []
    t0 = time.time()
    for it in range(1, iters + 1):
        S, stats = rollout(env, crowd, police, scripted, scr_mask)
        # foule : uniquement les pas ou le groupe est encore dans le quartier
        adv, ret = gae(S["rc"], S["vc"], S["mc"])
        m = S["mc"]
        ppo_update(crowd, oc_, S["oc"][m], S["ac"][m], S["lc"][m], adv[m], ret[m])
        # police apprise : uniquement ses propres parties
        allp = torch.ones_like(S["rp"], dtype=torch.bool)
        adv, ret = gae(S["rp"], S["vp"], allp)
        m = (~scr_mask)[None, :, None].expand_as(allp)
        if m.any():
            ppo_update(police, op_, S["op"][m], S["ap"][m], S["lp"][m], adv[m], ret[m])
        stats["iter"] = it
        log.append(stats)
        if it % 10 == 0 or it == 1:
            print(f"[{it:4d}] {time.time() - t0:6.0f}s  encercles script {stats['encercles_vs_script']:.2f} ia {stats['encercles_vs_ia']:.2f}  tenus {stats['tours_tenus']:.1f}  bloques {stats['tours_bloques']:.1f}  sortis {stats['sortis']:.2f}  "
                  f"depart {stats['depart_moyen']:.1f}  densite {stats['densite']:.2f}", flush=True)
        if it == 1 or it % 10 == 0 or it == iters:
            checkpoint = {"crowd": cpu(crowd), "police": cpu(police), "cfg": cfg.__dict__,
                          "iteration": it, "parent_checkpoint": str(resume) if resume else None}
            if it % 50 == 0 or it == iters:
                torch.save(checkpoint, out / f"ckpt_{it}.pt")
            # Publication atomique pour que le rapatriement ne lise pas une ecriture partielle.
            temporary = out / "ckpt_last.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(out / "ckpt_last.pt")
            temporary_log = out / "log.json.tmp"
            temporary_log.write_text(json.dumps(log))
            temporary_log.replace(out / "log.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--B", type=int, default=64)
    p.add_argument("--out", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", help="checkpoint a partir duquel reprendre")
    p.add_argument("--paris", action="store_true", help="vrais quartiers de Paris (rues OSM)")
    p.add_argument("--fermeture", default="arrivee", choices=["arrivee", "progressive"])
    p.add_argument("--presence", type=float, help="gain par tour passe a la manif (attachement)")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--bewater", action="store_true", help="la manif compte la ou la foule se regroupe")
    p.add_argument("--cohesion", type=float, default=0.0, help="cout d'un groupe isole (modele ESIM)")
    p.add_argument("--n-police", type=int, help="unites de police (anneau + reserve mobile)")
    a = p.parse_args()
    if torch.cuda.is_available():
        torch.set_default_device("cuda")  # sur Vast.ai : env et reseaux sur la carte graphique
    train(a.iters, a.B, a.out, a.seed, a.resume, a.paris, a.fermeture, a.presence, a.threads,
          a.bewater, a.cohesion, a.n_police)
