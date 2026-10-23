"""Bilan automatique des comparaisons v2, sans transformer le score en conseil terrain."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    path = Path(args.input)
    data = json.loads(path.read_text())
    if not data["complete"] or len(data["validation"]) != 108 or len(data["test"]) != 384:
        raise RuntimeError("Evaluation incomplete")
    lines = ["# Bilan automatique — sortie de cordon v2", "",
        "Succes : trois tours consecutifs dehors, puis retrait du groupe de la simulation.",
        "Les tours ne sont pas calibres en secondes. Aucune garantie de securite reelle.", "",
        f"Reseau retenu sur validation : `{data['selected_checkpoint']}`.",
        "Selection sur la pire combinaison lieu/orientation ; test sur une graine separee.", "",
        "Les moyennes suivantes excluent le controle sans ouverture (`sealed`).", "",
        "| Profil | Comportement | Sorties | Densite (proxy) | Tous sortis |",
        "|---|---|---:|---:|---:|"]
    summary = {}
    for profile in ("nominal", "difficult"):
        scores = {}
        for policy in ("full", "no_sharing", "local_explore", "random_legal"):
            rows = [r for r in data["test"] if r["profile"] == profile and r["policy"] == policy and r["scenario"] != "sealed"]
            score = {k: float(np.mean([r[k] for r in rows])) for k in ("sortis", "risque_densite_proxy", "tous_sortis")}
            scores[policy] = score
            lines.append(f"| {profile} | {policy} | {100*score['sortis']:.1f} % | {score['risque_densite_proxy']:.2f} | {100*score['tous_sortis']:.1f} % |")
        summary[profile] = scores
    lines += ["", "## Lecture", ""]
    for profile, scores in summary.items():
        delta = scores["full"]["sortis"] - scores["local_explore"]["sortis"]
        lines.append(f"- {profile} : ecart reseau / exploration locale = {100*delta:+.1f} points.")
    if all(scores["full"]["sortis"] > scores["local_explore"]["sortis"] for scores in summary.values()):
        lines.append("Le reseau depasse cette regle locale sur ces deux profils ; ce resultat doit encore etre explique par les trajectoires et confirme sur d'autres modeles de police.")
    else:
        lines.append("Le reseau ne depasse pas la regle locale dans tous les profils. Ne pas annoncer de nouvelle strategie superieure sur cette base.")
    lines += ["", "La regle local_explore memorise les cases deja visitees et demande parfois un passage lorsqu'un cordon est adjacent. Elle ne voit ni les passages futurs ni la carte globale.",
        "La densite est un indicateur interne, pas un taux de blessures. Les profils de police restent hypothetiques.",
        "Les replays dans evaluation_v2.json permettent d'examiner les comportements du reseau et de la regle, avec les memes conditions initiales."]
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    path.with_name("summary_v2.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
