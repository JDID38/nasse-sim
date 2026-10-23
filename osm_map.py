"""Transforme les rues OSM de Paris (data/paris_roads.json) en grille.

1 case = CELL metres. Largeur selon la classe OSM : grande voie 3 cases, rue et
petite rue 1 case (une case touche toute case a moins de RAYONS de la ligne,
ce qui garde les rues en diagonale praticables en 4 directions).
Les voies pietonnes fermees (places) sont remplies.

Coordonnees source : degres x 100 000.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

CELL = 10.0
M_LAT = 1.1132                       # metres par unite de latitude (1e-5 degre)
M_LON = 1.1132 * np.cos(np.radians(48.86))
RAYONS = {0: 1.5, 1: 0.75, 2: 0.75}  # en cases
PLACE_MAX_M = 300                     # au-dela, une voie fermee est une boucle, pas une place

LIEUX = {
    "republique": (48.8675, 2.3637),
    "nation": (48.8483, 2.3959),
    "bastille": (48.8532, 2.3691),
}


@lru_cache(1)
def _ways():
    d = json.loads((Path(__file__).parent / "data" / "paris_roads.json").read_text(encoding="utf-8"))
    out = []
    for w in d["w"]:
        pts = np.array(w[1:], dtype=float).reshape(-1, 2)
        out.append((w[0], pts))
    return out


def rasterize(lat, lon, h, w=None):
    """Grille h x w (True = rue) centree sur (lat, lon) en degres."""
    w = w or h
    lat0, lon0 = lat * 1e5, lon * 1e5
    grid = np.zeros((h, w), bool)
    samples, places = [], []
    for cls, pts in _ways():
        # y vers le bas = vers le sud
        y = (h * CELL / 2 - (pts[:, 0] - lat0) * M_LAT) / CELL
        x = ((pts[:, 1] - lon0) * M_LON + w * CELL / 2) / CELL
        if (y.max() < -2) | (y.min() > h + 2) | (x.max() < -2) | (x.min() > w + 2):
            continue
        for i in range(len(y) - 1):
            n = int(np.hypot(y[i + 1] - y[i], x[i + 1] - x[i]) / 0.25) + 1
            t = np.linspace(0, 1, n + 1)
            samples.append(np.stack([y[i] + (y[i + 1] - y[i]) * t, x[i] + (x[i + 1] - x[i]) * t,
                                     np.full(n + 1, RAYONS[cls])], -1))
        closed = len(pts) > 3 and (pts[0] == pts[-1]).all()
        if cls == 2 and closed and max(np.ptp(y), np.ptp(x)) * CELL <= PLACE_MAX_M:
            places.append(np.stack([y, x], -1))
    s = np.concatenate(samples)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            cy, cx = np.floor(s[:, 0]) + dy, np.floor(s[:, 1]) + dx
            ok = ((cy + 0.5 - s[:, 0]) ** 2 + (cx + 0.5 - s[:, 1]) ** 2 <= s[:, 2] ** 2)
            ok &= (cy >= 0) & (cy < h) & (cx >= 0) & (cx < w)
            grid[cy[ok].astype(int), cx[ok].astype(int)] = True
    if places:
        from matplotlib.path import Path as MPath  # seulement pour rasteriser, pas pour entrainer
        yy, xx = np.mgrid[:h, :w] + 0.5
        centers = np.stack([yy.ravel(), xx.ravel()], -1)
        for poly in places:
            y0, x0 = np.floor(poly.min(0)).astype(int).clip(0)
            y1, x1 = (np.ceil(poly.max(0)).astype(int) + 1)
            sub = (centers[:, 0] >= y0) & (centers[:, 0] <= y1) & (centers[:, 1] >= x0) & (centers[:, 1] <= x1)
            idx = np.nonzero(sub)[0]
            if len(idx):
                inside = MPath(poly).contains_points(centers[idx])
                grid.flat[idx[inside]] = True
    return grid


def ascii(grid):
    return "\n".join("".join("." if v else "#" for v in row) for row in grid)


if __name__ == "__main__":
    import sys
    lieu = sys.argv[1] if len(sys.argv) > 1 else "republique"
    g = rasterize(*LIEUX[lieu], 65)
    print(ascii(g))
    print(f"{lieu} : {g.mean():.0%} de cases de rue")


# ---------- tout Paris, pour tirer des quartiers d'entrainement ----------
GRAND = Path(__file__).parent / "data" / "paris_grid.npz"
CENTRE_PARIS = (48.8590, 2.3470)
GRAND_TAILLE = (1100, 1250)  # cases (11 km x 12,5 km), couvre Paris intra-muros


def big_grid():
    if not GRAND.exists():
        g = rasterize(*CENTRE_PARIS, *GRAND_TAILLE)
        np.savez_compressed(GRAND, grid=g)
    return np.load(GRAND)["grid"]


def to_cell(lat, lon):
    """Position (ligne, colonne) d'un point dans la grande grille."""
    h, w = GRAND_TAILLE
    y = h / 2 - (lat - CENTRE_PARIS[0]) * 1e5 * M_LAT / CELL
    x = w / 2 + (lon - CENTRE_PARIS[1]) * 1e5 * M_LON / CELL
    return int(y), int(x)


class CropSampler:
    """Tire des quartiers size x size centres sur un carrefour ou une place assez grande,
    a plus de `buffer_m` des lieux d'evaluation, avec rotations et symetries au hasard."""

    def __init__(self, size, demo_radius, min_street=0.5, buffer_m=1000):
        self.size, g = size, big_grid()
        self.g = g
        half = size // 2
        r = demo_radius
        # part de rue dans le carre de la manif, pour chaque centre possible
        c = np.cumsum(np.cumsum(np.pad(g, ((1, 0), (1, 0))).astype(np.int32), 0), 1)
        k = 2 * r + 1
        dens = (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
        ys, xs = np.nonzero(dens >= min_street)
        ys, xs = ys + r, xs + r
        ok = g[ys, xs] & (ys >= half) & (xs >= half) & (ys < g.shape[0] - half) & (xs < g.shape[1] - half)
        for lat, lon in LIEUX.values():
            ly, lx = to_cell(lat, lon)
            ok &= np.hypot(ys - ly, xs - lx) * CELL > buffer_m
        self.centers = np.stack([ys[ok], xs[ok]], -1)

    def crop(self, y, x):
        h = self.size // 2
        return self.g[y - h:y + h + 1, x - h:x + h + 1]

    def __call__(self, rng):
        y, x = self.centers[rng.integers(len(self.centers))]
        m = np.rot90(self.crop(y, x), rng.integers(4))
        return np.ascontiguousarray(m[:, ::-1] if rng.random() < 0.5 else m)


def place_map(lieu, size):
    """Quartier d'evaluation, pris dans la meme grande grille que l'entrainement."""
    y, x = to_cell(*LIEUX[lieu])
    h = size // 2
    return big_grid()[y - h:y + h + 1, x - h:x + h + 1]
