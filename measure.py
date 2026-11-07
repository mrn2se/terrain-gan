"""Measure the cascade against real terrain (TODO A1, A2, A3, A5, A6, A7, A8, C1 and F1).

For every level, real crops are compared with:
  generated     the whole cascade down to this level, from random worlds (C1: on generated input)
  refined real  this level's refiner on block means of the real crops (C1: on real input)
  bicubic       plain bicubic upscaling of the same block means (A2)
  bilinear      plain bilinear upscaling (A2)
  fBm noise     fractal noise with the real power-law spectrum and per-tile height spread (F1)
Measured: height and slope distributions and radial power spectra (A3), and the error
against the real crop for the methods that start from it (A2). A1 renders the same real
input with other random values: how similar the added detail stays shows whether the
refiner uses its noise. A7 compares block means of each generated level with the level
above at the same place. A6 compares how height differences grow with distance at level 0,
for real Earth and for large generated areas, and A8 how alike random pairs of tiles are. A5 fills depressions and routes water downhill at the finest
level: real rivers gather into few large ones, with drainage areas following a power law.

Usage: uv run measure.py [--ckpt models/world.pt models/regional.pt models/local.pt] [--tiles 256]
Writes <out>/stats_level<L>.png, <out>/variogram.png, <out>/hydrology.png and <out>/measure.json, and prints a summary.
"""

import argparse
import heapq
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from generate import load_cascade  # noqa: E402
from terrain_gan import Refiner, meters_to_norm, norm_to_meters  # noqa: E402
from train import cache_downsampled, cache_raster  # noqa: E402
from train_refiner import RefinerCrops  # noqa: E402

SIZE = 128
LEVEL_SCALE = 4
NAMES = ("world", "regional", "local")
COLOURS = {"real": "black", "generated": "tab:red", "refined real": "tab:blue", "bicubic": "tab:green", "bilinear": "tab:olive", "fBm noise": "tab:purple"}


def real_pairs(npy, ppd, count, seed, min_land, level):
    """count real (coarse, fine) pairs in metres, using the refiners' training crops."""
    cells = Refiner.cells_for(SIZE) if level > 0 else SIZE // Refiner.stride
    crops = iter(RefinerCrops(npy, ppd, cells, min_land=min_land, seed=seed))
    pairs = [next(crops) for _ in range(count)]
    coarse = norm_to_meters(torch.stack([c for c, _ in pairs]).numpy())
    fine = norm_to_meters(torch.stack([f for _, f in pairs]).numpy())
    if level == 0:
        return coarse, fine[:, 0]
    o = Refiner.origin
    return coarse, fine[:, 0, o : o + SIZE, o : o + SIZE]


def generated(cascade, level, count, rng, min_land, ocean_keep=0.1):
    """Tiles of random worlds at random places, filtered like the training crops.

    Also returns the RMS difference in metres between each tile's block means and the
    level above at the same place (A7), or None at level 0.
    """
    tiles, mismatch = [], []
    while len(tiles) < count:
        seed = int(rng.integers(1 << 31))
        x0, y0 = (LEVEL_SCALE * int(v) for v in rng.integers(-(1 << 20), 1 << 20, 2))
        meters = norm_to_meters(cascade.render(level, seed, x0, y0, SIZE, SIZE, "cpu"))
        if (meters > 0).mean() >= min_land or rng.random() < ocean_keep:
            tiles.append(meters)
            if level > 0:
                n = SIZE // LEVEL_SCALE
                above = norm_to_meters(cascade.render(level - 1, seed, x0 // 4, y0 // 4, n, n, "cpu"))
                blocks = meters.reshape(n, LEVEL_SCALE, n, LEVEL_SCALE).mean((1, 3))
                mismatch.append(np.sqrt(((blocks - above) ** 2).mean()))
    return np.stack(tiles), (float(np.sqrt(np.mean(np.square(mismatch)))) if mismatch else None)


def refine(refiner, coarse_m, seed_latent, seed_noise):
    """Refiner output in metres; the latent and the layer noise have separate seeds."""
    coarse = torch.from_numpy(meters_to_norm(coarse_m).astype(np.float32))
    gl, gn = torch.Generator().manual_seed(seed_latent), torch.Generator().manual_seed(seed_noise)
    latent = torch.randn(coarse.shape[0], refiner.latent_ch, *coarse.shape[2:], generator=gl)
    out = refiner(coarse, latent, lambda _, __, h, w: torch.randn(coarse.shape[0], 1, h, w, generator=gn))
    return norm_to_meters(out[:, 0, :SIZE, :SIZE].numpy())


def upscale(coarse_m, mode):
    x = F.interpolate(torch.from_numpy(coarse_m), scale_factor=LEVEL_SCALE, mode=mode, align_corners=False)
    o = Refiner.origin
    return x[:, 0, o : o + SIZE, o : o + SIZE].numpy()


def slopes(meters, km_per_px):
    dy, dx = np.gradient(meters, km_per_px * 1000.0, axis=(1, 2))
    return np.degrees(np.arctan(np.hypot(dx, dy)))


def spectrum(meters):
    """Radially averaged power spectrum, mean over tiles; index k is k cycles per tile."""
    n = meters.shape[1]
    window = np.outer(np.hanning(n), np.hanning(n))
    x = (meters - meters.mean((1, 2), keepdims=True)) * window
    power = (np.abs(np.fft.fft2(x)) ** 2).mean(0)
    f = np.fft.fftfreq(n) * n
    radius = np.rint(np.hypot(f[:, None], f[None, :])).astype(int)
    total = np.bincount(radius.ravel(), power.ravel())
    counts = np.bincount(radius.ravel())
    return (total / np.maximum(counts, 1))[1 : n // 2]


def w1(a, b):
    """Wasserstein-1 distance between two samples, through their quantiles."""
    q = np.linspace(0.0005, 0.9995, 1000)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def describe(meters, real, km_per_px, target=None):
    slope, slope_real = slopes(meters, km_per_px), slopes(real, km_per_px)
    spec, spec_real = spectrum(meters), spectrum(real)
    k = np.arange(1, SIZE // 2)
    fine = k >= SIZE // 8  # wavelengths up to 8 px
    middle = (k >= SIZE // 32) & (k < SIZE // 8)  # 8 to 32 px
    stats = {
        "land": float((meters > 0).mean()),
        "height_w1_m": w1(meters, real),
        "mean_slope_deg": float(slope.mean()),
        "slope_ratio": float(slope.mean() / slope_real.mean()),
        "slope_w1_deg": w1(slope, slope_real),
        # Mean log10 power ratio against real; 0 is right, -1 is ten times too little.
        "spectrum_fine_log10": float(np.log10(spec[fine] / spec_real[fine]).mean()),
        "spectrum_middle_log10": float(np.log10(spec[middle] / spec_real[middle]).mean()),
        "tile_slope_spread_deg": float(slope.mean((1, 2)).std()),
    }
    if target is not None:
        stats["rmse_m"] = float(np.sqrt(((meters - target) ** 2).mean()))
    return stats, slope, spec


def noise_use(refiner, coarse_m, base):
    """Correlation of the added detail between two renders that differ in the latent, noise or both."""
    up = upscale(coarse_m, "bilinear")
    detail = refine(refiner, coarse_m, 1, 2) - up

    def corr(seed_latent, seed_noise):
        other = refine(refiner, coarse_m, seed_latent, seed_noise) - up
        a, b = detail - detail.mean(), other - other.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    changed = refine(refiner, coarse_m, 3, 4)
    return {
        "detail_corr_new_latent": corr(3, 2),
        "detail_corr_new_noise": corr(1, 4),
        "detail_corr_new_both": corr(3, 4),
        # How far apart two random renders are, against how far either is from bilinear.
        "change_rms_m": float(np.sqrt(((changed - base) ** 2).mean())),
        "detail_rms_m": float(np.sqrt((detail**2).mean())),
    }


def fbm(real, km_per_px, rng):
    """Fractal noise tiles: the real mean power-law slope, each scaled to a random real tile."""
    k = np.arange(1, SIZE // 2)
    slope = np.polyfit(np.log(k), np.log(spectrum(real)), 1)[0]
    f = np.fft.fftfreq(SIZE)
    radius = np.hypot(f[:, None], f[None, :])
    amplitude = np.where(radius > 0, radius, np.inf) ** (slope / 2)
    tiles = []
    for _ in range(len(real)):
        phase = np.fft.fft2(rng.standard_normal((SIZE, SIZE)))
        field = np.fft.ifft2(phase * amplitude).real
        ref = real[rng.integers(len(real))]
        tiles.append(ref.mean() + (field - field.mean()) / field.std() * ref.std())
    return np.stack(tiles).astype(np.float32)


NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def fill_depressions(z):
    """Priority-flood (Barnes et al. 2014): raise pits until all land drains to the sea or the edge."""
    h, w = z.shape
    filled = z.astype(np.float64)
    done = np.zeros((h, w), bool)
    outlet = z <= 0
    outlet[0, :] = outlet[-1, :] = outlet[:, 0] = outlet[:, -1] = True
    heap = [(filled[r, c], r, c) for r, c in zip(*np.nonzero(outlet))]
    done[outlet] = True
    heapq.heapify(heap)
    while heap:
        v, r, c = heapq.heappop(heap)
        for dr, dc in NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not done[rr, cc]:
                done[rr, cc] = True
                if filled[rr, cc] <= v:
                    filled[rr, cc] = v + 1e-3  # a tiny gradient so that flats drain too
                heapq.heappush(heap, (filled[rr, cc], rr, cc))
    return filled


def drainage_area(filled, land):
    """D8 flow accumulation in cells; water stops at the sea and the tile edge."""
    h, w = filled.shape
    padded = np.pad(filled, 1, constant_values=np.inf)
    drops = np.stack([
        (filled - padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]) / np.hypot(dr, dc) for dr, dc in NEIGHBOURS
    ])  # fmt: skip
    best = drops.argmax(0)
    flows = (drops.max(0) > 0) & land
    acc = land.astype(np.float64).ravel()
    target = (np.arange(h)[:, None] + np.array(NEIGHBOURS)[best][..., 0]) * w + (
        np.arange(w)[None, :] + np.array(NEIGHBOURS)[best][..., 1]
    )
    target, flows = target.ravel(), flows.ravel()
    for i in np.argsort(-filled, axis=None):
        if flows[i]:
            acc[target[i]] += acc[i]
    return acc.reshape(h, w)


def hydrology(tiles):
    """Pooled drainage areas of land cells, and how much land lies in closed depressions."""
    areas, depth, pits = [], [], []
    for z in tiles:
        land = z > 0
        filled = fill_depressions(z)
        areas.append(drainage_area(filled, land)[land])
        depth.append((filled - z)[land])
    areas, depth = np.concatenate(areas), np.concatenate(depth)
    a = np.unique(np.geomspace(4, 400, 20).astype(int))
    ccdf = np.array([(areas >= v).mean() for v in a])
    return {
        "depression_share": float((depth > 1.0).mean()),
        "mean_fill_m": float(depth.mean()),
        # Real river networks: P(area >= a) ~ a^-0.43 (Rodriguez-Iturbe and Rinaldo).
        "area_exponent": float(np.polyfit(np.log(a), np.log(ccdf), 1)[0]),
        "river_share": float((areas >= 100).mean()),
        "ccdf_a": a.tolist(),
        "ccdf": ccdf.tolist(),
    }


def pair_similarity(tiles, rng, pairs=4000, side=32):
    """A8: correlation of random tile pairs, best of 8 orientations, after shrinking to side px.

    Repeating the same few forms across worlds and places would show as higher values than real.
    """
    n = len(tiles)
    small = torch.from_numpy(meters_to_norm(tiles).astype(np.float32))[:, None]
    small = F.adaptive_avg_pool2d(small, side)[:, 0].numpy()
    small = small - small.mean((1, 2), keepdims=True)
    small /= np.linalg.norm(small, axis=(1, 2), keepdims=True) + 1e-8
    views = np.stack([np.rot90(small, k, axes=(1, 2)) for k in range(4)])
    views = np.concatenate([views, views[:, :, :, ::-1]])  # (8, n, side, side)
    a, b = rng.integers(n, size=pairs), rng.integers(n, size=pairs)
    keep = a != b
    best = (views[:, b[keep]] * small[a[keep]][None]).sum((2, 3)).max(0)
    return {"median": float(np.median(best)), "p90": float(np.quantile(best, 0.9)), "above_0.9": float((best > 0.9).mean())}


def variogram(columns, lags):
    """Root mean squared height difference (m) between points lags pixels apart, north-south."""
    return np.array([np.sqrt(np.mean((columns[d:] - columns[:-d]) ** 2)) for d in lags])


def distance_check(cascade, npy, km_per_px, path, max_lat=60.0, worlds=8, side=1024):
    """A6: how height differences grow with distance, real Earth against generated worlds."""
    rng = np.random.default_rng(7)  # the same places for every model
    earth = np.load(npy, mmap_mode="r")
    rows = earth.shape[0]
    half = int(rows * max_lat / 180)
    real = np.asarray(earth[rows // 2 - half : rows // 2 + half], dtype=np.float32)
    lags = np.unique(np.geomspace(1, min(2 * half, side) - 1, 40).astype(int))
    fake = [
        norm_to_meters(cascade.render(0, int(rng.integers(1 << 31)), *(int(v) for v in rng.integers(-(1 << 20), 1 << 20, 2)), side, side, "cpu"))
        for _ in range(worlds)
    ]
    curves = {"real": variogram(real, lags), "generated": np.mean([variogram(f, lags) for f in fake], axis=0)}
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, values in curves.items():
        ax.loglog(lags * km_per_px, values, color=COLOURS[name], label=name, lw=2 if name == "real" else 1.3)
    ax.set(xlabel="distance (km)", ylabel="RMS height difference (m)",
           title=f"Height difference against distance, level 0\n(real: north-south within ±{max_lat:.0f}°)")  # fmt: skip
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"km": (lags * km_per_px).round(1).tolist(), **{k: v.round(1).tolist() for k, v in curves.items()}}


def plot_hydrology(hydro, km_per_px, path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, h in hydro.items():
        area = np.array(h["ccdf_a"]) * km_per_px**2
        label = f"{name} ({h['area_exponent']:.2f})"
        ax.loglog(area, h["ccdf"], color=COLOURS[name], label=label, lw=2 if name == "real" else 1.3)
    ax.set(xlabel="drainage area (km²)", ylabel="share of land cells draining at least this",
           title="Drainage areas at the finest level (slope of the fit)")  # fmt: skip
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot(level, km_per_px, curves, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    q = np.linspace(0, 1, 501)
    k = np.arange(1, SIZE // 2)
    wavelength = SIZE * km_per_px / k
    for name, (meters, slope, spec) in curves.items():
        style = dict(color=COLOURS[name], label=name, lw=2 if name == "real" else 1.3)
        axes[0].plot(q * 100, np.quantile(meters, q), **style)
        hist, edges = np.histogram(slope, bins=80, range=(0, np.quantile(curves["real"][1], 0.995)), density=True)
        axes[1].plot((edges[1:] + edges[:-1]) / 2, hist, **style)
        axes[2].loglog(wavelength, spec, **style)
    axes[0].set(xlabel="share of area below (%)", ylabel="height (m)", title="Hypsometric curve")
    axes[1].set(xlabel="slope (degrees)", ylabel="density", title="Slope distribution")
    axes[2].set(xlabel="wavelength (km)", ylabel="power", title="Radial power spectrum")
    axes[2].invert_xaxis()
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"Level {level} ({NAMES[level] if level < len(NAMES) else level}, {km_per_px:.2f} km/px), {SIZE} px tiles")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", default=["models/world.pt", "models/regional.pt", "models/local.pt"])
    p.add_argument("--data", default="data/ETOPO_2022_v1_30s_N90W180_bed.tif")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--tiles", type=int, default=256)
    p.add_argument("--min-land", type=float, default=0.3, help="as in training; generated tiles are filtered the same way")
    p.add_argument("--hydro-tiles", type=int, default=128, help="tiles for the slower hydrology check")
    p.add_argument("--out", default="docs/figures")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    cascade = load_cascade(args.ckpt, "cpu")
    levels = len(cascade.models)
    ppd_finest = cascade.px_per_degree
    npy_full, ppd_full = cache_raster(args.data, args.cache_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    report = {}

    for level in range(levels):
        factor = round(ppd_full / ppd_finest) * LEVEL_SCALE ** (levels - 1 - level)
        npy = cache_downsampled(npy_full, factor) if factor > 1 else npy_full
        ppd = ppd_full / factor
        km_per_px = 111.32 / ppd
        coarse, real = real_pairs(npy, ppd, args.tiles, 1000 + level, args.min_land, level)
        fake, mismatch = generated(cascade, level, args.tiles, rng, args.min_land)
        methods = {"generated": (fake, None)}
        if level > 0:
            refiner = cascade.models[level]
            methods["refined real"] = (refine(refiner, coarse, 1, 2), real)
            methods["bicubic"] = (upscale(coarse, "bicubic"), real)
            methods["bilinear"] = (upscale(coarse, "bilinear"), real)
        methods["fBm noise"] = (fbm(real, km_per_px, rng), None)

        _, real_slope, real_spec = describe(real, real, km_per_px)
        curves = {"real": (real, real_slope, real_spec)}
        entry = {"km_per_px": km_per_px, "real": describe(real, real, km_per_px)[0]}
        for name, (meters, target) in methods.items():
            entry[name], slope, spec = describe(meters, real, km_per_px, target)
            curves[name] = (meters, slope, spec)
        if level == 0:
            entry["similarity"] = {name: pair_similarity(meters, np.random.default_rng(3)) for name, (meters, _, _) in curves.items()}
            entry["distance"] = distance_check(cascade, npy, km_per_px, out / "variogram.png")
        else:
            entry["generated"]["rms_vs_level_above_m"] = mismatch
            entry["noise_use"] = noise_use(cascade.models[level], coarse, methods["refined real"][0])
        plot(level, km_per_px, curves, out / f"stats_level{level}.png")
        if level == levels - 1 and km_per_px < 2:  # drainage needs fine pixels
            hydro = {name: hydrology(tiles[: args.hydro_tiles]) for name, (tiles, _, _) in curves.items()}
            entry["hydrology"] = hydro
            plot_hydrology(hydro, km_per_px, out / "hydrology.png")
        report[f"level{level}"] = entry

        print(f"\nlevel {level}, {km_per_px:.2f} km/px, {args.tiles} tiles")
        keys = ["land", "rmse_m", "height_w1_m", "mean_slope_deg", "slope_ratio", "slope_w1_deg",
                "spectrum_middle_log10", "spectrum_fine_log10", "tile_slope_spread_deg"]  # fmt: skip
        print(f"{'':14}" + "".join(f"{k[:13]:>14}" for k in keys))
        for name in ["real", *methods]:
            row = entry[name]
            print(f"{name:14}" + "".join(f"{row[k]:14.3f}" if k in row else f"{'-':>14}" for k in keys))
        if level == 0:
            print("pair similarity:", {k: {m: round(v, 3) for m, v in x.items()} for k, x in entry["similarity"].items()})
            d = entry["distance"]
            print("RMS height difference by distance:", ", ".join(
                f"{km:.0f} km: {r:.0f}/{g:.0f} m" for km, r, g in list(zip(d["km"], d["real"], d["generated"]))[::6]))  # fmt: skip
        else:
            print(f"generated vs level above, block-mean RMS: {mismatch:.1f} m")
            print("noise use:", {k: round(v, 3) for k, v in entry["noise_use"].items()})
        if "hydrology" in entry:
            print("hydrology (land cells):")
            for name, h in entry["hydrology"].items():
                print(f"  {name:14} in depressions {h['depression_share']:.3f}, mean fill {h['mean_fill_m']:.1f} m, "
                      f"area exponent {h['area_exponent']:.3f}, river cells {h['river_share']:.4f}")  # fmt: skip

    (out / "measure.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
