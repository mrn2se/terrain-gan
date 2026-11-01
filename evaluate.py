"""Check whether a generator copies its training data.

For generated tiles, finds the best match anywhere on the real Earth (in all
rotations and flips) by normalised correlation. Real crops, excluding their own
location, give the baseline: fakes matching much better than that are copies.

Usage: uv run evaluate.py --ckpt runs/world/latest.pt --npy cache/..._x16.npy
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from generate import load_generator
from terrain_gan import meters_to_norm, norm_to_meters, relief_image, render, save_image_grid

BAND_LATS = (0, 30, -30, 50, -50, 65)


@dataclass
class Band:
    image: np.ndarray
    top: int  # first row in the full raster
    x_scale: float  # band columns per raster column


def earth_bands(earth, ppd, size):
    """Latitude bands stretched to about 1:1 ground scale, as the training crops are."""
    bands = []
    half = 2 * size
    earth_cols = earth.shape[1]
    for lat in BAND_LATS:
        centre = int((90 - lat) * ppd)
        top = max(centre - half, 0)
        rows = earth[top : centre + half]
        width = round(rows.shape[1] * math.cos(math.radians(lat)))
        image = np.asarray(Image.fromarray(rows, mode="F").resize((width, rows.shape[0]), Image.BOX))
        bands.append(Band(image, top, width / earth_cols))
    return bands


def correlation_map(band, tile):
    """Normalised cross-correlation of tile at every position in band."""
    t = tile - tile.mean()
    size = tile.shape[0]
    ft = np.fft.rfft2(band)
    kernel = np.fft.rfft2(t[::-1, ::-1], band.shape)
    num = np.fft.irfft2(ft * kernel, band.shape)[size - 1 :, size - 1 :]
    ones = np.fft.rfft2(np.ones_like(t), band.shape)
    local_sum = np.fft.irfft2(ft * ones, band.shape)[size - 1 :, size - 1 :]
    local_sq = np.fft.irfft2(np.fft.rfft2(band**2) * ones, band.shape)[size - 1 :, size - 1 :]
    var = np.maximum(local_sq - local_sum**2 / t.size, 1e-3 * t.size * 0.01)
    return num / np.sqrt(var * (t**2).sum())


def best_match(bands, tile, exclude=None):
    """Highest correlation and the orientated tile/matching crop that produced it.

    exclude=(row, col) is the tile's own place in the raster. It is masked in every
    orientation, because bands overlap and a rotated tile can overlap itself.
    """
    best, pair = -1.0, None
    size = tile.shape[0]
    for band in bands:
        for k in range(8):
            oriented = np.rot90(tile, k % 4)
            if k >= 4:
                oriented = oriented[:, ::-1]
            corr = correlation_map(band.image, oriented)
            if exclude is not None:
                row = exclude[0] - band.top
                col = round(exclude[1] * band.x_scale)
                near = round(size * max(band.x_scale, 1.0)) * 2
                corr[max(row - 2 * size, 0) : max(row + 2 * size, 0), max(col - near, 0) : col + near] = -1.0
            r, c = np.unravel_index(int(corr.argmax()), corr.shape)
            if corr[r, c] > best:
                best = float(corr[r, c])
                pair = oriented.copy(), band.image[r : r + size, c : c + size].copy()
    return best, pair


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--npy", required=True, help="cached raster used for training")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--count", type=int, default=24)
    p.add_argument("--closest", help="save the closest generated/real pairs to this image")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    gen, ppd = load_generator(args.ckpt, args.device)
    earth = meters_to_norm(np.load(args.npy).astype(np.float32)).astype(np.float32)
    bands = earth_bands(earth, ppd, args.size)
    rng = np.random.default_rng(0)

    def varied(tile):
        return tile.std() > 0.05  # flat tiles such as open ocean match almost anything

    fake, real = [], []
    seed = 0
    while len(fake) < args.count:
        seed += 1
        tile = render(gen, seed, 0, 0, args.size, args.size, args.device)
        if varied(tile):
            fake.append(best_match(bands, tile))
    while len(real) < args.count:
        lat = float(rng.uniform(-65, 65))
        r = int((90 - lat) * ppd) - args.size // 2
        c = int(rng.integers(earth.shape[1] - args.size))
        tile = earth[r : r + args.size, c : c + args.size]
        if varied(tile):
            real.append(best_match(bands, tile, exclude=(r, c)))

    for name, results in (("generated", fake), ("real baseline", real)):
        values = [score for score, _ in results]
        print(
            f"{name}: median best match {np.median(values):.3f}, highest {np.max(values):.3f}, "
            f"above 0.95: {np.mean(np.array(values) > 0.95):.0%}"
        )
    if args.closest:
        ranked = sorted(fake, key=lambda item: -item[0])[:6]
        images = [relief_image(norm_to_meters(t), 111.32 / ppd) for _, pair in ranked for t in pair]
        save_image_grid(images, args.closest, columns=2)


if __name__ == "__main__":
    main()
