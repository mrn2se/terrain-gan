"""Check whether a generator copies its training data.

For generated tiles, finds the best match anywhere on the real Earth (in all
rotations and flips) by normalised correlation. Real crops, excluding their own
location, give the baseline: fakes matching much better than that are copies.

For a refiner, give the whole cascade down to it. Its tiles are compared by their detail
only, the tile minus a blurred copy at the scale of the level above, since their overall
shape comes from that level.

Usage: uv run evaluate.py --ckpt models/world.pt --npy cache/..._x16.npy
       uv run evaluate.py --ckpt models/world.pt models/regional.pt models/local.pt --npy cache/....npy
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from generate import load_cascade
from terrain_gan import meters_to_norm, norm_to_meters, relief_image, save_image_grid

BAND_LATS = (0, 30, -30, 50, -50, 65)


@dataclass
class Band:
    image: np.ndarray
    top: int  # first row in the full raster
    x_scale: float  # band columns per raster column


def detail(image, factor):
    """image minus its block means, upsampled bilinearly: what a refiner of this factor adds."""
    if factor <= 1:
        return image
    h, w = image.shape
    small = Image.fromarray(image, mode="F").resize((max(w // factor, 1), max(h // factor, 1)), Image.BOX)
    return image - np.asarray(small.resize((w, h), Image.BILINEAR))


def earth_bands(earth, ppd, size, factor=1):
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
        bands.append(Band(detail(image, factor).astype(np.float32), top, width / earth_cols))
    return bands


_band_cache = {}


def band_transforms(band, size):
    """Local sums and variances of band for size x size windows, and its transform."""
    key = (id(band), size)
    if key not in _band_cache:
        b = torch.from_numpy(band).double()
        ft = torch.fft.rfft2(b)
        ones = torch.fft.rfft2(torch.ones(size, size, dtype=torch.float64), b.shape)
        local_sum = torch.fft.irfft2(ft * ones, b.shape)[size - 1 :, size - 1 :]
        local_sq = torch.fft.irfft2(torch.fft.rfft2(b**2) * ones, b.shape)[size - 1 :, size - 1 :]
        var = torch.clamp(local_sq - local_sum**2 / size**2, min=1e-3 * size**2 * 0.01)
        _band_cache[key] = ft, var
    return _band_cache[key]


def correlation_map(band, tile):
    """Normalised cross-correlation of tile at every position in band."""
    t = torch.from_numpy(np.ascontiguousarray(tile - tile.mean())).double()
    size = tile.shape[0]
    ft, var = band_transforms(band, size)
    kernel = torch.fft.rfft2(t.flip(0, 1), band.shape)
    num = torch.fft.irfft2(ft * kernel, band.shape)[size - 1 :, size - 1 :]
    return (num / torch.sqrt(var * (t**2).sum())).numpy()


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
    p.add_argument("--ckpt", nargs="+", required=True, help="world generator, then any refiners; the last is checked")
    p.add_argument("--npy", required=True, help="cached raster used for training")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--count", type=int, default=24)
    p.add_argument("--closest", help="save the closest generated/real pairs to this image")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=0, help="CPU threads for the transforms; 0 keeps the default")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    cascade = load_cascade(args.ckpt, args.device)
    level, ppd = len(cascade.models) - 1, cascade.px_per_degree
    factor = cascade.models[level].stride if level else 1  # coarse pixels are 4 of this level
    earth = meters_to_norm(np.load(args.npy).astype(np.float32)).astype(np.float32)
    bands = earth_bands(earth, ppd, args.size, factor)
    rng = np.random.default_rng(0)

    def varied(tile):
        return tile.std() > 0.05  # flat tiles such as open ocean match almost anything

    fake, real = [], []
    seed = 0
    while len(fake) < args.count:
        seed += 1
        tile = cascade.render(level, seed, 0, 0, args.size, args.size, args.device)
        if varied(tile):
            fake.append(best_match(bands, detail(tile, factor)))
    while len(real) < args.count:
        lat = float(rng.uniform(-65, 65))
        r = int((90 - lat) * ppd) - args.size // 2
        c = int(rng.integers(earth.shape[1] - args.size))
        tile = earth[r : r + args.size, c : c + args.size]
        if varied(tile):
            real.append(best_match(bands, detail(tile, factor), exclude=(r, c)))

    for name, results in (("generated", fake), ("real baseline", real)):
        values = [score for score, _ in results]
        print(
            f"{name}: median best match {np.median(values):.3f}, highest {np.max(values):.3f}, "
            f"above 0.95: {np.mean(np.array(values) > 0.95):.0%}"
        )
    if args.closest:
        ranked = sorted(fake, key=lambda item: -item[0])[:6]
        if factor > 1:  # detail only, scaled to a 300 m spread around 500 m so it shades as land
            images = [
                relief_image(500.0 + 300.0 * t / max(float(t.std()), 1e-6), 111.32 / ppd)
                for _, pair in ranked
                for t in pair
            ]
        else:
            images = [relief_image(norm_to_meters(t), 111.32 / ppd) for _, pair in ranked for t in pair]
        save_image_grid(images, args.closest, columns=2)


if __name__ == "__main__":
    main()
