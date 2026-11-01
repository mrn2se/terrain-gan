"""Render a piece of the endless world from a trained generator.

Usage: uv run generate.py --ckpt runs/world/latest.pt runs/regional/latest.pt --lon 12.5 --lat -40 --size 512 --out tile

--lon/--lat give the centre of the piece on an infinite plane, so values outside
the real -180..180 / -90..90 range work too. They are not places on the real Earth.
--seed selects the world. Writes <out>.png (relief) and <out>.npy (metres).
"""

import argparse

import numpy as np
import torch
from PIL import Image

from terrain_gan import Cascade, Generator, Refiner, norm_to_meters, relief_image


def load_model(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model_class = Refiner if state.get("refiner") else Generator
    model = model_class(**state["config"]).to(device).eval().requires_grad_(False)
    model.load_state_dict(state["ema"])
    return model, state["px_per_degree"]


def load_generator(path, device):
    return load_model(path, device)


def load_cascade(paths, device):
    """paths: the world generator first, then each refiner."""
    loaded = [load_model(path, device) for path in paths]
    return Cascade([model for model, _ in loaded], loaded[-1][1])


def generate(cascade, lon, lat, width, height, seed, device):
    """Height field in metres at the finest level, north up, centred on (lon, lat)."""
    x0 = round(lon * cascade.px_per_degree) - width // 2
    y0 = round(-lat * cascade.px_per_degree) - height // 2
    return norm_to_meters(cascade.render(len(cascade.models) - 1, seed, x0, y0, width, height, device))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, nargs="+", help="world generator, then any refiners")
    p.add_argument("--lon", type=float, default=0.0)
    p.add_argument("--lat", type=float, default=0.0)
    p.add_argument("--size", type=int, default=512, help="pixels per side at the finest level")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="tile")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cascade = load_cascade(args.ckpt, args.device)
    meters = generate(cascade, args.lon, args.lat, args.size, args.size, args.seed, args.device)
    np.save(f"{args.out}.npy", meters.astype(np.float32))
    Image.fromarray(relief_image(meters, 111.32 / cascade.px_per_degree)).save(f"{args.out}.png")
    print(f"{args.out}.png: {meters.min():.0f} m to {meters.max():.0f} m")


if __name__ == "__main__":
    main()
