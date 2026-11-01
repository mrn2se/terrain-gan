"""Convert trained checkpoints for the static page, and write reference data for its tests.

Usage, from the repository root:
    uv run --with onnx --with onnxscript python web/export_models.py --ckpt models/world.pt models/regional.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from generate import load_cascade  # noqa: E402
from terrain_gan import Blur, Generator, Refiner, world_latents, world_randn  # noqa: E402


def constant_channel_blur(self, x):
    # The tracer would make the channel count dynamic, which ONNX Conv cannot take.
    channels = int(x.shape[1])
    return torch.nn.functional.conv2d(x, self.kernel.repeat(channels, 1, 1, 1), groups=channels)


Blur.forward = constant_channel_blur


class Exported(torch.nn.Module):
    """Takes the noise layers as plain inputs instead of through a callback."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        if isinstance(self.model, Refiner):
            coarse, latent, *noises = inputs
            return self.model(coarse, latent, lambda layer, _, __, ___: noises[layer])
        latent, *noises = inputs
        return self.model(latent, lambda layer, _, __, ___: noises[layer])


def export(model, path):
    refiner = isinstance(model, Refiner)
    layers = 4 if refiner else 8
    cells = 12
    size = cells - 4 if refiner else cells
    noises = []
    # Dummy sizes only have to be valid; the axes are dynamic.
    for _ in range(layers // 2):
        size *= 2
        noises += [torch.zeros(1, 1, size - 4, size - 4), torch.zeros(1, 1, size - 6, size - 6)]
        size -= 6
    latent_ch = model.latent_ch if refiner else model.fine_ch + model.coarse_ch
    latent = torch.zeros(1, latent_ch, cells, cells)
    inputs = ([torch.zeros(1, 1, cells, cells)] if refiner else []) + [latent] + noises
    names = (["coarse"] if refiner else []) + ["latent"] + [f"noise{i}" for i in range(layers)]
    axes = {name: {2: f"{name}_h", 3: f"{name}_w"} for name in names}
    axes["height"] = {2: "height_h", 3: "height_w"}
    torch.onnx.export(
        Exported(model).eval(),
        tuple(inputs),
        str(path),
        input_names=names,
        output_names=["height"],
        dynamic_axes=axes,
        opset_version=17,
        dynamo=False,
    )
    return layers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True, help="world generator, then any refiners")
    args = p.parse_args()

    device = "cpu"
    cascade = load_cascade(args.ckpt, device)
    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)

    levels = []
    for i, model in enumerate(cascade.models):
        name = f"level{i}.onnx"
        layers = export(model, out_dir / name)
        level = dict(file=f"models/{name}", stride=model.stride, origin=model.origin, layers=layers)
        if isinstance(model, Generator):
            level.update(
                kind="generator",
                fine_ch=model.fine_ch,
                coarse_ch=model.coarse_ch,
                coarse_cells=model.coarse_cells,
            )
        else:
            level.update(kind="refiner", latent_ch=model.latent_ch)
        levels.append(level)
    manifest = dict(px_per_degree=cascade.px_per_degree, levels=levels)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # What the page must reproduce; compared by web/test/test.html.
    gen = cascade.models[0]
    randn_cases = [
        (0, 1, 3, [0, 1, 2], [0, 1]),
        (7, 101, 1, [-5, -4], [-3, -2]),
        (123456, 2002, 2, [5000000000, 5000000001], [-5000000000]),
    ]
    tile_cases = [(0, 3, -37, 21, 96, 64)] + [(i, 3, 200, -131, 96, 64) for i in range(1, len(cascade.models))]
    reference = dict(
        randn=[
            dict(seed=s, tag=t, channels=c, xs=xs, ys=ys, values=world_randn(s, t, c, xs, ys).ravel().tolist())
            for s, t, c, xs, ys in randn_cases
        ],
        latents=dict(
            seed=7,
            ix0=-13,
            iy0=5,
            nx=6,
            ny=7,
            values=world_latents(7, -13, 5, 6, 7, gen.fine_ch, gen.coarse_ch, gen.coarse_cells).ravel().tolist(),
        ),
        tiles=[
            dict(
                level=level,
                seed=seed,
                x0=x0,
                y0=y0,
                width=w,
                height=h,
                values=cascade.render(level, seed, x0, y0, w, h, device).ravel().tolist(),
            )
            for level, seed, x0, y0, w, h in tile_cases
        ],
    )
    test_dir = ROOT / "test"
    test_dir.mkdir(exist_ok=True)
    (test_dir / "reference.json").write_text(json.dumps(reference))
    for path in sorted(out_dir.iterdir()):
        print(f"{path.name}: {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
