"""Convert trained checkpoints for the static page, and write reference data for its tests.

Usage, from the repository root:
    uv run --with onnx --with onnxscript python web/export_models.py --ckpt models/world.pt models/regional.pt models/local.pt
or, for every model set of models/sets.json whose files exist (the page then offers a list):
    uv run --with onnx --with onnxscript python web/export_models.py --sets models/sets.json

The first set (or --ckpt) is written to models/manifest.json and used for the tests; each set
also gets models/sets/<name>.json. A checkpoint shared by several sets is converted once.
With --sets, the paper (PAPER.md and docs/figures) is copied next to the page as well, for
web/paper.html, and each set's paper section, note and starting view go into models/sets.json.
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from generate import load_cascade  # noqa: E402
from terrain_gan import Blur, Generator, Refiner, world_randn  # noqa: E402


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
    latent_ch = model.latent_ch if refiner else model.fine_ch + model.coarse_ch + model.slow_ch
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


def level_entry(model, file):
    layers = export(model, ROOT / file)
    level = dict(file=file, stride=model.stride, origin=model.origin, layers=layers)
    if isinstance(model, Generator):
        level.update(
            kind="generator",
            fine_ch=model.fine_ch,
            coarse_ch=model.coarse_ch,
            coarse_cells=model.coarse_cells,
        )
        if model.slow_ch:
            level.update(slow_ch=model.slow_ch, slow_cells=model.slow_cells)
    else:
        level.update(kind="refiner", latent_ch=model.latent_ch)
    return level


def section_titles(paper):
    """Paper section number -> title, from headings such as "### 6.12 Smoother level 0"."""
    titles = {}
    for line in paper.read_text().splitlines():
        match = re.match(r"#{2,3} (\d+(?:\.\d+)?)\.? (.+)", line)
        if match:
            titles[match.group(1)] = match.group(2)
    return titles


def copy_paper(out):
    """The paper and its figures, for web/paper.html."""
    repo = ROOT.parent
    shutil.copy(repo / "PAPER.md", out / "PAPER.md")
    shutil.copytree(repo / "docs" / "figures", out / "docs" / "figures", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.json"))


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--ckpt", nargs="+", help="world generator, then any refiners")
    group.add_argument("--sets", help="JSON list of model sets, as models/sets.json")
    args = p.parse_args()

    device = "cpu"
    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    if args.ckpt:
        sets = [dict(name="default", label="Default models", ckpt=args.ckpt)]
        # A list left from an earlier --sets export would offer models that are not these.
        (out_dir / "sets.json").unlink(missing_ok=True)
    else:
        sets = json.loads(Path(args.sets).read_text())
        missing = [s["name"] for s in sets if not all(os.path.exists(path) for path in s["ckpt"])]
        if missing:
            print(f"skipped model sets with missing files: {', '.join(missing)}")
        sets = [s for s in sets if s["name"] not in missing]
        (out_dir / "sets").mkdir(exist_ok=True)
        titles = section_titles(ROOT.parent / "PAPER.md")

    converted = {}  # checkpoint path -> manifest level entry
    listing = []
    for index, model_set in enumerate(sets):
        loaded = load_cascade(model_set["ckpt"], device)
        levels = []
        for path, model in zip(model_set["ckpt"], loaded.models):
            if path not in converted:
                if args.ckpt:
                    file = f"models/level{len(converted)}.onnx"
                else:  # named after the checkpoint, e.g. models/models_experiments_local_blur.onnx
                    file = f"models/{re.sub(r'[^A-Za-z0-9]+', '_', Path(path).with_suffix('').as_posix())}.onnx"
                converted[path] = level_entry(model, file)
            levels.append(converted[path])
        manifest = dict(px_per_degree=loaded.px_per_degree, levels=levels)
        if index == 0:
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
            cascade = loaded
        if args.sets:
            (out_dir / "sets" / f"{model_set['name']}.json").write_text(json.dumps(manifest, indent=1))
            entry = dict(name=model_set["name"], label=model_set["label"])
            for key in ("section", "note", "view"):
                if key in model_set:
                    entry[key] = model_set[key]
            if "section" in model_set:
                entry["title"] = titles.get(model_set["section"], "")
            listing.append(entry)
    if args.sets:
        (out_dir / "sets.json").write_text(json.dumps(listing, indent=1, ensure_ascii=False))
        copy_paper(ROOT)

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
            values=gen.latents(7, -13, 5, 6, 7).ravel().tolist(),
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
    for path in sorted(out_dir.glob("*.onnx")):
        print(f"{path.name}: {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
