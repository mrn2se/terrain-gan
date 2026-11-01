"""Train the terrain GAN.  Usage: uv run train.py [--data file.tif] [--size 128]"""

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.tensorboard import SummaryWriter

from terrain_gan import (
    Discriminator,
    Generator,
    meters_to_norm,
    norm_to_meters,
    relief_image,
    terrain_stats,
    render,
    save_image_grid,
    world_latents,
)


def cache_downsampled(npy, factor):
    """Block-averaged copy of a cached raster, for the coarser levels."""
    path = npy.with_name(f"{npy.stem}_x{factor}.npy")
    if path.exists():
        return path
    print(f"downsampling to {path} ...", flush=True)
    data = np.load(npy, mmap_mode="r")
    rows, cols = data.shape[0] // factor, data.shape[1] // factor
    out = np.empty((rows, cols), dtype=np.int16)
    for r in range(rows):
        block = np.asarray(data[r * factor : (r + 1) * factor, : cols * factor], dtype=np.float32)
        out[r] = np.round(block.reshape(factor, cols, factor).mean((0, 2)))
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, out)
    tmp.rename(path)
    return path


def cache_raster(tif, cache_dir):
    """Decompress the GeoTIFF once into an int16 array that workers can memory-map."""
    tif = Path(tif)
    path = Path(cache_dir) / f"{tif.stem}.npy"
    with rasterio.open(tif) as src:
        px_per_degree = round(1.0 / src.res[0], 6)
        if path.exists():
            return path, px_per_degree
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        print(f"caching {tif.name} to {path} ...", flush=True)
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int16, shape=(src.height, src.width))
        for row in range(0, src.height, 256):
            window = ((row, min(row + 256, src.height)), (0, src.width))
            data = src.read(1, window=window)
            data[data == src.nodata] = 0
            out[window[0][0] : window[0][1]] = np.round(data).astype(np.int16)
        out.flush()
        del out
        tmp.rename(path)
    return path, px_per_degree


class TerrainCrops(IterableDataset):
    """Endless random crops, undistorted to about 1:1 ground scale."""

    def __init__(self, npy, px_per_degree, size, max_lat=72.0, min_land=0.3, ocean_keep=0.1):
        self.npy, self.ppd, self.size = str(npy), px_per_degree, size
        self.max_lat, self.min_land, self.ocean_keep = max_lat, min_land, ocean_keep

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng(info.seed if info else None)
        data = np.load(self.npy, mmap_mode="r")
        rows, cols = data.shape
        size = self.size
        while True:
            lat = math.degrees(math.asin(rng.uniform(-1, 1) * math.sin(math.radians(self.max_lat))))
            # Equirectangular pixels get narrower towards the poles, so take more of them.
            width = int(round(size / math.cos(math.radians(lat))))
            row = int((90.0 - lat) * self.ppd) - size // 2
            # Longitude wraps around the date line.
            start = int(rng.integers(cols))
            col = np.arange(start, start + width) % cols
            crop = np.asarray(data[row : row + size][:, col], dtype=np.float32)
            if (crop > 0).mean() < self.min_land and rng.random() > self.ocean_keep:
                continue
            if width != size:
                crop = np.asarray(Image.fromarray(crop, mode="F").resize((size, size), Image.BOX))
            crop = np.rot90(crop, int(rng.integers(4)))
            if rng.random() < 0.5:
                crop = crop[:, ::-1]
            yield torch.from_numpy(meters_to_norm(crop).astype(np.float32).copy())[None]


def sample_fake(gen, batch, size, device, rng):
    """World-sampled latents, cropped at a random phase so the latent lattice is hidden."""
    cells = gen.cells_for(size + gen.stride)
    latents = [
        world_latents(
            int(rng.integers(1 << 62)),
            int(rng.integers(-(1 << 40), 1 << 40)),
            int(rng.integers(-(1 << 40), 1 << 40)),
            cells,
            cells,
            gen.fine_ch,
            gen.coarse_ch,
            gen.coarse_cells,
        )
        for _ in range(batch)
    ]
    latent = torch.from_numpy(np.stack(latents)).to(device)
    out = gen(latent, lambda _, __, h, w: torch.randn(batch, 1, h, w, device=device))
    dy, dx = (int(v) for v in rng.integers(0, gen.stride, 2))
    return out[:, :, dy : dy + size, dx : dx + size]


@torch.no_grad()
def variety_report(ema, real_iter, reference, batch, size, km_per_px, device, rng):
    """Compare how much tiles differ from each other, for generated and real terrain."""
    count = 256 // batch
    if not reference:
        crops = [next(real_iter).numpy()[:, 0] for _ in range(count)]
        reference.update(terrain_stats(norm_to_meters(np.concatenate(crops)), km_per_px))
    tiles = [sample_fake(ema, batch, size, device, rng).cpu().numpy()[:, 0] for _ in range(count)]
    fake = terrain_stats(norm_to_meters(np.concatenate(tiles)), km_per_px)
    report = {}
    for name in fake:
        # Wasserstein distance: 0 when the two distributions over tiles are identical.
        report[f"distance/{name}"] = float(np.abs(np.sort(fake[name]) - np.sort(reference[name])).mean())
        report[f"spread_fake/{name}"] = float(fake[name].std())
        report[f"spread_real/{name}"] = float(reference[name].std())
    return report


def save_samples(ema, reals, step, out_dir, km_per_px, size, device):
    fakes = [norm_to_meters(render(ema, s, 0, 0, size, size, device)) for s in range(24)]
    reals = [norm_to_meters(r[0].numpy()) for r in reals[:8]]
    images = [relief_image(m, km_per_px) for m in fakes + reals]
    return save_image_grid(images, out_dir / f"samples_{step:07d}.png", columns=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/ETOPO_2022_v1_30s_N90W180_bed.tif")
    p.add_argument("--out", default="runs/default")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--downsample", type=int, default=1, help="block-average the data by this factor")
    p.add_argument("--min-land", type=float, default=0.3, help="crops below this land fraction are mostly skipped")
    p.add_argument("--size", type=int, default=128, help="training crop size, power of two")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--r1", type=float, default=10.0, help="R1 penalty weight")
    p.add_argument("--r1-every", type=int, default=8)
    p.add_argument("--no-pixel-norm", action="store_true", help="train the generator without pixel normalisation")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--resume", action="store_true", help="continue from <out>/latest.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy, ppd = cache_raster(args.data, args.cache_dir)
    if args.downsample > 1:
        npy = cache_downsampled(npy, args.downsample)
        ppd /= args.downsample
    km_per_px = 111.32 / ppd
    loader = DataLoader(
        TerrainCrops(npy, ppd, args.size, min_land=args.min_land),
        batch_size=args.batch,
        num_workers=args.workers,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    real_iter = iter(loader)

    gen = Generator(pixel_norm=not args.no_pixel_norm).to(device)
    disc = Discriminator(args.size).to(device)
    ema = copy.deepcopy(gen).eval().requires_grad_(False)
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.0, 0.99))
    rng = np.random.default_rng()

    step = 0
    if args.resume:
        state = torch.load(out_dir / "latest.pt", map_location=device, weights_only=True)
        gen.load_state_dict(state["gen"])
        disc.load_state_dict(state["disc"])
        ema.load_state_dict(state["ema"])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        step = state["step"]
        print(f"resumed at step {step}")

    writer = SummaryWriter(out_dir, purge_step=step)
    print(
        f"G {sum(x.numel() for x in gen.parameters()) / 1e6:.2f}M params, "
        f"D {sum(x.numel() for x in disc.parameters()) / 1e6:.2f}M params, "
        f"{km_per_px:.2f} km/px, device {device}",
        flush=True,
    )

    reference = {}
    start, start_step = time.time(), step
    last_r1 = None
    while step < args.steps:
        # Discriminator: non-saturating logistic loss with a lazy R1 penalty.
        real = next(real_iter).to(device, non_blocking=True)
        with torch.no_grad():
            fake = sample_fake(gen, args.batch, args.size, device, rng)
        d_real, d_fake = disc(real), disc(fake)
        d_loss = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

        if step % args.r1_every == 0:
            real_r1 = real.detach().requires_grad_(True)
            grad = torch.autograd.grad(disc(real_r1).sum(), real_r1, create_graph=True)[0]
            r1_value = grad.square().sum((1, 2, 3)).mean()
            last_r1 = r1_value.item()
            opt_d.zero_grad(set_to_none=True)
            (0.5 * args.r1 * args.r1_every * r1_value).backward()
            opt_d.step()

        # Generator.
        fake = sample_fake(gen, args.batch, args.size, device, rng)
        g_loss = F.softplus(-disc(fake)).mean()
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()

        with torch.no_grad():
            for a, b in zip(ema.parameters(), gen.parameters()):
                a.lerp_(b, 1.0 - args.ema)

        step += 1
        if step % args.log_every == 0:
            rate = (step - start_step) / (time.time() - start)
            print(
                f"step {step} d {d_loss.item():.3f} g {g_loss.item():.3f} "
                f"D(real) {d_real.mean().item():.2f} D(fake) {d_fake.mean().item():.2f} "
                f"fake {fake.mean().item():.3f}+-{fake.std().item():.3f} "
                f"real {real.mean().item():.3f}+-{real.std().item():.3f} {rate:.1f} it/s",
                flush=True,
            )
            writer.add_scalar("loss/discriminator", d_loss.item(), step)
            writer.add_scalar("loss/generator", g_loss.item(), step)
            writer.add_scalar("score/real", d_real.mean().item(), step)
            writer.add_scalar("score/fake", d_fake.mean().item(), step)
            writer.add_scalar("height/fake_mean", fake.mean().item(), step)
            writer.add_scalar("height/fake_std", fake.std().item(), step)
            writer.add_scalar("height/real_mean", real.mean().item(), step)
            writer.add_scalar("height/real_std", real.std().item(), step)
            writer.add_scalar("speed/steps_per_second", rate, step)
            if last_r1 is not None:
                writer.add_scalar("loss/r1", last_r1, step)
            writer.flush()
        if step % args.sample_every == 0:
            # Benchmarking every new tile shape can take most of the GPU memory.
            with torch.backends.cudnn.flags(enabled=True, benchmark=False):
                grid = save_samples(ema, real.cpu(), step, out_dir, km_per_px, args.size, device)
                writer.add_image("samples (last row real)", grid, step, dataformats="HWC")
                world = relief_image(norm_to_meters(render(ema, 0, 0, 0, 1024, 512, device)), km_per_px)
                Image.fromarray(world).save(out_dir / f"world_{step:07d}.png")
                writer.add_image("world", world, step, dataformats="HWC")
                report = variety_report(ema, real_iter, reference, args.batch, args.size, km_per_px, device, rng)
                for name, value in report.items():
                    writer.add_scalar(f"variety/{name}", value, step)
                print(f"variety {step} " + " ".join(f"{k}={v:.3g}" for k, v in report.items()), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            state = {
                "step": step,
                "gen": gen.state_dict(),
                "disc": disc.state_dict(),
                "ema": ema.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "config": gen.config,
                "px_per_degree": ppd,
            }
            torch.save(state, out_dir / "latest.tmp")
            (out_dir / "latest.tmp").rename(out_dir / "latest.pt")
            torch.save(
                {k: state[k] for k in ("step", "ema", "config", "px_per_degree")},
                out_dir / f"generator_{step:07d}.pt",
            )


if __name__ == "__main__":
    main()
