"""Train a refiner: a conditional GAN adding 4x detail to the level above.

Usage: uv run train_refiner.py --downsample 4 --parent runs/world/generator_0020000.pt --out runs/regional
"""

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.tensorboard import SummaryWriter

from generate import load_cascade
from terrain_gan import (
    HEIGHT_NORM,
    HEIGHT_SCALE_M,
    Cascade,
    Discriminator,
    Refiner,
    meters_to_norm,
    norm_to_meters,
    relief_image,
    save_image_grid,
    terrain_stats,
)
from train import cache_downsampled, cache_raster


class RefinerCrops(IterableDataset):
    """Endless (coarse, fine) pairs. The coarse pixels are block means of the fine ones."""

    def __init__(self, npy, px_per_degree, cells, max_lat=72.0, min_land=0.3, ocean_keep=0.1):
        self.npy, self.ppd, self.cells = str(npy), px_per_degree, cells
        self.max_lat, self.min_land, self.ocean_keep = max_lat, min_land, ocean_keep

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng(info.seed if info else None)
        data = np.load(self.npy, mmap_mode="r")
        rows, cols = data.shape
        side = self.cells * Refiner.stride
        while True:
            lat = math.degrees(math.asin(rng.uniform(-1, 1) * math.sin(math.radians(self.max_lat))))
            width = int(round(side / math.cos(math.radians(lat))))
            row = int((90.0 - lat) * self.ppd) - side // 2
            start = int(rng.integers(cols))
            crop = np.asarray(data[row : row + side][:, np.arange(start, start + width) % cols], dtype=np.float32)
            if (crop > 0).mean() < self.min_land and rng.random() > self.ocean_keep:
                continue
            if width != side:
                crop = np.asarray(Image.fromarray(crop, mode="F").resize((side, side), Image.BOX))
            crop = np.rot90(crop, int(rng.integers(4)))
            if rng.random() < 0.5:
                crop = crop[:, ::-1]
            s = Refiner.stride
            coarse = crop.reshape(self.cells, s, self.cells, s).mean((1, 3))
            yield (
                torch.from_numpy(meters_to_norm(coarse).astype(np.float32).copy())[None],
                torch.from_numpy(meters_to_norm(crop).astype(np.float32).copy())[None],
            )


def sample(refiner, coarse, device):
    latent = torch.randn(coarse.shape[0], refiner.latent_ch, *coarse.shape[2:], device=device)
    return refiner(coarse, latent, lambda _, __, h, w: torch.randn(coarse.shape[0], 1, h, w, device=device))


def norm_block_mean(h, stride):
    """Block mean in metres, back in network units: exactly how the coarse data is made."""
    meters = torch.sinh(h * HEIGHT_NORM) * HEIGHT_SCALE_M
    return torch.asinh(F.avg_pool2d(meters, stride) / HEIGHT_SCALE_M) / HEIGHT_NORM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/ETOPO_2022_v1_30s_N90W180_bed.tif")
    p.add_argument("--downsample", type=int, default=4, help="block-average the target data by this factor")
    p.add_argument("--parent", nargs="+", required=True, help="checkpoints of the levels above, for previews")
    p.add_argument("--out", default="runs/regional")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--r1", type=float, default=1.0)
    p.add_argument("--r1-every", type=int, default=8)
    p.add_argument("--match", type=float, default=10.0, help="weight of staying true to the coarse input")
    p.add_argument("--coarse-noise", type=float, default=0.01, help="makes it tolerate imperfect generated input")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
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

    stride = Refiner.stride
    cells = Refiner.cells_for(args.size + stride)
    loader = DataLoader(
        RefinerCrops(npy, ppd, cells),
        batch_size=args.batch,
        num_workers=args.workers,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    data_iter = iter(loader)

    refiner = Refiner().to(device)
    disc = Discriminator(args.size, conditional=True).to(device)
    ema = copy.deepcopy(refiner).eval().requires_grad_(False)
    opt_g = torch.optim.Adam(refiner.parameters(), lr=args.lr, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.0, 0.99))
    parents = load_cascade(args.parent, device).models
    level = len(parents)
    rng = np.random.default_rng()

    step = 0
    if args.resume:
        state = torch.load(out_dir / "latest.pt", map_location=device, weights_only=True)
        refiner.load_state_dict(state["gen"])
        disc.load_state_dict(state["disc"])
        ema.load_state_dict(state["ema"])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        step = state["step"]
        print(f"resumed at step {step}")

    writer = SummaryWriter(out_dir, purge_step=step)
    print(
        f"refiner {sum(x.numel() for x in refiner.parameters()) / 1e6:.2f}M params, "
        f"D {sum(x.numel() for x in disc.parameters()) / 1e6:.2f}M params, "
        f"{km_per_px:.2f} km/px, device {device}",
        flush=True,
    )

    def next_batch():
        coarse, fine = next(data_iter)
        coarse = coarse.to(device, non_blocking=True)
        fine = fine.to(device, non_blocking=True)
        noisy = coarse + args.coarse_noise * torch.randn_like(coarse)
        # Random phase, the same for real and fake, hides the coarse lattice.
        dy, dx = (int(v) for v in rng.integers(0, stride, 2))
        return noisy, fine, dy, dx

    def crop(x, dy, dx):
        return x[:, :, dy : dy + args.size, dx : dx + args.size]

    def real_view(fine, dy, dx):
        return crop(fine[:, :, Refiner.origin :, Refiner.origin :], dy, dx)

    start, start_step = time.time(), step
    last_r1 = None
    while step < args.steps:
        noisy, fine, dy, dx = next_batch()
        condition = crop(refiner.upsampled(noisy), dy, dx)
        real = real_view(fine, dy, dx)
        with torch.no_grad():
            fake = crop(sample(refiner, noisy, device), dy, dx)
        d_real, d_fake = disc(real, condition), disc(fake, condition)
        d_loss = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

        if step % args.r1_every == 0:
            real_r1 = real.detach().requires_grad_(True)
            grad = torch.autograd.grad(disc(real_r1, condition).sum(), real_r1, create_graph=True)[0]
            r1_value = grad.square().sum((1, 2, 3)).mean()
            last_r1 = r1_value.item()
            opt_d.zero_grad(set_to_none=True)
            (0.5 * args.r1 * args.r1_every * r1_value).backward()
            opt_d.step()

        full = sample(refiner, noisy, device)
        fake = crop(full, dy, dx)
        # Output pixel 3 is the first to start a whole coarse block (origin 17 + 3 = 20 = 4 * 5).
        blocks = norm_block_mean(full[:, :, 3:, 3:], stride)
        target = noisy[:, :, 5 : 5 + blocks.shape[2], 5 : 5 + blocks.shape[3]]
        match_loss = (blocks - target).abs().mean()
        adv_loss = F.softplus(-disc(fake, condition)).mean()
        opt_g.zero_grad(set_to_none=True)
        (adv_loss + args.match * match_loss).backward()
        opt_g.step()

        with torch.no_grad():
            for a, b in zip(ema.parameters(), refiner.parameters()):
                a.lerp_(b, 1.0 - args.ema)

        step += 1
        if step % args.log_every == 0:
            rate = (step - start_step) / (time.time() - start)
            print(
                f"step {step} d {d_loss.item():.3f} g {adv_loss.item():.3f} match {match_loss.item():.4f} "
                f"D(real) {d_real.mean().item():.2f} D(fake) {d_fake.mean().item():.2f} {rate:.1f} it/s",
                flush=True,
            )
            writer.add_scalar("loss/discriminator", d_loss.item(), step)
            writer.add_scalar("loss/generator", adv_loss.item(), step)
            writer.add_scalar("loss/match_coarse", match_loss.item(), step)
            writer.add_scalar("score/real", d_real.mean().item(), step)
            writer.add_scalar("score/fake", d_fake.mean().item(), step)
            writer.add_scalar("speed/steps_per_second", rate, step)
            if last_r1 is not None:
                writer.add_scalar("loss/r1", last_r1, step)
            writer.flush()

        if step % args.sample_every == 0:
            with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, benchmark=False):
                # Same coarse input: its upsampling, then the refined output, then the truth.
                ema_fake = crop(sample(ema, noisy, device), dy, dx)
                rows = [condition[:8], ema_fake[:8], real[:8]]
                images = [
                    relief_image(norm_to_meters(t[0].cpu().numpy()), km_per_px) for row in rows for t in row
                ]
                grid = save_image_grid(images, out_dir / f"samples_{step:07d}.png", columns=8)
                writer.add_image("samples (coarse, refined, real)", grid, step, dataformats="HWC")

                cascade = Cascade(parents + [ema], ppd)
                norm = cascade.render(level, 0, 0, 0, 1024, 512, device)
                world = relief_image(norm_to_meters(norm), km_per_px)
                Image.fromarray(world).save(out_dir / f"world_{step:07d}.png")
                writer.add_image("world", world, step, dataformats="HWC")

                fake_stats = terrain_stats(norm_to_meters(ema_fake[:, 0].cpu().numpy()), km_per_px)
                real_stats = terrain_stats(norm_to_meters(real[:, 0].cpu().numpy()), km_per_px)
                ratio = fake_stats["roughness"].mean() / real_stats["roughness"].mean()
                writer.add_scalar("variety/roughness_fake_over_real", ratio, step)
                print(f"roughness fake/real at step {step}: {ratio:.2f}", flush=True)

        if step % args.save_every == 0 or step == args.steps:
            state = {
                "step": step,
                "gen": refiner.state_dict(),
                "disc": disc.state_dict(),
                "ema": ema.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "config": refiner.config,
                "px_per_degree": ppd,
                "refiner": True,
            }
            torch.save(state, out_dir / "latest.tmp")
            (out_dir / "latest.tmp").rename(out_dir / "latest.pt")
            torch.save(
                {k: state[k] for k in ("step", "ema", "config", "px_per_degree", "refiner")},
                out_dir / f"generator_{step:07d}.pt",
            )


if __name__ == "__main__":
    main()
