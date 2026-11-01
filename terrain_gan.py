"""Model, deterministic world noise, tile rendering and terrain visualisation.

The world is an infinite pixel plane. Every latent vector and noise value is a
hash of (seed, position), so any tile can be computed on its own. The generator
is purely convolutional with no padding and no normalisation layers, so each
output pixel depends only on nearby world values and neighbouring tiles agree.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Elevation <-> network range. asinh keeps sea level at 0 and gives lowlands
# more range than a linear scale would.
HEIGHT_SCALE_M = 300.0
HEIGHT_NORM = math.asinh(9000.0 / HEIGHT_SCALE_M)


def meters_to_norm(h):
    return np.arcsinh(h / HEIGHT_SCALE_M) / HEIGHT_NORM


def norm_to_meters(x):
    return np.sinh(x * HEIGHT_NORM) * HEIGHT_SCALE_M


# ---------------------------------------------------------------- world noise

_MASK = (1 << 64) - 1


def _mix(x):
    """splitmix64 finaliser on a uint64 array."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def world_randn(seed, tag, channels, xs, ys):
    """Standard normal values of shape (channels, len(ys), len(xs)) at lattice points."""
    key = (int(seed) * 0x9E3779B97F4A7C15 + int(tag) * 0xD1342543DE82EF95) & _MASK
    xs = np.asarray(xs, dtype=np.int64).astype(np.uint64)[None, None, :]
    ys = np.asarray(ys, dtype=np.int64).astype(np.uint64)[None, :, None]
    ch = np.arange(channels, dtype=np.uint64)[:, None, None]
    state = _mix(np.uint64(key) ^ (ch * np.uint64(0xA24BAED4963EE407)))
    state = _mix(state ^ (xs * np.uint64(0x9FB21C651E98DF25)))
    state = _mix(state ^ (ys * np.uint64(0xC13FA9A902A6328F)))
    a = _mix(state + np.uint64(1))
    b = _mix(state + np.uint64(2))
    u1 = (a.astype(np.float64) + 1.0) / 2.0**64
    u2 = b.astype(np.float64) / 2.0**64
    return (np.sqrt(-2.0 * np.log(np.minimum(u1, 1.0))) * np.cos(2.0 * np.pi * u2)).astype(np.float32)


def world_latents(seed, ix0, iy0, nx, ny, fine_ch, coarse_ch, coarse_cells):
    """Latent grid: per-cell detail vectors plus a smooth large-scale field."""
    xs, ys = np.arange(ix0, ix0 + nx), np.arange(iy0, iy0 + ny)
    fine = world_randn(seed, 1, fine_ch, xs, ys)

    cx = np.floor_divide(xs, coarse_cells)
    cy = np.floor_divide(ys, coarse_cells)
    lattice_x = np.arange(cx[0], cx[-1] + 2)
    lattice_y = np.arange(cy[0], cy[-1] + 2)
    lattice = world_randn(seed, 2, coarse_ch, lattice_x, lattice_y)
    tx = ((xs - cx * coarse_cells) / coarse_cells).astype(np.float32)[None, None, :]
    ty = ((ys - cy * coarse_cells) / coarse_cells).astype(np.float32)[None, :, None]
    # Smoothstep blending; dividing by the weight norm keeps unit variance everywhere.
    tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
    i, j = cx - cx[0], cy - cy[0]
    v00 = lattice[:, j[:, None], i[None, :]]
    v01 = lattice[:, j[:, None], i[None, :] + 1]
    v10 = lattice[:, j[:, None] + 1, i[None, :]]
    v11 = lattice[:, j[:, None] + 1, i[None, :] + 1]
    w00, w01, w10, w11 = (1 - ty) * (1 - tx), (1 - ty) * tx, ty * (1 - tx), ty * tx
    coarse = (w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11) / np.sqrt(
        w00**2 + w01**2 + w10**2 + w11**2
    )
    return np.concatenate([fine, coarse], axis=0)


# --------------------------------------------------------------------- models


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.strength = nn.Parameter(torch.full((1, channels, 1, 1), 0.05))

    def forward(self, x, noise):
        return x + self.strength * noise


class Blur(nn.Module):
    """Fixed unpadded [1 2 1] filter; turns nearest upsampling into bilinear."""

    def __init__(self):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        self.register_buffer("kernel", (k[None] * k[:, None] / 16.0)[None, None])

    def forward(self, x):
        return F.conv2d(x, self.kernel.repeat(x.shape[1], 1, 1, 1), groups=x.shape[1])


class Generator(nn.Module):
    """Latent grid -> height field. Each stage doubles resolution.

    One stage maps n samples to 2n - 6: three unpadded 3x3 filters per stage.
    """

    stages = 4
    stride = 16  # output pixels per latent cell
    # World pixel of the first output sample, relative to 16 * first latent cell.
    # Each stage moves it by -S/4 (upsampling) + 3 * S/2 (unpadded filters) for
    # S = 16, 8, 4, 2, which sums to 37.5, i.e. the pixel whose centre is at 37.5.
    origin = 37

    def __init__(self, fine_ch=48, coarse_ch=16, coarse_cells=8, channels=(192, 128, 64, 32)):
        super().__init__()
        self.config = dict(
            fine_ch=fine_ch, coarse_ch=coarse_ch, coarse_cells=coarse_cells, channels=tuple(channels)
        )
        self.fine_ch, self.coarse_ch, self.coarse_cells = fine_ch, coarse_ch, coarse_cells
        latent_ch = fine_ch + coarse_ch
        self.mapping = nn.Sequential(
            nn.Conv2d(latent_ch, channels[0], 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels[0], channels[0], 1),
            nn.LeakyReLU(0.2),
        )
        self.blur = Blur()
        self.convs, self.injections = nn.ModuleList(), nn.ModuleList()
        prev = channels[0]
        for ch in channels:
            self.convs.append(nn.ModuleList([nn.Conv2d(prev, ch, 3), nn.Conv2d(ch, ch, 3)]))
            self.injections.append(nn.ModuleList([NoiseInjection(ch), NoiseInjection(ch)]))
            prev = ch
        self.out = nn.Conv2d(prev, 1, 1)

    @staticmethod
    def output_size(cells):
        for _ in range(Generator.stages):
            cells = 2 * cells - 6
        return cells

    @staticmethod
    def cells_for(pixels):
        cells = 6
        while Generator.output_size(cells) < pixels:
            cells += 1
        return cells

    def forward(self, latent, noise_fn):
        """noise_fn(layer, stride, height, width) returns one noise channel per batch item."""
        x = self.mapping(latent)
        layer, stride = 0, self.stride
        for convs, injections in zip(self.convs, self.injections):
            stride //= 2
            x = self.blur(F.interpolate(x, scale_factor=2, mode="nearest"))
            for conv, inject in zip(convs, injections):
                x = conv(x)
                x = F.leaky_relu(inject(x, noise_fn(layer, stride, x.shape[2], x.shape[3])), 0.2)
                # Per-pixel, so unlike batch or instance norm it keeps tiles independent.
                x = x / (x.square().mean(1, keepdim=True) + 1e-8).sqrt()
                layer += 1
        return self.out(x)


class Refiner(nn.Module):
    """Coarse height field + noise -> height field at 4x the resolution.

    The network only adds detail on top of the upsampled coarse input. Like the
    Generator it has no padding, so neighbouring tiles agree. n coarse samples
    give 4n - 34 output pixels; coarse pixel i covers output pixels 4i .. 4i + 3.
    """

    stride = 4  # output pixels per coarse pixel
    # Centre of the first output sample relative to 4 * first coarse pixel: 2 (coarse
    # pixel centre) + 8 (two unpadded coarse filters) + 5 + 2.5 (the two stages) = 17.5.
    origin = 17

    def __init__(self, latent_ch=16, channels=(128, 96, 48)):
        super().__init__()
        self.config = dict(latent_ch=latent_ch, channels=tuple(channels))
        self.latent_ch = latent_ch
        self.head = nn.ModuleList(
            [nn.Conv2d(1 + latent_ch, channels[0], 3), nn.Conv2d(channels[0], channels[0], 3)]
        )
        self.blur = Blur()
        self.convs, self.injections = nn.ModuleList(), nn.ModuleList()
        prev = channels[0]
        for ch in channels[1:]:
            self.convs.append(nn.ModuleList([nn.Conv2d(prev, ch, 3), nn.Conv2d(ch, ch, 3)]))
            self.injections.append(nn.ModuleList([NoiseInjection(ch), NoiseInjection(ch)]))
            prev = ch
        self.out = nn.Conv2d(prev, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def output_size(cells):
        return 4 * cells - 34

    @staticmethod
    def cells_for(pixels):
        cells = 9
        while Refiner.output_size(cells) < pixels:
            cells += 1
        return cells

    def upsampled(self, coarse):
        """The coarse input on the output pixel grid; what the network adds detail to."""
        x = coarse[:, :, 2:-2, 2:-2]
        for _ in self.convs:
            x = self.blur(F.interpolate(x, scale_factor=2, mode="nearest"))[:, :, 2:-2, 2:-2]
        return x

    def forward(self, coarse, latent, noise_fn):
        x = torch.cat([coarse, latent], 1)
        for conv in self.head:
            x = F.leaky_relu(conv(x), 0.2)
        layer, stride = 0, self.stride
        for convs, injections in zip(self.convs, self.injections):
            stride //= 2
            x = self.blur(F.interpolate(x, scale_factor=2, mode="nearest"))
            for conv, inject in zip(convs, injections):
                x = conv(x)
                x = F.leaky_relu(inject(x, noise_fn(layer, stride, x.shape[2], x.shape[3])), 0.2)
                layer += 1
        return self.upsampled(coarse) + self.out(x)


class DiscBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        y = F.leaky_relu(self.conv1(x), 0.2)
        y = F.leaky_relu(self.conv2(y), 0.2)
        return F.avg_pool2d(y + self.skip(x), 2)


class Discriminator(nn.Module):
    def __init__(self, size, base=32, max_ch=256, conditional=False):
        super().__init__()
        self.size = size
        blocks = int(math.log2(size)) - 2
        self.from_input = nn.Conv2d(5 if conditional else 3, base, 1)
        chs = [min(base * 2**i, max_ch) for i in range(blocks + 1)]
        self.blocks = nn.Sequential(*[DiscBlock(a, b) for a, b in zip(chs, chs[1:])])
        self.final = nn.Conv2d(chs[-1] + 1, chs[-1], 3, padding=1)
        self.linear = nn.Sequential(nn.Linear(chs[-1] * 16, chs[-1]), nn.LeakyReLU(0.2), nn.Linear(chs[-1], 1))

    def forward(self, h, condition=None):
        # Slopes are small in absolute value but carry the look of real terrain,
        # so the critic gets them as explicit input channels.
        gx = F.pad(h[:, :, :, 1:] - h[:, :, :, :-1], (0, 1, 0, 0))
        gy = F.pad(h[:, :, 1:, :] - h[:, :, :-1, :], (0, 0, 0, 1))
        channels = [h, gx * 10, gy * 10]
        if condition is not None:
            # The detail is small next to the height itself, so it gets its own channel.
            channels += [condition, (h - condition) * 10]
        x = F.leaky_relu(self.from_input(torch.cat(channels, 1)), 0.2)
        x = self.blocks(x)
        std = x.std(0, unbiased=False).mean().expand(x.shape[0], 1, *x.shape[2:])
        x = F.leaky_relu(self.final(torch.cat([x, std], 1)), 0.2)
        return self.linear(x.flatten(1))


# ------------------------------------------------------------------ rendering


@torch.no_grad()
def render(gen, seed, x0, y0, width, height, device, chunk=512):
    """Normalised height field for world pixels [x0, x0+width) x [y0, y0+height)."""
    if width > chunk or height > chunk:
        rows = []
        for y in range(y0, y0 + height, chunk):
            h = min(chunk, y0 + height - y)
            rows.append(
                np.concatenate(
                    [
                        render(gen, seed, x, y, min(chunk, x0 + width - x), h, device, chunk)
                        for x in range(x0, x0 + width, chunk)
                    ],
                    axis=1,
                )
            )
        return np.concatenate(rows, axis=0)

    ix0 = (x0 - gen.origin) // gen.stride
    iy0 = (y0 - gen.origin) // gen.stride
    nx = gen.cells_for(x0 + width - (ix0 * gen.stride + gen.origin))
    ny = gen.cells_for(y0 + height - (iy0 * gen.stride + gen.origin))
    latent = world_latents(seed, ix0, iy0, nx, ny, gen.fine_ch, gen.coarse_ch, gen.coarse_cells)
    latent = torch.from_numpy(latent)[None].to(device)

    def noise_fn(layer, stride, h, w):
        xs = ix0 * gen.stride // stride + np.arange(w)
        ys = iy0 * gen.stride // stride + np.arange(h)
        return torch.from_numpy(world_randn(seed, 100 + layer, 1, xs, ys))[None].to(device)

    out = gen(latent, noise_fn)[0, 0].cpu().numpy()
    left = x0 - (ix0 * gen.stride + gen.origin)
    top = y0 - (iy0 * gen.stride + gen.origin)
    return out[top : top + height, left : left + width]


@torch.no_grad()
def render_refined(refiner, level, seed, x0, y0, width, height, device, coarse_fn, chunk=512):
    """Like render, for a Refiner. coarse_fn(x, y, w, h) gives the level above as an array."""
    if width > chunk or height > chunk:
        rows = []
        for y in range(y0, y0 + height, chunk):
            h = min(chunk, y0 + height - y)
            row = [
                render_refined(refiner, level, seed, x, y, min(chunk, x0 + width - x), h, device, coarse_fn, chunk)
                for x in range(x0, x0 + width, chunk)
            ]
            rows.append(np.concatenate(row, axis=1))
        return np.concatenate(rows, axis=0)

    ix0 = (x0 - refiner.origin) // refiner.stride
    iy0 = (y0 - refiner.origin) // refiner.stride
    nx = refiner.cells_for(x0 + width - (ix0 * refiner.stride + refiner.origin))
    ny = refiner.cells_for(y0 + height - (iy0 * refiner.stride + refiner.origin))
    coarse = torch.from_numpy(coarse_fn(ix0, iy0, nx, ny).copy())[None, None].to(device)
    xs, ys = np.arange(ix0, ix0 + nx), np.arange(iy0, iy0 + ny)
    latent = torch.from_numpy(world_randn(seed, 200 + level, refiner.latent_ch, xs, ys))[None].to(device)

    def noise_fn(layer, stride, h, w):
        nxs = ix0 * refiner.stride // stride + np.arange(w)
        nys = iy0 * refiner.stride // stride + np.arange(h)
        return torch.from_numpy(world_randn(seed, 1000 * level + layer, 1, nxs, nys))[None].to(device)

    out = refiner(coarse, latent, noise_fn)[0, 0].cpu().numpy()
    left = x0 - (ix0 * refiner.stride + refiner.origin)
    top = y0 - (iy0 * refiner.stride + refiner.origin)
    return out[top : top + height, left : left + width]


class Cascade:
    """A Generator followed by Refiners; level 0 is the coarsest."""

    def __init__(self, models, px_per_degree):
        self.models, self.px_per_degree = models, px_per_degree  # finest level

    def render(self, level, seed, x0, y0, width, height, device):
        if level == 0:
            return render(self.models[0], seed, x0, y0, width, height, device)

        def coarse_fn(x, y, w, h):
            return self.render(level - 1, seed, x, y, w, h, device)

        return render_refined(self.models[level], level, seed, x0, y0, width, height, device, coarse_fn)


# Elevation colour ramp: metres -> RGB.
_RAMP_M = np.array([-11000, -6000, -3000, -200, 0, 1, 100, 400, 1000, 2000, 3500, 5500, 9000], dtype=np.float32)
_RAMP_RGB = np.array(
    [
        [8, 16, 60], [14, 40, 100], [28, 80, 145], [90, 160, 200], [160, 210, 225],
        [70, 130, 75], [100, 150, 85], [160, 175, 105], [190, 165, 110],
        [150, 115, 85], [125, 105, 95], [225, 225, 225], [255, 255, 255],
    ],
    dtype=np.float32,
)  # fmt: skip


def relief_image(meters, km_per_px=0.926, exaggeration=None):
    """Coloured, hill-shaded RGB uint8 image of a height field in metres."""
    meters = np.asarray(meters, dtype=np.float32)
    if exaggeration is None:
        # Slopes flatten as pixels get larger, so shade coarse maps more strongly.
        exaggeration = 8.0 * math.sqrt(max(km_per_px / 0.926, 1.0))
    rgb = np.stack([np.interp(meters, _RAMP_M, _RAMP_RGB[:, c]) for c in range(3)], axis=-1)
    dy, dx = np.gradient(meters * exaggeration, km_per_px * 1000.0)
    normal = np.stack([-dx, dy, np.ones_like(dx)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = np.array([-0.5, 0.5, 0.7071], dtype=np.float32)
    shade = np.clip((normal * light).sum(-1), 0.0, 1.0)
    shade = np.where(meters <= 0, 0.5 + 0.5 * shade, 0.25 + 1.05 * shade)
    return np.clip(rgb * shade[..., None], 0, 255).astype(np.uint8)


def terrain_stats(meters, km_per_px):
    """Per-tile land fraction, height spread (m) and roughness (slope in m/km), for (N, H, W) input."""
    slope = np.abs(np.diff(meters, axis=2)).mean((1, 2)) / km_per_px
    return {"land": (meters > 0).mean((1, 2)), "height_std": meters.std((1, 2)), "roughness": slope}


def save_image_grid(images, path, columns):
    h, w, _ = images[0].shape
    rows = math.ceil(len(images) / columns)
    gap = 4
    canvas = np.full((rows * (h + gap) - gap, columns * (w + gap) - gap, 3), 255, dtype=np.uint8)
    for i, img in enumerate(images):
        y, x = i // columns * (h + gap), i % columns * (w + gap)
        canvas[y : y + h, x : x + w] = img
    Image.fromarray(canvas).save(path)
    return canvas
