"""Check that independently rendered tiles agree exactly. Usage: uv run test_seamless.py"""

import numpy as np
import torch

from terrain_gan import Cascade, Generator, Refiner, render


def main():
    torch.manual_seed(0)
    gen = Generator().eval()
    device = "cpu"
    args = (gen, 7)

    for x, y in [(0, 0), (-1037, 523), (123456789, -987654321)]:
        whole = render(*args, x, y, 300, 240, device)
        assert whole.shape == (240, 300)
        left = render(*args, x, y, 157, 240, device)
        right = render(*args, x + 150, y, 150, 240, device)
        below = render(*args, x + 11, y + 93, 200, 147, device)
        assert np.abs(whole[:, :157] - left).max() < 1e-4
        assert np.abs(whole[:, 150:] - right).max() < 1e-4
        assert np.abs(whole[93:, 11:211] - below).max() < 1e-4

    chunked = render(*args, -50, -50, 300, 300, device, chunk=64)
    direct = render(*args, -50, -50, 300, 300, device, chunk=512)
    assert np.abs(chunked - direct).max() < 1e-4

    other = render(gen, 8, 0, 0, 100, 100, device)
    assert np.abs(other - render(*args, 0, 0, 100, 100, device)).mean() > 1e-4
    print(f"ok: seamless (field std {direct.std():.4f})")

    # With the untrained zero output the refiner is plain interpolation, which pins
    # down the alignment: a ramp must keep its value at each pixel's ground position.
    refiner = Refiner().eval()
    ramp = torch.arange(30.0)[None, None, None].repeat(1, 1, 30, 1)
    out = refiner(ramp, torch.zeros(1, refiner.latent_ch, 30, 30), lambda _, __, h, w: torch.zeros(1, 1, h, w))
    assert out.shape[-1] == refiner.output_size(30)
    pixels = refiner.origin + torch.arange(out.shape[-1]) + 0.5
    assert (out[0, 0, 0] - (pixels - 2.0) / 4.0).abs().max() < 1e-3

    # A refiner with non-zero output must keep the seamless property.
    torch.nn.init.normal_(refiner.out.weight, std=0.1)
    cascade = Cascade([gen, refiner], 30.0)
    for x, y in [(0, 0), (-1037, 523), (123456789, -987654321)]:
        whole = cascade.render(1, 7, x, y, 300, 240, device)
        assert whole.shape == (240, 300)
        left = cascade.render(1, 7, x, y, 157, 240, device)
        right = cascade.render(1, 7, x + 150, y, 150, 240, device)
        assert np.abs(whole[:, :157] - left).max() < 1e-4
        assert np.abs(whole[:, 150:] - right).max() < 1e-4
    print("ok: refiner aligned and seamless")


if __name__ == "__main__":
    main()
