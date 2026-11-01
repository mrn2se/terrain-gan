// Runs the model levels and turns heights into coloured relief. Mirrors terrain_gan.py.
import { worldLatents, worldRandn } from "./noise.js";

export const ORT_VERSION = "1.30.0";
export const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

const HEIGHT_SCALE_M = 300;
const HEIGHT_NORM = Math.asinh(9000 / HEIGHT_SCALE_M);
const CHUNK = 512;

export function toMeters(x) {
  return Math.sinh(x * HEIGHT_NORM) * HEIGHT_SCALE_M;
}

// Fetches the manifest and models. Tries each execution provider until one works.
export async function loadTerrain(ort, providers) {
  const base = new URL("../", import.meta.url);
  const manifest = await (await fetch(new URL("models/manifest.json", base))).json();
  const buffers = await Promise.all(
    manifest.levels.map(async (level) => new Uint8Array(await (await fetch(new URL(level.file, base))).arrayBuffer())),
  );
  let lastError;
  for (const provider of providers) {
    try {
      const sessions = [];
      for (const buffer of buffers) {
        sessions.push(await ort.InferenceSession.create(buffer, { executionProviders: [provider] }));
      }
      return new Terrain(ort, manifest, sessions, provider);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}

function cellsFor(level, pixels) {
  // Output size: generator 16n - 90 (at least 6 cells), refiner 4n - 34 (at least 9).
  const generator = level.kind === "generator";
  const a = generator ? 16 : 4, b = generator ? 90 : 34, min = generator ? 6 : 9;
  return Math.max(min, Math.ceil((pixels + b) / a));
}

function noiseSizes(level, nx, ny) {
  // Each stage doubles and blurs, then runs two unpadded filters that each take noise.
  const sizes = [];
  let w = level.kind === "refiner" ? nx - 4 : nx;
  let h = level.kind === "refiner" ? ny - 4 : ny;
  for (let i = 0; i < level.layers / 2; i++) {
    w = 2 * w - 2;
    h = 2 * h - 2;
    sizes.push([w - 2, h - 2], [w - 4, h - 4]);
    w -= 4;
    h -= 4;
  }
  return sizes;
}

export class Terrain {
  constructor(ort, manifest, sessions, provider) {
    this.ort = ort;
    this.manifest = manifest;
    this.sessions = sessions;
    this.provider = provider;
  }

  // Normalised heights (rows of width values) for one level's world pixels.
  async render(index, seed, x0, y0, width, height) {
    if (width <= CHUNK && height <= CHUNK) return this.renderChunk(index, seed, x0, y0, width, height);
    const out = new Float32Array(width * height);
    for (let y = 0; y < height; y += CHUNK) {
      for (let x = 0; x < width; x += CHUNK) {
        const w = Math.min(CHUNK, width - x), h = Math.min(CHUNK, height - y);
        const part = await this.renderChunk(index, seed, x0 + x, y0 + y, w, h);
        for (let r = 0; r < h; r++) out.set(part.subarray(r * w, (r + 1) * w), (y + r) * width + x);
      }
    }
    return out;
  }

  async renderChunk(index, seed, x0, y0, width, height) {
    const level = this.manifest.levels[index];
    const { Tensor } = this.ort;
    const { stride, origin } = level;
    const ix0 = Math.floor((x0 - origin) / stride), iy0 = Math.floor((y0 - origin) / stride);
    const left = x0 - (ix0 * stride + origin), top = y0 - (iy0 * stride + origin);
    const nx = cellsFor(level, left + width), ny = cellsFor(level, top + height);

    const feeds = {};
    if (level.kind === "generator") {
      const channels = level.fine_ch + level.coarse_ch;
      const latent = worldLatents(seed, ix0, iy0, nx, ny, level.fine_ch, level.coarse_ch, level.coarse_cells);
      feeds.latent = new Tensor("float32", latent, [1, channels, ny, nx]);
    } else {
      const coarse = await this.render(index - 1, seed, ix0, iy0, nx, ny);
      feeds.coarse = new Tensor("float32", coarse, [1, 1, ny, nx]);
      const latent = worldRandn(seed, 200 + index, level.latent_ch, ix0, iy0, nx, ny);
      feeds.latent = new Tensor("float32", latent, [1, level.latent_ch, ny, nx]);
    }
    let noiseStride = stride;
    noiseSizes(level, nx, ny).forEach(([w, h], layer) => {
      if (layer % 2 === 0) noiseStride /= 2;
      const tag = level.kind === "generator" ? 100 + layer : 1000 * index + layer;
      const scale = stride / noiseStride;
      const noise = worldRandn(seed, tag, 1, ix0 * scale, iy0 * scale, w, h);
      feeds[`noise${layer}`] = new Tensor("float32", noise, [1, 1, h, w]);
    });

    const { height: output } = await this.sessions[index].run(feeds);
    const fullWidth = output.dims[3];
    const data = output.data;
    const out = new Float32Array(width * height);
    for (let r = 0; r < height; r++) {
      const start = (top + r) * fullWidth + left;
      out.set(data.subarray(start, start + width), r * width);
    }
    return out;
  }
}

const RAMP_M = [-11000, -6000, -3000, -200, 0, 1, 100, 400, 1000, 2000, 3500, 5500, 9000];
const RAMP_RGB = [
  [8, 16, 60], [14, 40, 100], [28, 80, 145], [90, 160, 200], [160, 210, 225],
  [70, 130, 75], [100, 150, 85], [160, 175, 105], [190, 165, 110],
  [150, 115, 85], [125, 105, 95], [225, 225, 225], [255, 255, 255],
];

// RGBA for the interior of a (side + 2)^2 field in metres; the outer ring only feeds the slopes.
export function reliefImage(meters, side, kmPerPx) {
  const full = side + 2;
  const exaggeration = 8 * Math.sqrt(Math.max(kmPerPx / 0.926, 1));
  const spacing = kmPerPx * 1000;
  const rgba = new Uint8ClampedArray(side * side * 4);
  for (let y = 0; y < side; y++) {
    for (let x = 0; x < side; x++) {
      const i = (y + 1) * full + x + 1;
      const m = meters[i];
      const dx = ((meters[i + 1] - meters[i - 1]) * exaggeration) / (2 * spacing);
      const dy = ((meters[i + full] - meters[i - full]) * exaggeration) / (2 * spacing);
      const lit = Math.max(0, Math.min(1, (0.5 * dx + 0.5 * dy + 0.7071) / Math.hypot(dx, dy, 1)));
      const shade = m <= 0 ? 0.5 + 0.5 * lit : 0.25 + 1.05 * lit;
      let band = 0;
      while (band < RAMP_M.length - 2 && m > RAMP_M[band + 1]) band++;
      const t = Math.max(0, Math.min(1, (m - RAMP_M[band]) / (RAMP_M[band + 1] - RAMP_M[band])));
      const out = (y * side + x) * 4;
      for (let c = 0; c < 3; c++) {
        rgba[out + c] = (RAMP_RGB[band][c] + t * (RAMP_RGB[band + 1][c] - RAMP_RGB[band][c])) * shade;
      }
      rgba[out + 3] = 255;
    }
  }
  return rgba;
}
