// Renders tiles away from the page's main thread, newest request first, so that
// tiles for the current view come before ones already scrolled past.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";
import { DEFAULT_CHUNK, loadTerrain, ORT_BASE, reliefImage, toMeters } from "./terrain.js";

const TILE = 256;
const LEVEL_SCALE = 4; // resolution ratio between model levels

ort.env.wasm.wasmPaths = ORT_BASE;
// Threads need cross-origin isolation headers, which GitHub Pages does not send.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 1, 4) : 1;

let terrain = null;
let resolutions = null;
const queue = [];
let busy = false;
let gpuName = "";

async function renderTile({ seed, z, x, y }) {
  const levels = terrain.manifest.levels.length;
  const resolution = resolutions[z];
  // The coarsest level that still has enough detail for this zoom.
  let level = levels - 1, scale = 1;
  while (level > 0 && scale * LEVEL_SCALE <= resolution) {
    level -= 1;
    scale *= LEVEL_SCALE;
  }
  const factor = resolution / scale;
  // One extra pixel per side so that hill shading is continuous across tiles.
  const full = TILE + 2;
  const side = full * factor;
  const norm = await terrain.render(level, seed, (x * TILE - 1) * factor, (y * TILE - 1) * factor, side, side);
  const meters = new Float32Array(full * full);
  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      meters[Math.floor(r / factor) * full + Math.floor(c / factor)] += toMeters(norm[r * side + c]) / (factor * factor);
    }
  }
  const rgba = reliefImage(meters, TILE, (111.32 / terrain.manifest.px_per_degree) * resolution);
  const canvas = new OffscreenCanvas(TILE, TILE);
  canvas.getContext("2d").putImageData(new ImageData(rgba, TILE, TILE), 0, 0);
  return canvas.convertToBlob({ type: "image/png" });
}

async function pump() {
  if (busy) return;
  busy = true;
  while (queue.length) {
    const job = queue.pop();
    const start = performance.now();
    try {
      const blob = await renderTile(job);
      self.postMessage({ type: "tile", id: job.id, blob, ms: performance.now() - start });
    } catch (error) {
      self.postMessage({ type: "tile", id: job.id, error: String(error) });
    }
  }
  busy = false;
}

self.onmessage = async ({ data }) => {
  if (data.type === "tile") {
    queue.push(data);
    pump();
  } else if (data.type === "cancel") {
    for (const job of queue.splice(0)) self.postMessage({ type: "tile", id: job.id, cancelled: true });
  }
};

try {
  // ?backend=webgpu or ?backend=wasm on the page forces one. By default a real GPU is
  // used if there is one; software WebGPU is far slower than the CPU backend.
  const query = new URL(self.location.href).searchParams;
  const forced = query.get("backend");
  // Safari on iPhones and iPads (which may claim to be a Mac) closed the page after a
  // tile or two with 512 px pieces. With 256 px pieces both backends work there, so
  // the cause was most likely its lower memory limit per page.
  const ios = /iPhone|iPad|iPod/.test(navigator.userAgent) || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);
  const chunk = Number(query.get("chunk")) || (ios ? 256 : DEFAULT_CHUNK);
  const adapter = "gpu" in navigator ? await navigator.gpu.requestAdapter().catch(() => null) : null;
  gpuName = adapter?.info ? `${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.description}`.trim() : "";
  const useGpu = adapter && !adapter.info?.isFallbackAdapter;
  terrain = await loadTerrain(ort, forced ? [forced] : useGpu ? ["webgpu", "wasm"] : ["wasm"], chunk);
  const levels = terrain.manifest.levels.length;
  // Finest-level pixels per screen pixel, one entry per zoom level. The furthest zoom is
  // 2x the coarsest model, because further out costs too much to compute on a CPU.
  resolutions = Array.from({ length: 2 * levels }, (_, i) => 2 ** (2 * levels - 1 - i));
  self.postMessage({
    type: "ready",
    provider: terrain.provider,
    resolutions,
    pxPerDegree: terrain.manifest.px_per_degree,
    threads: ort.env.wasm.numThreads,
    gpuName,
  });
} catch (error) {
  self.postMessage({ type: "failed", error: String(error) });
}
