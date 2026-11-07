const TILE = 256;

const params = new URLSearchParams(location.search);
const seedInput = document.getElementById("seed");
const status = document.getElementById("status");
seedInput.value = params.get("seed") || "0";

const workerParams = new URLSearchParams({ backend: params.get("backend") || "", chunk: params.get("chunk") || "" });
const worker = new Worker(new URL(`worker.js?${workerParams}`, import.meta.url), { type: "module" });
const pending = new Map();
let nextId = 0;
let info = null;
let rendered = 0;
let totalMs = 0;

function showStatus() {
  const backend = info.provider === "webgpu" ? "WebGPU" : `CPU (${info.threads} thread${info.threads > 1 ? "s" : ""})`;
  const speed = rendered ? `, ${Math.round(totalMs / rendered)} ms per tile` : "";
  status.textContent = `${backend}${speed}, ${pending.size} waiting`;
  // Read by web/test/run_test.py.
  window.terrainStats = { provider: info.provider, gpuName: info.gpuName, rendered, totalMs, waiting: pending.size };
}

worker.onerror = (event) => fail(event.message || "The worker could not start.");
worker.onmessage = ({ data }) => {
  if (data.type === "ready") {
    info = data;
    startMap();
    showStatus();
  } else if (data.type === "failed") {
    fail(data.error);
  } else if (data.type === "tile") {
    const tile = pending.get(data.id);
    pending.delete(data.id);
    if (data.error) {
      status.classList.add("error");
      status.textContent = data.error;
      window.terrainError = data.error;
    }
    if (data.blob) {
      rendered += 1;
      totalMs += data.ms;
      const url = URL.createObjectURL(data.blob);
      const image = tile.getImage();
      image.addEventListener("load", () => URL.revokeObjectURL(url), { once: true });
      image.src = url;
    } else {
      // state ERROR; a cancelled tile is requested again if still in view
      tile.setState(3);
    }
    if (!data.error) showStatus();
  }
};

function fail(message) {
  status.classList.add("error");
  status.textContent = `Could not start: ${message}`;
  window.terrainError = message;
}

function startMap() {
  const { resolutions, pxPerDegree } = info;
  const projection = new ol.proj.Projection({ code: "WORLD", units: "pixels" });
  const source = new ol.source.TileImage({
    projection,
    wrapX: false,
    transition: 0,
    tileGrid: new ol.tilegrid.TileGrid({ origin: [0, 0], resolutions, tileSize: TILE }),
    tileUrlFunction: ([z, x, y]) => `${seedInput.value}/${z}/${x}/${y}`,
    tileLoadFunction: (tile, key) => {
      const [seed, z, x, y] = key.split("/").map(Number);
      const id = nextId++;
      pending.set(id, tile);
      worker.postMessage({ type: "tile", id, seed, z, x, y });
    },
  });
  const view = new ol.View({
    projection,
    center: [Number(params.get("lon") || 0) * pxPerDegree, Number(params.get("lat") || 0) * pxPerDegree],
    resolution: Number(params.get("res")) || resolutions[0] / 2,
    minResolution: 0.25,
    maxResolution: resolutions[0],
  });
  const map = new ol.Map({ target: "map", layers: [new ol.layer.Tile({ source, preload: 0 })], view });
  window.terrainMap = map;

  const updateUrl = () => {
    const [x, y] = view.getCenter();
    const query = new URLSearchParams({
      seed: seedInput.value,
      lon: (x / pxPerDegree).toFixed(3),
      lat: (y / pxPerDegree).toFixed(3),
      res: view.getResolution().toFixed(3),
    });
    history.replaceState(null, "", `?${query}`);
  };
  seedInput.addEventListener("change", () => {
    worker.postMessage({ type: "cancel" });
    source.refresh();
    updateUrl();
  });
  map.on("moveend", updateUrl);
  map.on("pointermove", ({ coordinate: [x, y] }) => {
    document.getElementById("pos").textContent = `lon ${(x / pxPerDegree).toFixed(3)}, lat ${(y / pxPerDegree).toFixed(3)}`;
  });
}
