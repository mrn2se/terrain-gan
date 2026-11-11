const TILE = 256;

const params = new URLSearchParams(location.search);
const seedInput = document.getElementById("seed");
const status = document.getElementById("status");
seedInput.value = params.get("seed") || "0";

const setSelect = document.getElementById("set");
const setName = params.get("set") || "";

// The list of model sets exists only when the models were exported with --sets. Changing the
// set reloads the page, as sets can have different numbers of levels and so zoom steps.
const setsLoaded = fetch(new URL("../models/sets.json", import.meta.url))
  .then((response) => (response.ok ? response.json() : []))
  .catch(() => [])
  .then((sets) => {
    const current = sets.find(({ name }) => name === setName) || sets[0];
    if (sets.length < 2) return current;
    // One group per paper section, in the order of sets.json.
    const groups = new Map();
    for (const set of sets) {
      const key = set.section ? `§${set.section} ${set.title || ""}`.trim() : "Models";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(set);
    }
    for (const [key, members] of groups) {
      const group = document.createElement("optgroup");
      group.label = key;
      for (const { name, label } of members) group.append(new Option(label, name, false, name === current.name));
      setSelect.append(group);
    }
    setSelect.parentElement.hidden = false;
    showNote(current);
    setSelect.addEventListener("change", () => {
      const query = new URLSearchParams(location.search);
      query.set("set", setSelect.value);
      location.search = `?${query}`;
    });
    return current;
  });

function showNote(set) {
  const note = document.getElementById("note");
  if (!set || !set.note) return;
  const text = document.getElementById("note-text");
  text.textContent = `${set.note} `;
  if (set.section) {
    const link = document.createElement("a");
    link.href = `paper.html#s${set.section.replace(".", "-")}`;
    link.textContent = `Paper, section ${set.section} →`;
    text.append(link);
  }
  // Closed on phones, where it would cover much of the map.
  note.open = !matchMedia("(max-width: 600px)").matches;
  note.hidden = false;
}

const workerParams = new URLSearchParams({ backend: params.get("backend") || "", chunk: params.get("chunk") || "", set: setName });
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
worker.onmessage = async ({ data }) => {
  if (data.type === "ready") {
    info = data;
    startMap(await setsLoaded);
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

function startMap(set) {
  const { resolutions, pxPerDegree } = info;
  // A set opened without a position, as from the paper, starts at its suggested view.
  const view0 = (setName && set && set.view && !params.has("lon") && !params.has("lat")) ? set.view : {};
  if (view0.seed !== undefined && !params.has("seed")) seedInput.value = String(view0.seed);
  const start = (key) => params.get(key) ?? (view0[key] !== undefined ? String(view0[key]) : null);
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
    center: [Number(start("lon") || 0) * pxPerDegree, Number(start("lat") || 0) * pxPerDegree],
    resolution: Number(start("res")) || resolutions[0] / 2,
    minResolution: 0.25,
    maxResolution: resolutions[0],
  });
  const map = new ol.Map({ target: "map", layers: [new ol.layer.Tile({ source, preload: 0 })], view });
  window.terrainMap = map;

  const updateUrl = () => {
    const [x, y] = view.getCenter();
    const query = new URLSearchParams({
      ...(setName ? { set: setName } : {}),
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
