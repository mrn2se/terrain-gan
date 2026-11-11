"""Browse the endless world in OpenLayers.

Usage: uv run serve.py [--sets models/sets.json | --ckpt WORLD [REFINER ...]] [--port 8000] [--device cpu]
The page offers each model set of --sets whose files exist, in a drop-down list; --ckpt serves
just one set instead. Checkpoints are reloaded when they change, so this can run alongside
training. World seed -1 shows the real elevation data instead of the GAN.
"""

import argparse
import html
import io
import json
import os
import re
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import numpy as np
import torch
from PIL import Image

from generate import load_cascade
from terrain_gan import norm_to_meters, relief_image
from train import cache_downsampled, cache_raster

TILE = 256
LEVEL_SCALE = 4  # resolution ratio between cascade levels
GROUND_TRUTH_SEED = -1

PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Terrain GAN</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ol@10.6.1/ol.css">
<script src="https://cdn.jsdelivr.net/npm/ol@10.6.1/dist/ol.js"></script>
<style>
  html, body, #map { margin: 0; height: 100%%; }
  #panel { position: absolute; top: 8px; right: 8px; z-index: 10; padding: 8px 10px;
           background: #fffd; font: 14px sans-serif; border-radius: 4px; }
  #panel input { width: 6em; }
  #panel select { max-width: 30em; }
  #panel label { display: block; margin-bottom: 4px; }
</style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <label %(set_style)s>Models <select id="set">%(options)s</select></label>
  <label>World seed (-1 = real Earth) <input id="seed" type="number" min="-1" step="1"></label>
  <div id="pos">&nbsp;</div>
</div>
<script>
const PX_PER_DEGREE = %(ppd)f;
const RESOLUTIONS = %(resolutions)s;
const SET = %(set)s;
const params = new URLSearchParams(location.search);
const seedInput = document.getElementById('seed');
const setSelect = document.getElementById('set');
seedInput.value = params.get('seed') || '0';

const projection = new ol.proj.Projection({code: 'WORLD', units: 'pixels'});
const source = new ol.source.TileImage({
  projection,
  wrapX: false,
  tileGrid: new ol.tilegrid.TileGrid({origin: [0, 0], resolutions: RESOLUTIONS, tileSize: %(tile)d}),
  tileUrlFunction: ([z, x, y]) => `/tile/${SET}/${seedInput.value}/${z}/${x}/${y}.png`,
});
const view = new ol.View({
  projection,
  center: [Number(params.get('lon') || 0) * PX_PER_DEGREE, Number(params.get('lat') || 0) * PX_PER_DEGREE],
  resolution: RESOLUTIONS[Math.floor(RESOLUTIONS.length / 2)],
  minResolution: 0.25,
  maxResolution: RESOLUTIONS[0] * 2,
});
const map = new ol.Map({target: 'map', layers: [new ol.layer.Tile({source})], view});

function query() {
  const [x, y] = view.getCenter();
  return new URLSearchParams({set: SET, seed: seedInput.value, lon: (x / PX_PER_DEGREE).toFixed(3), lat: (y / PX_PER_DEGREE).toFixed(3)});
}
function updateUrl() {
  history.replaceState(null, '', '?' + query());
}
// Model sets can have different numbers of levels and so different zoom steps: reload the page.
setSelect.addEventListener('change', () => {
  const q = query();
  q.set('set', setSelect.value);
  location.search = '?' + q;
});
seedInput.addEventListener('change', () => { source.refresh(); updateUrl(); });
map.on('moveend', updateUrl);
map.on('pointermove', (e) => {
  const [x, y] = e.coordinate;
  document.getElementById('pos').textContent = `lon ${(x / PX_PER_DEGREE).toFixed(3)}, lat ${(y / PX_PER_DEGREE).toFixed(3)}`;
});
</script>
</body>
</html>
"""


class GroundTruth:
    """Real elevation at the same pixel grids as the cascade levels."""

    def __init__(self, tif, cache_dir):
        self.tif, self.cache_dir = tif, cache_dir
        self.rasters = {}

    def raster(self, px_per_degree):
        if px_per_degree not in self.rasters:
            npy, source_ppd = cache_raster(self.tif, self.cache_dir)
            factor = round(source_ppd / px_per_degree)
            if factor > 1:
                npy = cache_downsampled(npy, factor)
            self.rasters[px_per_degree] = np.load(npy, mmap_mode="r")
        return self.rasters[px_per_degree]

    def read(self, px_per_degree, x0, y0, width, height):
        """Metres for world pixels, where world x = lon * ppd and y = -lat * ppd. NaN past the poles."""
        data = self.raster(px_per_degree)
        rows, cols = data.shape
        row = np.arange(y0, y0 + height) + rows // 2
        col = (np.arange(x0, x0 + width) + cols // 2) % cols
        out = np.full((height, width), np.nan, dtype=np.float32)
        inside = (row >= 0) & (row < rows)
        out[inside] = data[np.ix_(row[inside], col)]
        return out


class TileRenderer:
    def __init__(self, ckpts, device, ground_truth):
        self.ckpts, self.device, self.ground_truth = ckpts, device, ground_truth
        self.lock = threading.Lock()
        self.mtime = None
        self.reload()
        self.tile = lru_cache(maxsize=2048)(self._tile)

    def reload(self):
        mtime = tuple(os.path.getmtime(path) for path in self.ckpts)
        if mtime != self.mtime:
            self.cascade = load_cascade(self.ckpts, self.device)
            self.ppd = self.cascade.px_per_degree
            levels = len(self.ckpts)
            # Finest-level pixels per screen pixel, one entry per zoom level.
            self.resolutions = [2**i for i in range(3 + 2 * (levels - 1), -1, -1)]
            self.mtime = mtime
            print(f"loaded {self.ckpts}", flush=True)
        return mtime

    def _tile(self, seed, z, x, y, version):
        resolution = self.resolutions[z]
        # Use the coarsest level that still has enough detail for this zoom.
        level, scale = len(self.ckpts) - 1, 1
        while level > 0 and scale * LEVEL_SCALE <= resolution:
            level, scale = level - 1, scale * LEVEL_SCALE
        factor = resolution // scale
        # One extra output pixel per side so hill shading is continuous across tiles.
        x0, y0 = (x * TILE - 1) * factor, (y * TILE - 1) * factor
        side = (TILE + 2) * factor
        if seed == GROUND_TRUTH_SEED:
            meters = self.ground_truth.read(self.ppd / scale, x0, y0, side, side)
        else:
            with self.lock:
                norm = self.cascade.render(level, seed, x0, y0, side, side, self.device)
            meters = norm_to_meters(norm)
        if factor > 1:
            meters = meters.reshape(TILE + 2, factor, TILE + 2, factor).mean((1, 3))
        valid = np.isfinite(meters)
        rgb = relief_image(np.where(valid, meters, 0.0), 111.32 / self.ppd * resolution)[1:-1, 1:-1]
        if not valid.all():  # beyond the poles of the real data
            rgb = np.dstack([rgb, 255 * valid[1:-1, 1:-1].astype(np.uint8)])
        out = io.BytesIO()
        Image.fromarray(rgb).save(out, "PNG", compress_level=1)
        return out.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", default="models/sets.json", help="model sets for the drop-down list")
    p.add_argument("--ckpt", nargs="+", help="serve only this set: world generator, then any refiners")
    p.add_argument("--data", default="data/ETOPO_2022_v1_30s_N90W180_bed.tif", help="source of the seed -1 ground truth")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.ckpt:
        sets = [{"name": "custom", "label": " ".join(args.ckpt), "ckpt": args.ckpt}]
    else:
        with open(args.sets) as f:
            sets = json.load(f)
        missing = [s["name"] for s in sets if not all(os.path.exists(path) for path in s["ckpt"])]
        if missing:
            print(f"skipped model sets with missing files: {', '.join(missing)}", flush=True)
        sets = [s for s in sets if s["name"] not in missing]
    ground_truth = GroundTruth(args.data, args.cache_dir)
    renderers = {}  # loaded on first use
    renderers_lock = threading.Lock()

    def renderer_for(name):
        with renderers_lock:
            if name not in renderers:
                ckpt = next(s["ckpt"] for s in sets if s["name"] == name)
                renderers[name] = TileRenderer(ckpt, args.device, ground_truth)
            return renderers[name]

    names = {s["name"] for s in sets}
    tile_path = re.compile(r"^/tile/([\w-]+)/(-?\d+)/(\d+)/(-?\d+)/(-?\d+)\.png$")

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/":
                name = parse_qs(url.query).get("set", [sets[0]["name"]])[0]
                name = name if name in names else sets[0]["name"]
                renderer = renderer_for(name)
                options = "".join(
                    f'<option value="{s["name"]}"{" selected" if s["name"] == name else ""} title="{html.escape(s.get("note", ""))}">'
                    f'{html.escape(("§" + s["section"] + " ") if s.get("section") else "")}{html.escape(s["label"])}</option>'
                    for s in sets
                )
                page = PAGE % dict(
                    ppd=renderer.ppd, resolutions=renderer.resolutions, tile=TILE, set=json.dumps(name),
                    options=options, set_style='style="display:none"' if len(sets) == 1 else "",
                )
                self.respond(page.encode(), "text/html; charset=utf-8")
                return
            match = tile_path.match(url.path)
            if not match or match[1] not in names:
                self.send_error(404)
                return
            renderer = renderer_for(match[1])
            if int(match[3]) >= len(renderer.resolutions):
                self.send_error(404)
                return
            with renderer.lock:
                version = renderer.reload()
            seed, z, x, y = (int(v) for v in match.groups()[1:])
            if seed < GROUND_TRUTH_SEED:
                self.send_error(404)
                return
            self.respond(renderer.tile(seed, z, x, y, version), "image/png")

        def log_message(self, *_):
            pass

    print(f"http://{args.host}:{args.port}/", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
