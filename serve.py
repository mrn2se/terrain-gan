"""Browse the endless world in OpenLayers.

Usage: uv run serve.py [--ckpt runs/world/latest.pt runs/regional/latest.pt] [--port 8000] [--device cpu]
Checkpoints are reloaded when they change, so this can run alongside training.
"""

import argparse
import io
import os
import re
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

from generate import load_cascade
from terrain_gan import norm_to_meters, relief_image

TILE = 256
LEVEL_SCALE = 4  # resolution ratio between cascade levels

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
</style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <label>World seed <input id="seed" type="number" min="0" step="1"></label>
  <div id="pos">&nbsp;</div>
</div>
<script>
const PX_PER_DEGREE = %(ppd)f;
const RESOLUTIONS = %(resolutions)s;
const params = new URLSearchParams(location.search);
const seedInput = document.getElementById('seed');
seedInput.value = params.get('seed') || '0';

const projection = new ol.proj.Projection({code: 'WORLD', units: 'pixels'});
const source = new ol.source.TileImage({
  projection,
  wrapX: false,
  tileGrid: new ol.tilegrid.TileGrid({origin: [0, 0], resolutions: RESOLUTIONS, tileSize: %(tile)d}),
  tileUrlFunction: ([z, x, y]) => `/tile/${seedInput.value}/${z}/${x}/${y}.png`,
});
const view = new ol.View({
  projection,
  center: [Number(params.get('lon') || 0) * PX_PER_DEGREE, Number(params.get('lat') || 0) * PX_PER_DEGREE],
  resolution: RESOLUTIONS[Math.floor(RESOLUTIONS.length / 2)],
  minResolution: 0.25,
  maxResolution: RESOLUTIONS[0] * 2,
});
const map = new ol.Map({target: 'map', layers: [new ol.layer.Tile({source})], view});

function updateUrl() {
  const [x, y] = view.getCenter();
  const q = new URLSearchParams({seed: seedInput.value, lon: (x / PX_PER_DEGREE).toFixed(3), lat: (y / PX_PER_DEGREE).toFixed(3)});
  history.replaceState(null, '', '?' + q);
}
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


class TileRenderer:
    def __init__(self, ckpts, device):
        self.ckpts, self.device = ckpts, device
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
        with self.lock:
            norm = self.cascade.render(level, seed, x0, y0, side, side, self.device)
        meters = norm_to_meters(norm)
        if factor > 1:
            meters = meters.reshape(TILE + 2, factor, TILE + 2, factor).mean((1, 3))
        rgb = relief_image(meters, 111.32 / self.ppd * resolution)[1:-1, 1:-1]
        out = io.BytesIO()
        Image.fromarray(rgb).save(out, "PNG", compress_level=1)
        return out.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", default=["runs/world/latest.pt"], help="world generator, then any refiners")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    renderer = TileRenderer(args.ckpt, args.device)
    tile_path = re.compile(r"^/tile/(\d+)/(\d+)/(-?\d+)/(-?\d+)\.png$")

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                page = PAGE % dict(ppd=renderer.ppd, resolutions=renderer.resolutions, tile=TILE)
                self.respond(page.encode(), "text/html; charset=utf-8")
                return
            match = tile_path.match(path)
            if not match or int(match[2]) >= len(renderer.resolutions):
                self.send_error(404)
                return
            with renderer.lock:
                version = renderer.reload()
            seed, z, x, y = (int(v) for v in match.groups())
            self.respond(renderer.tile(seed, z, x, y, version), "image/png")

        def log_message(self, *_):
            pass

    print(f"http://{args.host}:{args.port}/", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
