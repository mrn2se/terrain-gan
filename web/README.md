# Terrain GAN in the browser

A static page that runs the trained models with no server code: `onnxruntime-web` computes
the terrain and OpenLayers shows it. OpenLayers and the runtime load from the jsDelivr CDN,
so viewers need internet access.

Test it locally (it needs a web server, not `file://`):

    uv run --no-project python -m http.server 8080 --directory web

`./serve_all.sh` in the repository root runs it alongside the map server and TensorBoard.

## Create the models

The page's models and test data are generated from `models/` and are not in git, so export them
once after cloning and again whenever the models change. From the repository root, listing the
world model first and then each refiner:

    uv run --with onnx --with onnxscript python web/export_models.py --ckpt models/world.pt models/regional.pt models/local.pt

## Test

Compares the page with PyTorch and checks the map in headless Chrome, served under a
sub-path like GitHub Pages does:

    uv run --no-project --with playwright python web/test/run_test.py

## Publish on GitHub Pages

`.github/workflows/pages.yml` runs when the page or the models change on `master`. It exports the
models, checks the page against PyTorch and publishes the page, scripts and exported models. They
appear in the `preview/` subdirectory, the site root stays empty and search
engines are asked not to index the page. This does not protect it: even from a private
repository, a GitHub Pages site can be opened by anyone with the link.

## Options

Add to the page URL: `seed`, `lon`, `lat`, `res` (finest-level pixels per screen pixel), and
`backend=wasm` or `backend=webgpu` to force a backend.
