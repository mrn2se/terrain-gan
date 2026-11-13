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

Or export every model set listed in `models/sets.json` whose files exist, as GitHub Pages does.
The page then shows a Models list with the experiments of [PAPER.md](../PAPER.md); the first set
is the default and the one tested against PyTorch. A checkpoint shared by several sets is
converted once:

    uv run --with onnx --with onnxscript python web/export_models.py --sets models/sets.json

With `--sets` it also copies [PAPER.md](../PAPER.md) and its figures next to the page, for
`paper.html`: the paper, with links under each section that open the model sets it discusses
on the map. In `models/sets.json`, each set's `section` names its paper section, `note` says what
to look for, and `view` gives the seed, position and zoom that a link from the paper opens. The
Models list groups the sets by section and shows the note of the chosen one.

`legal.html` holds the legal notice, disclaimer and privacy information, linked from the map
and the paper. Keep its list of third-party libraries and external hosts in step with the pages.

## Test

Compares the page with PyTorch and checks the map in headless Chrome, served under a
sub-path like GitHub Pages does:

    uv run --no-project --with playwright python web/test/run_test.py

## Publish on GitHub Pages

`.github/workflows/pages.yml` runs when the page or the models change on `master`. It exports the
models, checks the page against PyTorch and publishes the map, the paper, the legal notice and the
exported models at the root of the site, <https://mrn2se.github.io/terrain-gan/>.

## Options

Add to the page URL: `seed`, `lon`, `lat`, `res` (finest-level pixels per screen pixel), `set` (a
model set name from `models/sets.json`), and
`backend=wasm` or `backend=webgpu` to force a backend. With `set` but without `lon` and `lat`, the
page starts at that set's suggested view, and `chunk` (default 512) for the largest
piece of a level rendered at once.

## iPhone and iPad

Safari on an iPhone closed the page after one or two tiles. Desktop tests could not reproduce
this: Chrome's page process peaks near 400 MB on the CPU backend. iPhones and iPads (also when
they claim to be a Mac) therefore render in 256 px chunks instead of 512 px. With that, both the
CPU and the WebGPU backend work on an iPhone, so the likely cause was Safari's lower memory
limit per page. They use WebGPU by default, like other devices.
