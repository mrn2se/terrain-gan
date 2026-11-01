"""Test the static page in headless Chrome, served the way GitHub Pages would serve it:
plain files, under a sub-path, with no special headers.

Usage, from the repository root:
    uv run --no-project --with playwright python web/test/run_test.py [--shot out.png] [--viewport 390x844]
"""

import argparse
import functools
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

WEB = Path(__file__).resolve().parent.parent
SUBPATH = "terrain-gan"


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def wait_tiles(page, timeout_s):
    """Wait until the map has shown tiles and nothing is left to render."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        stats = page.evaluate("window.terrainStats || null")
        error = page.evaluate("window.terrainError || null")
        if error:
            return dict(error=error)
        if stats and stats["rendered"] > 0 and stats["waiting"] == 0:
            page.wait_for_timeout(1500)
            again = page.evaluate("window.terrainStats")
            if again["waiting"] == 0 and again["rendered"] == stats["rendered"]:
                return again
        page.wait_for_timeout(500)
    return dict(timeout=True, stats=page.evaluate("window.terrainStats || null"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shot", help="save a screenshot of the map here")
    p.add_argument("--viewport", default="1200x800")
    p.add_argument("--webgpu", action="store_true", help="also run the tests on WebGPU")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--backend", default="", help="force the page's backend: wasm or webgpu")
    p.add_argument("--res", default="", help="starting zoom: finest-level pixels per screen pixel")
    p.add_argument("--page-only", action="store_true", help="skip the comparison with PyTorch")
    args = p.parse_args()

    # A sub-path like username.github.io/<repo>/ catches any absolute URL in the page.
    site = Path(tempfile.mkdtemp())
    os.symlink(WEB, site / SUBPATH)
    handler = functools.partial(Quiet, directory=str(site))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/{SUBPATH}"

    failed = False
    width, height = (int(v) for v in args.viewport.split("x"))
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            channel="chrome", headless=True, args=["--enable-unsafe-webgpu", "--enable-features=Vulkan"]
        )
        problems = []

        def watch(page):
            page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
            page.on("console", lambda m: problems.append(f"console: {m.text}") if m.type == "error" else None)
            page.on("requestfailed", lambda r: problems.append(f"request failed: {r.url}"))
            page.on("response", lambda r: problems.append(f"HTTP {r.status}: {r.url}") if r.status >= 400 else None)

        for provider in [] if args.page_only else ["wasm"] + (["webgpu"] if args.webgpu else []):
            page = browser.new_page()
            watch(page)
            page.goto(f"{base}/test/test.html?provider={provider}")
            page.wait_for_function("window.testResults !== undefined", timeout=args.timeout * 1000)
            report = page.evaluate("window.testResults")
            print(f"--- tests on {provider}")
            if report.get("error"):
                print("ERROR", report["error"])
                failed = True
            for r in report["results"]:
                print(f"{'ok  ' if r['ok'] else 'FAIL'} {r['name']}: largest difference {r['difference']:.2e}")
                failed |= not r["ok"]
            page.close()

        page = browser.new_page(viewport=dict(width=width, height=height))
        watch(page)
        page.goto(f"{base}/index.html?seed=3&backend={args.backend}&res={args.res}")
        page.wait_for_function("window.terrainStats !== undefined || window.terrainError !== undefined",
                               timeout=args.timeout * 1000)
        result = wait_tiles(page, args.timeout)
        print("--- page:", json.dumps(result))
        print("status text:", page.inner_text("#status"))
        failed |= "error" in result or "timeout" in result
        if args.shot:
            page.screenshot(path=args.shot)
        overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        if overflow:
            print("FAIL page scrolls sideways")
            failed = True
        browser.close()

    for problem in sorted(set(problems)):
        print("PROBLEM", problem)
    failed |= bool(problems)
    shutil.rmtree(site)
    print("FAILED" if failed else "ALL PASSED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
