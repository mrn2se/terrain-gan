#!/bin/sh
# Runs the map server, the static browser page and TensorBoard together; Ctrl+C stops them all.
# Add --device cpu while training on a small GPU. HOST=127.0.0.1 keeps them on this computer.
cd "$(dirname "$0")" || exit 1
HOST="${HOST:-0.0.0.0}"
trap 'kill 0' INT TERM EXIT
uv run serve.py --host "$HOST" --port 8000 "$@" &
uv run --no-project python -m http.server 8080 --bind "$HOST" --directory web &
uv run tensorboard --logdir runs --host "$HOST" --port 6006 &
echo "Map http://$HOST:8000/  Browser version http://$HOST:8080/  TensorBoard http://$HOST:6006/"
wait
