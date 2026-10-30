#!/bin/sh
# Training status view at http://localhost:6006 (extra arguments go to tensorboard).
cd "$(dirname "$0")" && exec uv run tensorboard --logdir runs --port 6006 "$@"
