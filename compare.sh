#!/usr/bin/env bash
# This script compares the "fat" and "slim" Docker images built in this lesson.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log() {
  printf '\n== %s ==\n' "$1"
}

run() {
  printf '+ %s\n' "$*"
  "$@"
}

inspect_layers() {
  local image="$1"
  local label="$2"
  docker inspect "$image" | python3 -c "import json,sys; d=json.load(sys.stdin); print('${label} layers:', len(d[0]['RootFS']['Layers']))"
}

inspect_rootfs() {
  local image="$1"
  docker inspect "$image" | python3 -m json.tool | grep -A5 '"RootFS"'
}

TEST_IMAGE="${1:-test.jpg}"

log "Environment setup commands"
printf '%s\n' 'chmod +x install_dev_tools.sh'
printf '%s\n' 'sudo ./install_dev_tools.sh'
printf '%s\n' 'cat install.log'

log "Export TorchScript model"
if [[ "${FORCE_EXPORT:-0}" == "1" || ! -f model.pt ]]; then
  run python3 export_model.py
else
  printf '%s\n' 'model.pt already exists; skipping export. Set FORCE_EXPORT=1 to regenerate it.'
fi

log "Build Docker images"
run docker build -f Dockerfile.fat -t lesson3-fat .
run docker build -f Dockerfile.slim -t lesson3-slim .

log "Compare image sizes"
run bash -lc 'docker images | grep lesson3'

log "Count layers"
inspect_layers lesson3-fat fat
inspect_layers lesson3-slim slim

log "Run inference with Docker"
if [[ -f "$TEST_IMAGE" ]]; then
  run docker run --rm -v "$(pwd)/$TEST_IMAGE:/app/test.jpg" lesson3-fat /app/test.jpg
  run docker run --rm -v "$(pwd)/$TEST_IMAGE:/app/test.jpg" lesson3-slim /app/test.jpg
else
  printf 'Skipping Docker inference because %s was not found.\n' "$TEST_IMAGE"
  printf '%s\n' 'docker run --rm -v "$(pwd)/test.jpg:/app/test.jpg" lesson3-fat  /app/test.jpg'
  printf '%s\n' 'docker run --rm -v "$(pwd)/test.jpg:/app/test.jpg" lesson3-slim /app/test.jpg'
fi

log "Inspect image metadata"
inspect_rootfs lesson3-fat
inspect_rootfs lesson3-slim

log "Local inference commands"
printf '%s\n' 'source .venv/bin/activate'
printf '%s\n' 'python inference.py path/to/image.jpg'
printf '%s\n' 'python inference.py --image path/to/image.jpg'
printf '%s\n' 'python inference.py path/to/folder/'
printf '%s\n' '.venv/bin/python inference.py path/to/image.jpg'
