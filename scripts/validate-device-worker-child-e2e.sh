#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

moon build cmd/device_worker_child tests/device_worker_child_e2e \
  --target native --release --deny-warn --warn-list +73

worker="$repo_root/_build/native/release/build/cmd/device_worker_child/device_worker_child.exe"
gate="$repo_root/_build/native/release/build/tests/device_worker_child_e2e/device_worker_child_e2e.exe"

if [ ! -x "$worker" ] || [ ! -x "$gate" ]; then
  printf '%s\n' 'device-worker child executables are missing' >&2
  exit 1
fi

"$gate" "$worker"
printf '%s\n' 'LunaFlux device-worker child no-Ready gate passed.'
