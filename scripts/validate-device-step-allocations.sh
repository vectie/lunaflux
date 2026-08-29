#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon run --target native --release tests/device_step_alloc

stage_source="engine/device_step/stage.mbt"
frame_source="engine/device_step/preflight_frame_capability.mbt"
if rg -Fq 'invalidate_counts' "$stage_source"; then
  printf '%s\n' 'device-step restored the redundant zero-count publication' >&2
  exit 1
fi
if [ "$(rg -F -c 'preflight_and_write(' "$stage_source")" -ne 1 ] ||
  [ "$(rg -F -c 'preflight_frame_and_write(' "$frame_source")" -ne 1 ]; then
  printf '%s\n' 'device-step staging no longer has one descriptor validation scan' >&2
  exit 1
fi
if rg -n 'preflight_(frame_and_write|and_write)' \
  engine/device_step/paged_executor_completion.mbt \
  engine/device_step/paged_executor_wire_completion.mbt; then
  printf '%s\n' 'device-step completion restored a redundant descriptor rescan' >&2
  exit 1
fi
