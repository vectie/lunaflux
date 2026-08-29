#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

task_dir=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-nccl-asan.XXXXXX")
cleanup() { rm -rf "$task_dir"; }
trap cleanup EXIT HUP INT TERM

cc_bin=/usr/bin/clang
if [ ! -x "$cc_bin" ]; then
  printf '%s\n' 'system clang is required for the NCCL sanitizer gate' >&2
  exit 1
fi

"$cc_bin" -std=c11 -Wall -Wextra -Werror -g \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"${MOON_HOME:-$HOME/.moon}/include" \
  internal/nccl/sanitizer_probe.c \
  internal/cuda/collective_interop.c \
  internal/cuda/collective_interop_probe_runtime.c \
  internal/cuda/regions.c \
  internal/cuda/resources.c \
  -o "$task_dir/probe"

ASAN_OPTIONS=detect_leaks=0:fast_unwind_on_malloc=0 \
  sh scripts/run-sanitized.sh "$task_dir/probe"

"$cc_bin" -std=c11 -Wall -Wextra -Werror -g \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"${MOON_HOME:-$HOME/.moon}/include" \
  internal/nccl/nonblocking_sanitizer_probe.c \
  internal/cuda/collective_interop.c \
  internal/cuda/collective_interop_probe_runtime.c \
  internal/cuda/resources.c \
  internal/cuda/regions.c \
  -o "$task_dir/nonblocking-probe"

ASAN_OPTIONS=detect_leaks=0:fast_unwind_on_malloc=0 \
  sh scripts/run-sanitized.sh "$task_dir/nonblocking-probe"

printf '%s\n' \
  'LunaFlux NCCL exact-TU ASan/UBSan lifecycle/nonblocking gate passed.'
