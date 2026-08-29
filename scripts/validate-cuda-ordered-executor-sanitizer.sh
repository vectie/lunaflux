#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

task_dir=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-ordered-executor-asan.XXXXXX")
cleanup() { rm -rf "$task_dir"; }
trap cleanup EXIT HUP INT TERM

cc_bin=/usr/bin/clang
if [ ! -x "$cc_bin" ]; then
  echo 'system clang is required for the ordered-executor sanitizer gate' >&2
  exit 1
fi

moon_bin=$(command -v moon)
moon_root=$(CDPATH= cd -- "$(dirname -- "$moon_bin")/.." && pwd)

"$cc_bin" -std=c11 -Wall -Wextra -Werror -g -pthread \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"$moon_root/include" \
  internal/cuda/ordered_executor_sanitizer_main.c \
  internal/cuda/ordered_executor.c \
  internal/cuda/ordered_executor_probe.c \
  internal/cuda/ordered_graph.c \
  internal/cuda/ordered_graph_probe.c \
  internal/cuda/resources.c \
  internal/cuda/regions.c \
  internal/cuda/modules.c \
  -o "$task_dir/probe"

leak_detection=1
if [ "$(uname -s)" = Darwin ]; then
  leak_detection=0
fi
ASAN_OPTIONS="detect_leaks=$leak_detection:fast_unwind_on_malloc=0" \
  sh scripts/run-sanitized.sh "$task_dir/probe"

echo 'LunaFlux ordered-executor ASan/UBSan lease/race/retry gate passed.'
