#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

task_dir=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-process-maintenance.XXXXXX")
cleanup() { rm -rf "$task_dir"; }
trap cleanup EXIT HUP INT TERM

cc_bin=/usr/bin/clang
if [ ! -x "$cc_bin" ]; then
  printf '%s\n' 'system clang is required for the process maintenance sanitizer gate' >&2
  exit 1
fi

"$cc_bin" -std=c11 -Wall -Wextra -Werror -g \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"${MOON_HOME:-$HOME/.moon}/include" \
  internal/process/process_maintenance_asan_probe.c \
  -o "$task_dir/probe"

ASAN_OPTIONS=detect_leaks=0:fast_unwind_on_malloc=0 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  sh scripts/run-sanitized.sh "$task_dir/probe"

printf '%s\n' \
  'LunaFlux cooperative process maintenance ASan+UBSan gate passed.'
