#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

task_dir=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-process-child-asan.XXXXXX")
cleanup() { rm -rf "$task_dir"; }
trap cleanup EXIT HUP INT TERM

cc_bin=${CC:-/usr/bin/cc}
"$cc_bin" -std=c11 -Wall -Wextra -Werror -g \
  -fsanitize=address -fno-omit-frame-pointer \
  -I"${MOON_HOME:-$HOME/.moon}/include" \
  internal/process/child_control_asan_probe.c \
  internal/process/child_control.c \
  -o "$task_dir/probe"

ASAN_OPTIONS=detect_leaks=0:fast_unwind_on_malloc=0 "$task_dir/probe"

"$cc_bin" -std=c11 -Wall -Wextra -Werror -g \
  -fsanitize=address -fno-omit-frame-pointer \
  -I"${MOON_HOME:-$HOME/.moon}/include" \
  internal/process/process_io_asan_probe.c \
  internal/process/process_io.c \
  internal/process/process_nonblocking.c \
  -o "$task_dir/io-probe"

ASAN_OPTIONS=detect_leaks=0:fast_unwind_on_malloc=0 "$task_dir/io-probe"

"$cc_bin" -std=c11 -Wall -Wextra -Werror -g \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"${MOON_HOME:-$HOME/.moon}/include" \
  internal/process/process_spawn_asan_probe.c \
  internal/process/process_io.c \
  -o "$task_dir/spawn-probe"

ASAN_OPTIONS=detect_leaks=0:fast_unwind_on_malloc=0 "$task_dir/spawn-probe"

printf '%s\n' 'LunaFlux process child-control and nonblocking-I/O AddressSanitizer gate passed.'
