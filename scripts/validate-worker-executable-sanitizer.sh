#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

task_dir=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-worker-executable-asan.XXXXXX")
cleanup() { rm -rf "$task_dir"; }
trap cleanup EXIT HUP INT TERM

cc_bin=${CC:-/usr/bin/cc}
leak_detection=0
if [ "$(uname -s)" = Linux ]; then leak_detection=1; fi
"$cc_bin" -std=c11 -Wall -Wextra -Werror -g \
  internal/process/approved_spawn_probe_child.c \
  -o "$task_dir/approved-child"

"$cc_bin" -std=c11 -Wall -Wextra -Werror -g -pthread \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"${MOON_HOME:-$HOME/.moon}/include" \
  deploy/worker_executable_file/approved_executable_asan_probe.c \
  internal/process/process_approved_spawn.c \
  -o "$task_dir/ownership-probe"

ASAN_OPTIONS=detect_leaks=$leak_detection:fast_unwind_on_malloc=0 \
  sh scripts/run-sanitized.sh "$task_dir/ownership-probe" \
  "$task_dir/approved-child"

"$cc_bin" -std=c11 -Wall -Wextra -Werror -g -pthread \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  internal/process/approved_spawn_asan_probe.c \
  internal/process/process_approved_spawn.c \
  -o "$task_dir/spawn-probe"

ASAN_OPTIONS=detect_leaks=$leak_detection:fast_unwind_on_malloc=0 \
  sh scripts/run-sanitized.sh "$task_dir/spawn-probe" "$task_dir/approved-child"

printf '%s\n' 'LunaFlux pinned worker executable sanitizer gate passed.'
