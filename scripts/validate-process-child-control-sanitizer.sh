#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

if ! rg -q 'pub fn ChildProcess::begin_write_frame_until' \
    internal/process/pending_frame.mbt ||
  ! rg -q 'pub fn ChildProcess::begin_read_frame_until' \
    internal/process/pending_frame.mbt ||
  ! rg -q 'write and ordered read share one caller absolute frame deadline' \
    internal/process/absolute_deadline_wbtest.mbt ||
  ! rg -q 'expired absolute read deadline poisons before consuming a response' \
    internal/process/absolute_deadline_wbtest.mbt; then
  printf '%s\n' 'absolute process frame-deadline coverage drifted' >&2
  exit 1
fi

task_dir=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-process-child-asan.XXXXXX")
cleanup() { rm -rf "$task_dir"; }
trap cleanup EXIT HUP INT TERM

cc_bin=${CC:-/usr/bin/cc}
"$cc_bin" -std=c11 -Wall -Wextra -Werror -g \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"${MOON_HOME:-$HOME/.moon}/include" \
  internal/process/child_control_asan_probe.c \
  internal/process/child_control.c \
  -o "$task_dir/probe"

ASAN_OPTIONS=detect_leaks=0:fast_unwind_on_malloc=0 \
  sh scripts/run-sanitized.sh "$task_dir/probe"

"$cc_bin" -std=c11 -Wall -Wextra -Werror -g \
  -fsanitize=address -fno-omit-frame-pointer \
  -I"${MOON_HOME:-$HOME/.moon}/include" \
  internal/process/process_io_asan_probe.c \
  internal/process/process_io.c \
  internal/process/process_nonblocking.c \
  -o "$task_dir/io-probe"

ASAN_OPTIONS=detect_leaks=0:fast_unwind_on_malloc=0 \
  sh scripts/run-sanitized.sh "$task_dir/io-probe"

"$cc_bin" -std=c11 -Wall -Wextra -Werror -g \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"${MOON_HOME:-$HOME/.moon}/include" \
  internal/process/inherited_wait_asan_probe.c \
  internal/process/inherited_wait.c \
  internal/process/process_io.c \
  -o "$task_dir/inherited-wait-probe"

ASAN_OPTIONS=detect_leaks=0:fast_unwind_on_malloc=0 \
  sh scripts/run-sanitized.sh "$task_dir/inherited-wait-probe"

"$cc_bin" -std=c11 -Wall -Wextra -Werror -g \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"${MOON_HOME:-$HOME/.moon}/include" \
  internal/process/process_spawn_asan_probe.c \
  internal/process/process_io.c \
  internal/process/process_approved_spawn.c \
  -o "$task_dir/spawn-probe"

ASAN_OPTIONS=detect_leaks=0:fast_unwind_on_malloc=0 \
  sh scripts/run-sanitized.sh "$task_dir/spawn-probe"

printf '%s\n' 'LunaFlux process child-control and nonblocking-I/O sanitizer gate passed.'
