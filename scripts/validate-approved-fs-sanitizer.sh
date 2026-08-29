#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

task_dir=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-approved-fs-asan.XXXXXX")
cleanup() { rm -rf "$task_dir"; }
trap cleanup EXIT HUP INT TERM

cc_bin=${CC:-/usr/bin/cc}
"$cc_bin" -std=c11 -Wall -Wextra -Werror -g \
  -fsanitize=address -fno-omit-frame-pointer \
  -I"${MOON_HOME:-$HOME/.moon}/include" \
  internal/approved_fs/asan_probe.c internal/approved_fs/traversal.c \
  internal/approved_fs/identity.c internal/approved_fs/inheritance.c \
  -o "$task_dir/probe"

ASAN_OPTIONS=detect_leaks=0:fast_unwind_on_malloc=0 \
  sh scripts/run-sanitized.sh "$task_dir/probe"

printf '%s\n' 'LunaFlux approved filesystem AddressSanitizer gate passed.'
