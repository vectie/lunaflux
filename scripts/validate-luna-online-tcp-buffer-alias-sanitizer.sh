#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

task_dir=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-tcp-alias-asan.XXXXXX")
cleanup() { rm -rf "$task_dir"; }
trap cleanup EXIT HUP INT TERM

cc_bin=/usr/bin/clang
if [ ! -x "$cc_bin" ]; then
  printf '%s\n' 'system clang is required for the online TCP alias sanitizer' >&2
  exit 1
fi

"$cc_bin" -std=c11 -Wall -Wextra -Werror -g \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"${MOON_HOME:-$HOME/.moon}/include" \
  internal/online_tcp_buffer_alias/alias_probe.c \
  internal/online_tcp_buffer_alias/alias.c \
  -o "$task_dir/probe"

ASAN_OPTIONS=detect_leaks=0:fast_unwind_on_malloc=0 "$task_dir/probe"

printf '%s\n' \
  'LunaFlux online TCP exact-TU alias ASan/UBSan gate passed.'
