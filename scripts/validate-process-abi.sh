#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

if rg -n 'system\s*\(|popen\s*\(|fork\s*\(|posix_spawnp\s*\(' \
  internal/process/process.c; then
  printf '%s\n' 'worker process ABI must not invoke a shell, fork, or PATH lookup' >&2
  exit 1
fi

if ! rg -q 'posix_spawn\s*\(' internal/process/process.c ||
  ! rg -q 'socketpair\s*\(' internal/process/process.c ||
  ! rg -q 'CLOCK_MONOTONIC' internal/process/process.c; then
  printf '%s\n' 'worker process ABI is missing exact spawn, private channel, or monotonic timeout' >&2
  exit 1
fi

if rg -n 'extern\s+"[cC]"' internal/process --glob '*.mbt' \
  --glob '!ffi.mbt'; then
  printf '%s\n' 'worker process extern declarations must remain in ffi.mbt' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux worker process ABI boundary is valid.'
