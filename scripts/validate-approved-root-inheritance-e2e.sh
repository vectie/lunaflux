#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

moon build cmd/approved_root_echo tests/approved_root_inheritance_e2e \
  --target native --release --deny-warn --warn-list +73

worker="$repo_root/_build/native/release/build/cmd/approved_root_echo/approved_root_echo.exe"
gate="$repo_root/_build/native/release/build/tests/approved_root_inheritance_e2e/approved_root_inheritance_e2e.exe"

if [ ! -x "$worker" ] || [ ! -x "$gate" ]; then
  printf '%s\n' 'approved-root inheritance executables are missing' >&2
  exit 1
fi

LUNAFLUX_INHERITANCE_SENTINEL=present "$gate" "$worker"
printf '%s\n' 'LunaFlux approved-root inheritance end-to-end gate passed.'
