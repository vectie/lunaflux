#!/bin/sh

set -eu

if [ "$#" -gt 1 ]; then
  printf '%s\n' 'usage: validate-moonbit-annotations.sh [repository-root]' >&2
  exit 2
fi

if [ "$#" -eq 1 ]; then
  repo_root=$(CDPATH= cd -- "$1" && pwd -P)
else
  repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
fi

cd "$repo_root"
moon check --target native --deny-warn --warn-list +73

printf '%s\n' 'LunaFlux MoonBit annotation boundary is valid.'
