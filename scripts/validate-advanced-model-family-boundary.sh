#!/bin/sh

set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
exec moon run scripts/validate-advanced-model-family-boundary.mbtx "$@"
