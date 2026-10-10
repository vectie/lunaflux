#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)
cd "$repo_root"

exec moon run scripts/validate-fp8-release-authority-boundary.mbtx
