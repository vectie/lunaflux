#!/bin/sh
set -eu
LC_ALL=C
export LC_ALL

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"

production_roots='config contracts device engine internal kernels kv model prefix runtime sampling scheduler service tokenizer'
if rg -n '"vectie/lunaflux/release/' $production_roots --glob 'moon.pkg'; then
  printf '%s\n' 'non-release production package imports release/*' >&2
  exit 1
fi

printf '%s\n' 'production packages are release-import-free'
