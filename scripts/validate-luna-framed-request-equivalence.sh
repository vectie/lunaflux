#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test \
  service/framed_wire/luna_request_work_wbtest.mbt \
  service/framed_wire/luna_request_hostile_wbtest.mbt \
  service/framed_wire/luna_request_lifecycle_wbtest.mbt \
  --target native --release --deny-warn --warn-list +73

printf '%s\n' 'LunaFlux bounded framed-request frozen-corpus gate passed.'
