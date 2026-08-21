#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test \
  tokenizer/luna_work_wbtest.mbt \
  tokenizer/luna_work_reference_wbtest.mbt \
  tokenizer/luna_input_write_wbtest.mbt \
  tokenizer/luna_input_equivalence_wbtest.mbt \
  tokenizer/luna_input_integration_wbtest.mbt \
  --target native --release --deny-warn --warn-list +73

printf '%s\n' 'LunaFlux bounded tokenizer-work equivalence gate passed.'
