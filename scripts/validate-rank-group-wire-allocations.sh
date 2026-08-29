#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon run tests/rank_group_wire_alloc --target native --release --deny-warn

generated_c="_build/native/release/build/tests/rank_group_wire_alloc/rank_group_wire_alloc.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'rank-group wire allocation release C output is missing' >&2
  exit 1
fi

for symbol in \
  'RankGroupWireFrameBuffer15encode__control(' \
  'RankGroupWireFrameBuffer15encode__payload(' \
  'RankGroupWireFrameBuffer4load(' \
  'RankGroupWireTranscript6accept(' \
  'RankGroupWireTranscript17load__and__accept('; do
  if ! rg -Fq "$symbol" "$generated_c"; then
    printf 'rank-group wire allocation symbol was not emitted/executed: %s\n' \
      "$symbol" >&2
    exit 1
  fi
done

for marker in \
  'lunaflux_rank_wire_execution_reset' \
  'lunaflux_rank_wire_execution_mark' \
  'lunaflux_rank_wire_execution_check' \
  'lunaflux_alloc_probe_begin' \
  'lunaflux_alloc_probe_end' \
  'lunaflux_alloc_probe_check'; do
  if ! rg -Fq "$marker" "$generated_c"; then
    printf 'rank-group wire allocation execution evidence is missing: %s\n' \
      "$marker" >&2
    exit 1
  fi
done

if ! rg -Fq 'FixedArray::make(8, alloc_probe_seed())' \
  tests/rank_group_wire_alloc/main.mbt; then
  printf '%s\n' 'rank-group wire allocation positive control is missing' >&2
  exit 1
fi

printf '%s\n' 'rank-group wire warmed release allocation gate passed.'
