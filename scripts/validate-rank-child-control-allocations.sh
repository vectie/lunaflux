#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon run tests/rank_child_control_alloc --target native --release --deny-warn

generated_c="_build/native/release/build/tests/rank_child_control_alloc/rank_child_control_alloc.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'rank-child allocation release C output is missing' >&2
  exit 1
fi

for symbol in \
  'RankChildControl14begin__receive(' \
  'RankChildControl17progress__receive(' \
  'RankChildControl14accept__submit(' \
  'RankChildControl23begin__leader__complete(' \
  'RankChildControl32begin__graph__telemetry__sidecar(' \
  'RankChildControl14progress__send('; do
  if ! rg -Fq "$symbol" "$generated_c"; then
    printf 'rank-child allocation symbol was not emitted/executed: %s\n' \
      "$symbol" >&2
    exit 1
  fi
done

for marker in \
  'lunaflux_rank_child_fixture_send' \
  'lunaflux_rank_child_fixture_receive' \
  'lunaflux_rank_child_fixture_check_evidence' \
  'lunaflux_alloc_probe_begin' \
  'lunaflux_alloc_probe_end' \
  'lunaflux_alloc_probe_check'; do
  if ! rg -Fq "$marker" "$generated_c"; then
    printf 'rank-child allocation evidence is missing: %s\n' "$marker" >&2
    exit 1
  fi
done

if ! rg -Fq 'FixedArray::make(8, alloc_probe_seed())' \
  tests/rank_child_control_alloc/fixture_api.mbt; then
  printf '%s\n' 'rank-child allocation positive control is missing' >&2
  exit 1
fi

if ! tail -n +3 tests/rank_child_control_alloc/allocation_probe.c | \
  cmp -s - tests/rank_group_wire_alloc/allocation_probe.c; then
  printf '%s\n' 'rank-child allocation probe drifted from proven wire probe' >&2
  exit 1
fi

if rg -n 'native-stub.*\.\.|"\.\./.*\.c"' \
  tests/rank_child_control_alloc/moon.pkg; then
  printf '%s\n' 'rank-child allocation package escaped its owned stub root' >&2
  exit 1
fi

for file in tests/rank_child_control_alloc/*; do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    printf '%s\n' "$file exceeds the strict 499-line allocation harness budget" >&2
    exit 1
  fi
done

printf '%s\n' 'rank-child warmed release allocation gate passed.'
