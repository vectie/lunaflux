#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

rg -q 'terminal_cached_input_tokens' scheduler/core/owner_types.mbt
rg -q 'notice\.cached_input_tokens\(\)' scheduler/core/lifecycle.mbt
rg -q 'publication\.cached_input_tokens\(\)' service/online_session/progression.mbt
rg -q 'publication\.cached_input_tokens\(\)' service/online_session/termination.mbt
rg -q 'publication\.cached_input_tokens\(\)' service/online_session/online_multi_progress.mbt
rg -q 'validate_spawned_prefix_reuse' ops/runtime_instance/spawned_prefix_validation.mbt
rg -q 'prefix_hits == 1UL' ops/runtime_instance/spawned_prefix_check.mbt
rg -q 'prefix_publications == 1UL' ops/runtime_instance/spawned_prefix_check.mbt

if rg -n 'Scheduler::new|PrefixIndex::new|prepare_owned' \
  ops/runtime_instance/spawned_prefix_*.mbt >/dev/null; then
  echo "spawned prefix validator introduced a parallel mutable owner" >&2
  exit 1
fi

if rg -n 'device|internal/cuda' scheduler/core/moon.pkg prefix/radix/moon.pkg \
  >/dev/null; then
  echo "scheduler/prefix dependency boundary leaked device authority" >&2
  exit 1
fi

moon check ops/runtime_instance --target native --deny-warn
moon test scheduler/core/prefix_wbtest.mbt --target native --deny-warn
moon test engine/worker_service/online_lanes_wbtest.mbt --target native --deny-warn
moon test service/online_session/online_multi_wbtest.mbt --target native --deny-warn
moon test ops/runtime_instance/spawned_prefix_validation_wbtest.mbt \
  --target native --deny-warn

echo "LunaFlux spawned prefix-reuse ownership gate passed."
