#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
package="$root/engine/rank_group_protocol"

if rg -n 'scheduler/|internal/(cuda|nccl)|device/|model/llama|service/|http' "$package/moon.pkg" "$package"/*.mbt >/dev/null; then
  echo "rank-group protocol crossed a forbidden scheduler, backend, model-family, or service boundary" >&2
  exit 1
fi

if rg -n '\b(Array|Map|HashMap)::|:\s*Array\[|:\s*Map\[' "$package"/*.mbt | rg -v 'fixture_test|_test|_wbtest' >/dev/null; then
  echo "rank-group steady-state source contains dynamic collection storage" >&2
  exit 1
fi

if rg -n 'extern "c"|global mutable|String\)' "$package"/*.mbt >/dev/null; then
  echo "rank-group protocol leaked native, global, or text payload authority" >&2
  exit 1
fi

for required in \
  'self.lost_ranks[rank] = true' \
  'pub fn RankGroupOwner::mark_drain_rank_lost' \
  'additional rank loss cannot strand a faulted group drain' \
  'pub fn RankGroupOwner::reauthenticate_submitted' \
  'pub fn RankGroupOwner::reauthenticate_completed' \
  'submitted capability is rederived only for the exact live plan' \
  'completed capability is rederived from exact accepted epoch evidence' \
  'pub fn RankGroupFailureReport::has_canonical_failure_rank' \
  'failure publication rejects impossible reason and failed-rank pairs'; do
  if ! rg -Fq "$required" "$package"; then
    echo "rank-group failure-drain invariant is missing: $required" >&2
    exit 1
  fi
done

for file in "$package"/*.mbt; do
  lines=$(wc -l < "$file")
  if [ "$lines" -gt 500 ]; then
    echo "$file exceeds 500 lines" >&2
    exit 1
  fi
done

echo "rank-group protocol boundaries: ok"
