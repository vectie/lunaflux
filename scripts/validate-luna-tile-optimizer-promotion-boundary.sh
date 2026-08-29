#!/usr/bin/env bash
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

package=release/luna_tile_optimizer_promotion

if rg -n 'vectie/lunaflux/(engine|runtime|service|device|internal/(cuda|process|approved_fs))' \
  "$package/moon.pkg" >/dev/null; then
  printf '%s\n' 'LunaTile optimizer promotion imports runtime/device authority' >&2
  exit 1
fi

if rg -n 'luna_tile_optimizer_promotion' engine runtime service device \
  --glob 'moon.pkg' >/dev/null; then
  printf '%s\n' 'production runtime imports offline LunaTile promotion evidence' >&2
  exit 1
fi

if rg -n 'LunaExternalApprovalVerifier|LunaExternalApprovalPolicy|LunaExternalSignedApproval|LunaStartupVerifierKeyOwner' \
  "$package" --glob '*.mbt' >/dev/null; then
  printf '%s\n' 'LunaTile promotion constructs verifier or approval authority' >&2
  exit 1
fi

rg -F 'admit_luna_external_approved_record(' "$package/promotion.mbt" >/dev/null
rg -F 'LunaAuthenticatedExternalApproval' "$package/promotion.mbt" >/dev/null
rg -F 'candidate_time_ns >= observation.fallback_time_ns' \
  "$package/benchmark.mbt" >/dev/null
rg -F 'LunaTileBaselineSelected' "$package/promotion.mbt" >/dev/null
rg -F 'LunaTileOptimizerPromotionMismatch(Replay)' \
  "$package/promotion.mbt" >/dev/null
rg -F 'parallel_fallback_digest()' "$package/promotion.mbt" >/dev/null
rg -F 'manifest_bindable()' "$package/promotion.mbt" >/dev/null

for file in "$package"/*.mbt kernels/luna_profile_priority/*.mbt; do
  lines=$(wc -l <"$file" | tr -d ' ')
  if [ "$lines" -ge 500 ]; then
    printf 'LunaTile/profile file exceeds debt ceiling: %s (%s lines)\n' \
      "$file" "$lines" >&2
    exit 1
  fi
done

rg -F 'launch_identity' kernels/luna_profile_priority/paged_types.mbt >/dev/null
rg -F 'append_paged_launch' kernels/luna_profile_priority/paged_canonical.mbt >/dev/null
rg -F 'LunaPagedProfileCounterKind' \
  kernels/luna_profile_priority/paged_capture_types.mbt >/dev/null

printf '%s\n' 'LunaTile optimizer promotion runtime/debt boundary passed.'
