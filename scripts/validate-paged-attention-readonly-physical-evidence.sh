#!/bin/sh
set -eu
LC_ALL=C
export LC_ALL

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
package=release/luna_paged_attention_readonly_physical_evidence

for file in "$package"/*.mbt; do
  lines=$(wc -l <"$file" | tr -d ' ')
  [ "$lines" -lt 500 ] || {
    printf 'read-only physical evidence file exceeds budget: %s (%s)\n' "$file" "$lines" >&2
    exit 1
  }
done

if rg -n 'internal/(cuda|process|approved_fs)|vectie/lunaflux/(engine|runtime|service|cmd)/|extern[[:space:]]+"c"|@fs\.|@process\.|async[[:space:]]+fn' \
  "$package" --glob '*.mbt' --glob 'moon.pkg'; then
  printf '%s\n' 'read-only physical evidence gained active runtime authority' >&2
  exit 1
fi

for required in \
  'READONLY_PHYSICAL_RESULT_LINES : Int = 98' \
  'lunaflux-paged-attention-readonly-physical-campaign.v1' \
  'standalone-positioned-rope-paged-kvwrite-complete-v1' \
  'dispatch_canary_publication", "exactly-once-after-output"' \
  'dispatch_canary_tail_zero", "true"' \
  'cache_snapshot_unchanged", "true"' \
  'input_guards_unchanged", "true"' \
  'output_guards_unchanged", "true"' \
  'cpu_oracle", "independent-ordered-f32-v1"' \
  'serial_cuda_oracle", "independent-ordered-f32-kernel-v1"' \
  'compiler_version", "13.1.115"' \
  '"device_uuid", approved_policy.device_uuid' \
  'approved_policy.device_pci_bus_id' \
  '"scheduler_modes"' \
  'reader.exact_int("dispatch_grid_x", profile.max_query_tokens(), Candidate)' \
  'physical_cuda_observed", "true"' \
  'manifest_bindable", "false"' \
  'promotion_authority", "absent"' \
  'pub fn admit_paged_attention_readonly_physical_evidence(' \
  'pub fn bind_qualified_paged_attention_readonly_artifact('; do
  rg -F -q "$required" "$package" --glob '*.mbt' || {
    printf 'read-only physical evidence invariant missing: %s\n' "$required" >&2
    exit 1
  }
done

for hostile in \
  'hostile read-only physical evidence was admitted' \
  'mutated result passed stale outer seal' \
  'critical manifest path replay was admitted'; do
  rg -F -q "$hostile" "$package" --glob '*_wbtest.mbt' || {
    printf 'read-only physical hostile test missing: %s\n' "$hostile" >&2
    exit 1
  }
done

moon fmt --check "$package"
moon check "$package" --target native --deny-warn --warn-list +73
moon test "$package" --target native --deny-warn --warn-list +73
moon info "$package" --target native >/dev/null

interface=$package/pkg.generated.mbti
rg -U -q 'pub struct AdmittedPagedAttentionReadOnlyPhysicalEvidence \{\n  // private fields\n\}' "$interface"
rg -U -q 'pub struct QualifiedPagedAttentionReadOnlyArtifact \{\n  // private fields\n\}' "$interface"
if rg -n '^pub fn (AdmittedPagedAttentionReadOnlyPhysicalEvidence|QualifiedPagedAttentionReadOnlyArtifact)::(new|create|open|execute|promote|bind_manifest)' "$interface"; then
  printf '%s\n' 'read-only physical evidence exposes a fabricator/authority method' >&2
  exit 1
fi

bash -n scripts/run-paged-attention-readonly-physical-campaign.sh
bash -n scripts/test-paged-attention-readonly-physical-campaign.sh
scripts/test-paged-attention-readonly-physical-campaign.sh
printf '%s\n' 'sealed read-only paged-attention physical evidence remains qualification-only: PASS'
