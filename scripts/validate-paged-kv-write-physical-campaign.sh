#!/bin/sh
set -eu
LC_ALL=C
export LC_ALL

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
runner=scripts/run-paged-kv-write-physical-campaign.sh
test_gate=scripts/test-paged-kv-write-physical-campaign.sh
policy=scripts/validate-paged-kv-write-approved-policy.sh
fixture=scripts/fixtures/paged-kv-write-physical/fake-compute-sanitizer.sh
fail() { printf 'paged KV-write campaign boundary failed: %s\n' "$1" >&2; exit 1; }

for file in "$runner" "$test_gate" "$policy" "$fixture"; do
  [ -f "$file" ] && [ -x "$file" ] || fail "missing executable: $file"
  lines=$(wc -l <"$file" | tr -d ' ')
  [ "$lines" -lt 500 ] || fail "file reached 500 lines: $file"
  bash -n "$file" || fail "invalid shell syntax: $file"
done
for anchor in \
  'APPROVED_POLICY#sha256=DIGEST' \
  'validate-paged-kv-write-approved-policy.sh' \
  'approved policy drifted during campaign' \
  'receipt "$expected_policy_sha"' \
  'candidate CUBIN nondeterministic' \
  'oracle CUBIN nondeterministic' \
  'run_tool memcheck' \
  'run_tool racecheck' \
  'run_tool initcheck' \
  'scalar_oracle_pass=pass' \
  'serial_cuda_oracle_pass=pass' \
  'CUDA context device differs from approved policy' \
  'SASS global load absent' \
  'SASS global store absent' \
  'physical_cuda_observed=%s\nsynthetic_test_only=%s' \
  'files_manifest_sha256=' \
  'outer_seal_sha256=' \
  'lunaflux_seal_evidence_directory "$stage"' \
  'published campaign seal drifted' \
  'physical_cuda_observed=$physical_observed' \
  'manifest_bindable=false' \
  'promotion_authority=absent'; do
  grep -Fq -- "$anchor" "$runner" || fail "runner anchor missing: $anchor"
done
for anchor in \
  'LUNAFLUX_SYNTHETIC_TEST_ONLY=true' \
  'physical_cuda_observed=false' \
  'admission_exercised=false' \
  'public typed admission accepted synthetic canonical replay' \
  'FAKE_LUNA_CUDA_NONDETERMINISTIC=1' \
  'existing campaign overwritten' \
  'unapproved tool substitution passed' \
  'wrong policy pin passed'; do
  grep -Fq -- "$anchor" "$test_gate" || fail "hostile transaction anchor missing: $anchor"
done
if rg -ni 'nvrtc|--ptx|\.ptx|promotion_authority=(present|granted)|manifest_bindable=true|runtime_dispatch_authority=(present|granted)' "$runner" "$test_gate" >/dev/null; then
  fail 'campaign gained JIT, manifest, promotion, or runtime authority'
fi

scripts/validate-paged-kv-write-cuda-probe.sh >/dev/null
scripts/validate-paged-kv-write-physical-evidence.sh >/dev/null
scripts/validate-immutable-evidence-directory.sh >/dev/null
"$test_gate"

printf '%s\n' 'Paged KV-write campaign remains externally pinned, deterministic, sealed, and qualification-only.'
