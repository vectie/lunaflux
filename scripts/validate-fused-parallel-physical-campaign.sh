#!/bin/sh
set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

runner=scripts/run-fused-parallel-physical-campaign.sh
helper=scripts/fused-physical-campaign-functions.sh
test_gate=scripts/test-fused-parallel-physical-campaign.sh
fixture=scripts/fixtures/fused-physical-campaign/fake-compute-sanitizer.sh

fail() {
  printf 'fused physical campaign boundary failed: %s\n' "$1" >&2
  exit 1
}

for file in "$runner" "$test_gate" "$fixture"; do
  [ -f "$file" ] && [ -x "$file" ] || fail "missing executable: $file"
  lines=$(wc -l <"$file" | tr -d ' ')
  [ "$lines" -lt 500 ] || fail "file reached 500 lines: $file"
  bash -n "$file" || fail "invalid shell syntax: $file"
done

[ -f "$helper" ] && [ ! -L "$helper" ] || fail "missing regular helper: $helper"
lines=$(wc -l <"$helper" | tr -d ' ')
[ "$lines" -lt 500 ] || fail "file reached 500 lines: $helper"
bash -n "$helper" || fail "invalid shell syntax: $helper"
grep -Fq '. "$repo_root/scripts/fused-physical-campaign-functions.sh"' "$runner" ||
  fail 'runner does not load the bounded campaign helper'
grep -Fq 'lunaflux_fused_build_family()' "$helper" ||
  fail 'campaign build helper is missing'
grep -Fq 'lunaflux_fused_audit_resources()' "$helper" ||
  fail 'campaign resource-audit helper is missing'

for anchor in \
  'ABSOLUTE_NEW_OUTPUT' \
  'output path already exists' \
  'independent fused CUBIN publications differ' \
  '--tool memcheck' \
  '--tool racecheck' \
  'artifact_seal_sha256=$lunaflux_evidence_manifest_sha256' \
  'evidence_seal_sha256=$lunaflux_evidence_manifest_sha256' \
  'fused-physical-evidence-v1.txt' \
  'canonical_sha256=$(sha256_file "$canonical")' \
  '"$probe" admit' \
  'CAMPAIGN_RESULT.txt' \
  'lunaflux_seal_evidence_directory "$stage"' \
  'chmod 0555 "$output"' \
  'promotion_authority=absent'; do
  grep -Fq -- "$anchor" "$runner" || fail "runner anchor missing: $anchor"
done

for anchor in \
  'parse_fused_campaign_observation' \
  'runtime_stdout == race_stdout' \
  'require_clean_sanitizer_log' \
  'render_fused_campaign_canonical' \
  'admit_fused_physical_evidence' \
  'value.to_string() == field' \
  '========= ERROR SUMMARY: 0 errors' \
  'manifest_bindable()' \
  'promotion_authorized()'; do
  rg -Fq "$anchor" tests/fused_parallel_cuda_probe ||
    fail "typed composition anchor missing: $anchor"
done

if rg -ni \
  'manifest_bindable\([^)]*true|promotion_authority=(present|granted)|runtime.serv|nvrtc|--ptx|\.ptx' \
  "$runner" "$helper" tests/fused_parallel_cuda_probe --glob '*.sh' --glob '*.mbt' \
  >/dev/null; then
  fail 'campaign composition gained runtime, manifest, promotion, PTX, or JIT authority'
fi

scripts/validate-fused-parallel-cuda-probe.sh >/dev/null
scripts/validate-luna-fused-physical-evidence.sh >/dev/null
scripts/validate-immutable-evidence-directory.sh >/dev/null
"$test_gate"

printf '%s\n' \
  'Fused physical campaign remains measured, admitted, non-circular, sealed, and authority-free.'
