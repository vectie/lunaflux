#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
tmp=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-device-greedy-hostile.XXXXXX")
tmp=$(CDPATH= cd -- "$tmp" && pwd -P)
cleanup() { chmod -R u+rwX "$tmp" 2>/dev/null || true; rm -rf -- "$tmp"; }
trap cleanup EXIT HUP INT TERM
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
make_fixture() {
  local out=$1
  mkdir "$out" "$out/logs"
  printf '%s\n' synthetic >"$out/logs/memcheck.stdout"
  . "$root/scripts/immutable-evidence-directory.sh"
  lunaflux_prepare_evidence_manifest "$out"
  local files_sha=$lunaflux_evidence_manifest_sha256
  printf '%s\n' 'schema=lunaflux-spawned-device-greedy-physical-campaign.v1' \
    'outcome=spawned-device-greedy-physical-pass' "campaign_executable_sha256=$(printf a%.0s {1..64})" \
    "worker_executable_sha256=$(printf b%.0s {1..64})" \
    "embedded_launch_sha256=$(printf c%.0s {1..64})" \
    "host_launch_sha256=$(printf d%.0s {1..64})" \
    "compute_sanitizer_sha256=$(printf e%.0s {1..64})" \
    "nvidia_smi_sha256=$(printf f%.0s {1..64})" \
    'fused_v2_runtime=optional-absent' 'request_plans=2' 'readback_bytes=16' \
    'host_referee=full-logits-production-route' 'memcheck_errors=0' \
    'racecheck_errors=0' 'initcheck_errors=0' 'resources=closed' \
    'adversarial_tie_policy=source-and-host-contract-only' \
    'adversarial_nonfinite_policy=source-and-host-contract-only' \
    'physical_cuda_observed=true' 'qualification_only=true' \
    'promotion_authority=absent' "evidence_files_manifest_sha256=$files_sha" \
    >"$out/RESULT.txt"
  {
    printf '%s  FILES.sha256\n' "$(sha256_file "$out/FILES.sha256")"
    printf '%s  RESULT.txt\n' "$(sha256_file "$out/RESULT.txt")"
  } >"$out/OUTER_SEAL.sha256"
  FIXTURE_OUTER=$(sha256_file "$out/OUTER_SEAL.sha256")
  lunaflux_seal_evidence_directory "$out"
}
make_fixture "$tmp/good"
"$root/scripts/verify-spawned-device-greedy-physical-campaign.sh" \
  "$tmp/good" "$FIXTURE_OUTER" >/dev/null

cp -R "$tmp/good" "$tmp/substitution"
chmod -R u+w "$tmp/substitution"
printf '%s\n' substituted >"$tmp/substitution/logs/memcheck.stdout"
if "$root/scripts/verify-spawned-device-greedy-physical-campaign.sh" \
  "$tmp/substitution" "$FIXTURE_OUTER" >/dev/null 2>&1; then
  printf '%s\n' 'payload substitution was accepted' >&2; exit 1
fi

cp -R "$tmp/good" "$tmp/extra"
chmod -R u+w "$tmp/extra"
printf '%s\n' ambient >"$tmp/extra/AMBIENT"
if "$root/scripts/verify-spawned-device-greedy-physical-campaign.sh" \
  "$tmp/extra" "$FIXTURE_OUTER" >/dev/null 2>&1; then
  printf '%s\n' 'ambient file was accepted' >&2; exit 1
fi

wrong=$(printf 0%.0s {1..64})
if "$root/scripts/verify-spawned-device-greedy-physical-campaign.sh" \
  "$tmp/good" "$wrong" >/dev/null 2>&1; then
  printf '%s\n' 'wrong outer pin was accepted' >&2; exit 1
fi
printf '%s\n' 'spawned device-greedy verifier hostile tests: pass'
