#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
work=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-fused-v2-verifier-test.XXXXXX")
work=$(CDPATH= cd -- "$work" && pwd -P)
cleanup() { chmod -R u+rwX "$work" 2>/dev/null || true; rm -rf -- "$work"; }
trap cleanup EXIT HUP INT TERM
evidence=$work/good
mkdir -p "$evidence/artifacts" "$evidence/measurements"
for family in qkv readonly residual; do
  placeholder=$work/$family.cubin
  printf '%s\n' "production-v2-$family-cubin-placeholder" >"$placeholder"
  digest=$(sha256_file "$placeholder")
  destination=$evidence/artifacts/builds/$family-first/kernels/sha256/$digest.cubin
  mkdir -p "$(dirname -- "$destination")"
  cp "$placeholder" "$destination"
  eval "${family}_sha=$digest"
done
printf '%s\n' 'schema=lunaflux-fused-production-v2-spawn-join.v1' \
  "qkv_artifact_relative=artifacts/builds/qkv-first/kernels/sha256/$qkv_sha.cubin" \
  "qkv_module_sha256=$qkv_sha" "qkv_source_sha256=$(printf 1%.0s {1..64})" \
  'qkv_symbol=lunaflux_qkv_v2' \
  "readonly_artifact_relative=artifacts/builds/readonly-first/kernels/sha256/$readonly_sha.cubin" \
  "readonly_module_sha256=$readonly_sha" "readonly_source_sha256=$(printf 2%.0s {1..64})" \
  'readonly_symbol=lunaflux_readonly_v2' \
  "residual_artifact_relative=artifacts/builds/residual-first/kernels/sha256/$residual_sha.cubin" \
  "residual_module_sha256=$residual_sha" "residual_source_sha256=$(printf 3%.0s {1..64})" \
  'residual_symbol=lunaflux_residual_v2' "identities_sha256=$(printf 4%.0s {1..64})" \
  'aggregate_runtime=externally-prepared-required' 'normal_spawn_admission=not-claimed' \
  'qualification_only=true' 'promotion_authority=absent' >"$evidence/artifacts/SPAWN_JOIN.v1"
spawn_join_sha=$(sha256_file "$evidence/artifacts/SPAWN_JOIN.v1")
printf '%s\n' '========= ERROR SUMMARY: 0 errors' >"$evidence/measurements/memcheck.log"
policy_sha=$(printf policy | shasum -a 256 | awk '{print $1}')
. "$root/scripts/immutable-evidence-directory.sh"
lunaflux_prepare_evidence_manifest "$evidence"
inner_sha=$lunaflux_evidence_manifest_sha256
printf '%s\n' \
  'schema=lunaflux-fused-production-v2-physical-campaign.v1' \
  'outcome=fused-production-v2-physical-campaign-pass' \
  'target=sm_120' 'compiler_version=13.1.115' \
  "approved_policy_sha256=$policy_sha" \
  "nvcc_sha256=$(printf nvcc | shasum -a 256 | awk '{print $1}')" \
  "compute_sanitizer_sha256=$(printf sanitizer | shasum -a 256 | awk '{print $1}')" \
  "identities_sha256=$(printf identities | shasum -a 256 | awk '{print $1}')" \
  "spawn_join_sha256=$spawn_join_sha" \
  'cycles=1' 'launches=24' \
  'production_abis=qkv-v2,readonly-attention-v2,residual-rmsnorm-v2' \
  'executor=ordered-kernel-executor' 'graph_policy=capture-required' \
  'graph_mode=captured' 'oracle=standalone-cuda' \
  'memcheck_errors=0' 'racecheck_errors=0' 'initcheck_errors=0' \
  'physical_cuda_observed=true' 'qualification_only=true' \
  'manifest_bindable=false' 'promotion_authority=absent' \
  "evidence_files_manifest_sha256=$inner_sha" >"$evidence/RESULT.txt"
{
  printf '%s  FILES.sha256\n' "$(sha256_file "$evidence/FILES.sha256")"
  printf '%s  RESULT.txt\n' "$(sha256_file "$evidence/RESULT.txt")"
} >"$evidence/OUTER_SEAL.sha256"
outer_sha=$(sha256_file "$evidence/OUTER_SEAL.sha256")
lunaflux_seal_evidence_directory "$evidence"
scripts/verify-fused-production-v2-physical-campaign.sh "$evidence" "$outer_sha" "$policy_sha" >/dev/null

hostile_copy() { cp -R "$evidence" "$1"; chmod -R u+rwX "$1"; }
hostile_copy "$work/substitution"
printf '%s\n' substituted >"$work/substitution/artifacts/SPAWN_JOIN.v1"
chmod -R a-w "$work/substitution"
if scripts/verify-fused-production-v2-physical-campaign.sh "$work/substitution" "$outer_sha" "$policy_sha" >/dev/null 2>&1; then
  echo 'verifier accepted payload substitution' >&2; exit 1
fi
hostile_copy "$work/extra"
printf '%s\n' extra >"$work/extra/artifacts/extra.cubin"
chmod -R a-w "$work/extra"
if scripts/verify-fused-production-v2-physical-campaign.sh "$work/extra" "$outer_sha" "$policy_sha" >/dev/null 2>&1; then
  echo 'verifier accepted unsealed extra file' >&2; exit 1
fi
if scripts/verify-fused-production-v2-physical-campaign.sh "$evidence" "$(printf wrong | shasum -a 256 | awk '{print $1}')" "$policy_sha" >/dev/null 2>&1; then
  echo 'verifier accepted wrong independent outer pin' >&2; exit 1
fi
echo 'LunaFlux fused production V2 hostile evidence verifier tests passed.'
