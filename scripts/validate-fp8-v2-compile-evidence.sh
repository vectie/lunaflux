#!/usr/bin/env bash

set -euo pipefail
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
runner=scripts/run-fp8-v2-compile-evidence.sh
evidence_helper=scripts/immutable-evidence-directory.sh
package=tests/fp8_v2_compile_evidence
failed=0

fail() {
  printf '%s\n' "$1" >&2
  failed=1
}

bash -n "$runner" "$evidence_helper" ||
  fail 'FP8 v2 compile-evidence runner or evidence helper is invalid'
for anchor in \
  '. "$repo_root/scripts/immutable-evidence-directory.sh"' \
  'lunaflux_prepare_evidence_manifest "$evidence_dir"' \
  'lunaflux_seal_evidence_directory "$evidence_dir"'; do
  grep -Fq "$anchor" "$runner" ||
    fail "FP8 v2 compile-evidence helper join is missing: $anchor"
done
for anchor in \
  'compiler must be exact CUDA 13.1.115' \
  'for target in sm_89 sm_90 sm_120' \
  'for family in simple gated' \
  'derive_fp8_projection_operands_v2' \
  'lower_fp8_projection_cuda_aot_candidate_v2' \
  'simple_operation_id=2' \
  'gated_operation_id=8' \
  'caller_source_authority=0' \
  'runtime_authority=0' \
  'numerical_execution=0' \
  'readiness=0' \
  'compile_only=1' \
  'sm120_compile=1' \
  'sm120_execution=0' \
  'nvcc emitted output' \
  'nvcc did not produce an ELF CUBIN' \
  'compiler_identity_sha256=%s' \
  'candidate_manifest_sha256=%s' \
  'model_plan_sha256=%s' \
  'source_sha256=%s' \
  'recipe_sha256=%s' \
  'artifact_sha256=%s' \
  'evidence_sealed=%s'; do
  rg -Fq "$anchor" "$runner" "$package" ||
    fail "FP8 v2 compile-evidence anchor is missing: $anchor"
done

if rg -ni 'internal/cuda|cuModuleLoad|cuLaunch|nvrtc|\.ptx|--ptx|runtime.{0,12}compil|readiness=1|numerical_execution=1' \
  "$runner" "$package" --glob '*.mbt' --glob '*.sh' --glob '*.md'; then
  fail 'FP8 v2 compile evidence crossed into execution/JIT/readiness authority'
fi
while IFS= read -r file; do
  [ "$(wc -l <"$file")" -lt 500 ] ||
    fail "FP8 v2 compile-evidence file exceeds 499 lines: $file"
done < <(find "$package" -type f \( -name '*.mbt' -o -name '*.sh' \) -print)
[ "$(wc -l <"$runner")" -lt 500 ] ||
  fail 'FP8 v2 compile-evidence runner exceeds 499 lines'

scratch=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-fp8-v2-compile-gate.XXXXXX")
scratch=$(CDPATH= cd -- "$scratch" && pwd -P)
cleanup() {
  chmod -R u+rwX "$scratch" 2>/dev/null || true
  rm -rf -- "$scratch"
}
trap cleanup EXIT HUP INT TERM
mkdir "$scratch/toolchain"
cp "$package/fixtures/fake_nvcc.sh" "$scratch/toolchain/nvcc"
cp "$package/fixtures/fake_ptxas.sh" "$scratch/toolchain/ptxas"
chmod 0555 "$scratch/toolchain/nvcc" "$scratch/toolchain/ptxas"

if "$runner" relative-nvcc "$scratch/relative" >/dev/null 2>&1; then
  fail 'runner accepted relative NVCC authority'
fi
mkdir "$scratch/existing"
if "$runner" "$scratch/toolchain/nvcc" "$scratch/existing" >/dev/null 2>&1; then
  fail 'runner accepted an existing evidence directory'
fi

printf '%s\n' quiet >"$scratch/toolchain/mode"
"$runner" "$scratch/toolchain/nvcc" "$scratch/passed" \
  >"$scratch/passed.stdout"
grep -Fq 'outcome=fp8-v2-compile-pass' "$scratch/passed/RESULT.txt" ||
  fail 'synthetic compile campaign did not pass'
grep -Fq 'evidence_sealed=1' "$scratch/passed/RESULT.txt" ||
  fail 'synthetic compile evidence was not sealed'
[ "$(find "$scratch/passed/artifacts" -type f -name '*.cubin' | wc -l | tr -d ' ')" -eq 6 ] ||
  fail 'synthetic compile campaign did not emit six CUBINs'
[ "$(wc -l <"$scratch/passed/compile-records.sha256" | tr -d ' ')" -eq 6 ] ||
  fail 'synthetic compile campaign did not seal six compile records'
if find "$scratch/passed" -type f -perm -0200 | grep -q .; then
  fail 'sealed synthetic evidence remained writable'
fi

for mode in noisy nonelf wrong-version; do
  printf '%s\n' "$mode" >"$scratch/toolchain/mode"
  if "$runner" "$scratch/toolchain/nvcc" "$scratch/rejected-$mode" \
    >/dev/null 2>&1; then
    fail "runner accepted hostile compiler mode: $mode"
  fi
  grep -Fq 'evidence_sealed=0' "$scratch/rejected-$mode/RESULT.txt" ||
    fail "hostile compiler failure was not preserved: $mode"
done

moon check "$package" --target native --deny-warn --warn-list +73
moon test "$package" --target native --deny-warn --warn-list +73
scripts/validate-immutable-evidence-directory.sh

if [ "$failed" -ne 0 ]; then exit 1; fi
printf '%s\n' 'LunaFlux FP8 v2 compile-evidence boundary passed.'
