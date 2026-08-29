#!/usr/bin/env bash

set -euo pipefail
LC_ALL=C
TZ=UTC
export LC_ALL TZ
umask 077

fail() {
  printf 'LunaFlux FP8 v2 compile evidence rejected: %s\n' "$1" >&2
  exit 1
}

usage() {
  printf '%s\n' \
    'usage: run-fp8-v2-compile-evidence.sh ABSOLUTE_NVCC_13_1 ABSOLUTE_NEW_EVIDENCE_DIR' >&2
  exit 2
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

manifest_value() {
  local key=$1
  local file=$2
  sed -n "s/^${key}=//p" "$file"
}

[ "$#" -eq 2 ] || usage
nvcc=$1
evidence_dir=$2
case "$nvcc" in /*) ;; *) usage ;; esac
case "$evidence_dir" in /*) ;; *) usage ;; esac
[ -f "$nvcc" ] && [ -x "$nvcc" ] && [ ! -L "$nvcc" ] || usage
[ "$(realpath -- "$nvcc")" = "$nvcc" ] || usage
[ ! -e "$evidence_dir" ] && [ ! -L "$evidence_dir" ] ||
  fail 'evidence directory already exists'
evidence_parent=$(CDPATH= cd -- "$(dirname -- "$evidence_dir")" && pwd -P)
[ "$evidence_parent/$(basename -- "$evidence_dir")" = "$evidence_dir" ] ||
  fail 'evidence directory path is not canonical'

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/immutable-evidence-directory.sh"
case "$evidence_dir" in "$repo_root"|"$repo_root"/*)
  fail 'evidence directory must be outside the source tree'
esac

mkdir "$evidence_dir"
mkdir "$evidence_dir/candidates" "$evidence_dir/artifacts" \
  "$evidence_dir/compile-records" "$evidence_dir/logs"
stage=toolchain-identity
campaign_complete=0
compile_records_sha=unavailable

seal_evidence() {
  local status=$?
  trap - EXIT
  trap '' HUP INT TERM
  lunaflux_prepare_evidence_manifest "$evidence_dir" || exit 1
  local files_sha=$lunaflux_evidence_manifest_sha256
  local outcome=failed
  local sealed=0
  if [ "$campaign_complete" -eq 1 ] && [ "$status" -eq 0 ]; then
    outcome=fp8-v2-compile-pass
    sealed=1
  fi
  {
    printf '%s\n' 'evidence_schema=lunaflux.fp8-v2.compile-evidence.v1'
    printf 'outcome=%s\n' "$outcome"
    printf 'exit_status=%s\n' "$status"
    printf 'terminal_stage=%s\n' "$stage"
    printf 'compiler_identity_sha256=%s\n' "${compiler_identity:-unavailable}"
    printf 'compile_records_sha256=%s\n' "$compile_records_sha"
    printf 'files_manifest_sha256=%s\n' "$files_sha"
    printf 'evidence_sealed=%s\n' "$sealed"
    printf '%s\n' 'targets=sm_89,sm_90,sm_120'
    printf '%s\n' 'families=qkv_projection,gated_mlp'
    printf '%s\n' 'compile_only=1'
    printf '%s\n' 'device_opened=0'
    printf '%s\n' 'kernel_launches=0'
    printf '%s\n' 'numerical_execution=0'
    printf '%s\n' 'runtime_authority=0'
    printf '%s\n' 'readiness=0'
    printf '%s\n' 'sm120_compile=1'
    printf '%s\n' 'sm120_execution=0'
  } >"$evidence_dir/RESULT.txt"
  lunaflux_seal_evidence_directory "$evidence_dir" || exit 1
  if [ "$status" -eq 0 ]; then
    cat "$evidence_dir/RESULT.txt"
  fi
  exit "$status"
}
trap seal_evidence EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$repo_root"
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" \
  >"$evidence_dir/compiler.v1" 2>"$evidence_dir/logs/compiler-inspection.stderr" ||
  fail 'CUDA toolchain identity inspection failed'
[ ! -s "$evidence_dir/logs/compiler-inspection.stderr" ] ||
  fail 'CUDA toolchain inspection emitted stderr'
compiler_version=$(manifest_value compiler_version "$evidence_dir/compiler.v1")
[ "$compiler_version" = 13.1.115 ] || fail 'compiler must be exact CUDA 13.1.115'
compiler_identity=$(manifest_value driver_identity_sha256 "$evidence_dir/compiler.v1")
printf '%s\n' "$compiler_identity" | grep -Eq '^[0-9a-f]{64}$' ||
  fail 'compiler identity digest is malformed'

stage=build-candidate-emitter
moon build tests/fp8_v2_compile_evidence --target native --deny-warn \
  --warn-list +73 >"$evidence_dir/logs/moon-build.stdout" \
  2>"$evidence_dir/logs/moon-build.stderr"
emitter=$repo_root/_build/native/debug/build/tests/fp8_v2_compile_evidence/fp8_v2_compile_evidence.exe
[ -x "$emitter" ] || fail 'FP8 v2 candidate emitter is missing'

: >"$evidence_dir/compile-records.sha256"
for target in sm_89 sm_90 sm_120; do
  stage="emit-$target"
  candidate_dir=$evidence_dir/candidates/$target
  artifact_dir=$evidence_dir/artifacts/$target
  record_dir=$evidence_dir/compile-records/$target
  mkdir "$candidate_dir" "$artifact_dir" "$record_dir"
  "$emitter" "$target" "$compiler_identity" "$candidate_dir" \
    >"$evidence_dir/logs/emitter-$target.stdout" \
    2>"$evidence_dir/logs/emitter-$target.stderr"
  [ ! -s "$evidence_dir/logs/emitter-$target.stderr" ] ||
    fail "candidate emitter produced stderr for $target"
  grep -Eq "^outcome=fp8-v2-candidates-emitted target=$target model_content_sha256=[0-9a-f]{64} model_plan_sha256=[0-9a-f]{64} candidate_manifest_sha256=[0-9a-f]{64} runtime_authority=0 numerical_execution=0 readiness=0$" \
    "$evidence_dir/logs/emitter-$target.stdout" ||
    fail "candidate emitter evidence is malformed for $target"
  candidate_manifest=$candidate_dir/candidates.v1
  candidate_manifest_sha=$(sha256_file "$candidate_manifest")
  emitted_manifest_sha=$(sed -n \
    's/.* candidate_manifest_sha256=\([0-9a-f][0-9a-f]*\) runtime_authority=.*/\1/p' \
    "$evidence_dir/logs/emitter-$target.stdout")
  [ "$candidate_manifest_sha" = "$emitted_manifest_sha" ] ||
    fail "emitted candidate manifest digest mismatch for $target"
  candidate_plan_sha=$(manifest_value model_plan_sha256 "$candidate_manifest")
  printf '%s\n' "$candidate_plan_sha" | grep -Eq '^[0-9a-f]{64}$' ||
    fail "candidate model plan digest is malformed for $target"
  [ "$(manifest_value target "$candidate_manifest")" = "$target" ] ||
    fail "candidate target mismatch for $target"
  [ "$(manifest_value toolchain_sha256 "$candidate_manifest")" = "$compiler_identity" ] ||
    fail "candidate compiler identity mismatch for $target"
  [ "$(manifest_value compiler_version "$candidate_manifest")" = 13.1.115 ] ||
    fail "candidate compiler version mismatch for $target"
  [ "$(manifest_value candidate_count "$candidate_manifest")" = 2 ] ||
    fail "candidate count mismatch for $target"
  for family in simple gated; do
    stage="compile-$target-$family"
    source=$candidate_dir/$family.cu
    recipe=$candidate_dir/$family.recipe
    source_sha=$(sha256_file "$source")
    recipe_sha=$(sha256_file "$recipe")
    [ "$source_sha" = "$(manifest_value "${family}_source_sha256" "$candidate_manifest")" ] ||
      fail "source digest mismatch for $target/$family"
    [ "$recipe_sha" = "$(manifest_value "${family}_recipe_sha256" "$candidate_manifest")" ] ||
      fail "recipe digest mismatch for $target/$family"
    [ "$(manifest_value target "$recipe")" = "$target" ] ||
      fail "recipe target mismatch for $target/$family"
    [ "$(manifest_value toolchain_sha256 "$recipe")" = "$compiler_identity" ] ||
      fail "recipe compiler mismatch for $target/$family"
    [ "$(manifest_value compiler_version "$recipe")" = "$compiler_version" ] ||
      fail "recipe compiler version mismatch for $target/$family"
    [ "$(manifest_value source_sha256 "$recipe")" = "$source_sha" ] ||
      fail "recipe source mismatch for $target/$family"
    cubin=$artifact_dir/$family.cubin
    compile_stdout=$evidence_dir/logs/nvcc-$target-$family.stdout
    compile_stderr=$evidence_dir/logs/nvcc-$target-$family.stderr
    "$nvcc" -std=c++17 -O3 --fmad=false --prec-div=true \
      --prec-sqrt=true --ftz=false --maxrregcount=128 -arch="$target" \
      --cubin "$source" -o "$cubin" >"$compile_stdout" 2>"$compile_stderr"
    [ ! -s "$compile_stdout" ] && [ ! -s "$compile_stderr" ] ||
      fail "nvcc emitted output for $target/$family"
    [ -s "$cubin" ] || fail "nvcc produced an empty CUBIN for $target/$family"
    magic=$(od -An -tx1 -N4 "$cubin" | tr -d ' \n')
    [ "$magic" = 7f454c46 ] ||
      fail "nvcc did not produce an ELF CUBIN for $target/$family"
    artifact_sha=$(sha256_file "$cubin")
    record=$record_dir/$family.v1
    {
      printf '%s\n' 'schema=lunaflux.fp8-v2.compile-record.v1'
      printf 'family=%s\n' "$(manifest_value "${family}_family" "$candidate_manifest")"
      printf 'target=%s\n' "$target"
      printf 'model_plan_sha256=%s\n' "$candidate_plan_sha"
      printf 'candidate_manifest_sha256=%s\n' "$candidate_manifest_sha"
      printf 'source_sha256=%s\n' "$source_sha"
      printf 'recipe_sha256=%s\n' "$recipe_sha"
      printf 'compiler_identity_sha256=%s\n' "$compiler_identity"
      printf 'compiler_version=%s\n' "$compiler_version"
      printf 'artifact_sha256=%s\n' "$artifact_sha"
      printf '%s\n' 'artifact_format=elf-cubin'
      printf '%s\n' 'compile_stdout_bytes=0'
      printf '%s\n' 'compile_stderr_bytes=0'
      printf '%s\n' 'compile_only=1'
      printf '%s\n' 'numerical_execution=0'
      printf '%s\n' 'readiness=0'
    } >"$record"
    printf '%s  %s/%s.v1\n' "$(sha256_file "$record")" "$target" "$family" \
      >>"$evidence_dir/compile-records.sha256"
  done
done

stage=sealed
compile_records_sha=$(sha256_file "$evidence_dir/compile-records.sha256")
campaign_complete=1
