#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/luna-bf16-kernel-producer-common.sh"

test_root=$(realpath -- "$(mktemp -d /tmp/lunaflux-kernel-root-test.XXXXXX)")
cleanup() {
  chmod -R u+w "$test_root" 2>/dev/null || true
  rm -rf -- "$test_root"
}
trap cleanup EXIT HUP INT TERM

make_source() {
  source_root=$1
  manifest_text=$2
  module_name=$3
  mkdir "$source_root" "$source_root/payload" "$source_root/payload/sha256"
  printf '%s\n' "$manifest_text" >"$source_root/payload/lunaflux.execution.json"
  printf '%s\n' 'fixture-cubin' >"$source_root/payload/sha256/$module_name.cubin"
  {
    printf '%s  %s\n' \
      "$(lbf_sha256_file "$source_root/payload/lunaflux.execution.json")" \
      lunaflux.execution.json
    printf '%s  %s\n' \
      "$(lbf_sha256_file "$source_root/payload/sha256/$module_name.cubin")" \
      "sha256/$module_name.cubin"
  } >"$source_root/kernel.files.sha256"
  manifest_sha=$(lbf_sha256_file "$source_root/payload/lunaflux.execution.json")
  inventory_sha=$(lbf_sha256_file "$source_root/kernel.files.sha256")
  {
    printf '%s\n' 'schema=lunaflux-bf16-kernel-root-identity.v1'
    printf 'execution_manifest_sha256=%s\n' "$manifest_sha"
    printf 'files_inventory_sha256=%s\n' "$inventory_sha"
  } >"$test_root/identity.tmp"
  identity_sha=$(lbf_sha256_file "$test_root/identity.tmp")
  module_bytes=$(wc -c <"$source_root/payload/sha256/$module_name.cubin" | tr -d ' ')
  {
    printf '%s\n' 'schema=lunaflux-bf16-kernel-root-plan.v1'
    printf '%s\n' 'execution_manifest_relative=lunaflux.execution.json'
    printf 'execution_manifest_sha256=%s\n' "$manifest_sha"
    printf '%s\n' 'manifest_schema_version=2'
    printf '%s\n' 'catalog_version=3'
    printf '%s\n' 'operation_count=9'
    printf '%s\n' 'module_count=1'
    printf '%s\n' 'entry_point_count=9'
    printf 'total_module_bytes=%s\n' "$module_bytes"
    printf 'files_inventory_sha256=%s\n' "$inventory_sha"
    printf 'kernel_root_identity_sha256=%s\n' "$identity_sha"
    printf '%s\n' 'exact_inventory=1'
    printf '%s\n' 'compiler_jit_free=1'
  } >"$source_root/kernel-root.plan.v1"
}

module_digest=$(printf '%s\n' 'fixture-cubin' >"$test_root/module.tmp" &&
  lbf_sha256_file "$test_root/module.tmp")
source_root=$test_root/source
make_source "$source_root" '{"schema_version":2,"modules":[],"operations":[]}' "$module_digest"
plan_sha=$(lbf_sha256_file "$source_root/kernel-root.plan.v1")
output=$test_root/assembled
"$repo_root/scripts/assemble-luna-kernel-root.sh" \
  "$source_root#sha256=$plan_sha" "$output" >"$test_root/assemble.out"
"$repo_root/scripts/verify-luna-kernel-root.sh" "$output" >"$test_root/verify.out"
grep -qx 'kernel_manifest_relative=lunaflux.execution.json' "$test_root/assemble.out"
[ -f "$output/kernel-root/sha256/$module_digest.cubin" ]

runtime=$test_root/reusable-fused-residual.runtime.v1
printf '%s\n' \
  'schema=lunaflux-reusable-fused-residual-rmsnorm-runtime.v1' \
  'payload=fixture' >"$runtime"
runtime_digest=$(lbf_sha256_file "$runtime")
augmented_source=$test_root/augmented-source
"$repo_root/scripts/augment-luna-kernel-root-plan-with-fused-runtime.sh" \
  "$source_root#sha256=$plan_sha" "$runtime#sha256=$runtime_digest" \
  "$augmented_source" >"$test_root/augment.out"
augmented_plan_sha=$(sed -n 's/^kernel_plan_sha256=//p' \
  "$test_root/augment.out")
lbf_is_sha256 "$augmented_plan_sha" ||
  lbf_fail 'reusable fused residual augmentation omitted plan digest'
augmented_output=$test_root/augmented-output
"$repo_root/scripts/assemble-luna-kernel-root.sh" \
  "$augmented_source#sha256=$augmented_plan_sha" "$augmented_output" >/dev/null
"$repo_root/scripts/verify-luna-kernel-root.sh" "$augmented_output" >/dev/null
[ -f "$augmented_output/kernel-root/reusable-fused-residual.runtime.v1" ] ||
  lbf_fail 'reusable fused residual runtime was not assembled'
[ "$(lbf_sha256_file \
  "$augmented_output/kernel-root/reusable-fused-residual.runtime.v1")" = \
  "$runtime_digest" ] || lbf_fail 'reusable fused residual runtime changed'

bundle_runtime=$test_root/reusable-fused-runtime-bundle.v2
printf '%s\n' \
  'schema=lunaflux-reusable-fused-runtime-bundle.v2' \
  'payload=fixture' >"$bundle_runtime"
bundle_runtime_digest=$(lbf_sha256_file "$bundle_runtime")
bundle_augmented_source=$test_root/bundle-augmented-source
"$repo_root/scripts/augment-luna-kernel-root-plan-with-fused-runtime.sh" \
  "$source_root#sha256=$plan_sha" \
  "$bundle_runtime#sha256=$bundle_runtime_digest" \
  "$bundle_augmented_source" >/dev/null
bundle_augmented_plan_sha=$(lbf_sha256_file \
  "$bundle_augmented_source/kernel-root.plan.v1")
bundle_augmented_output=$test_root/bundle-augmented-output
"$repo_root/scripts/assemble-luna-kernel-root.sh" \
  "$bundle_augmented_source#sha256=$bundle_augmented_plan_sha" \
  "$bundle_augmented_output" >/dev/null
"$repo_root/scripts/verify-luna-kernel-root.sh" \
  "$bundle_augmented_output" >/dev/null
[ -f "$bundle_augmented_output/kernel-root/reusable-fused-runtime-bundle.v2" ] ||
  lbf_fail 'reusable fused runtime bundle was not assembled'
[ "$(lbf_sha256_file \
  "$bundle_augmented_output/kernel-root/reusable-fused-runtime-bundle.v2")" = \
  "$bundle_runtime_digest" ] || lbf_fail 'reusable fused runtime bundle changed'

if "$repo_root/scripts/assemble-luna-kernel-root.sh" \
  "$source_root#sha256=$plan_sha" "$output" >/dev/null 2>&1; then
  lbf_fail 'kernel-root assembler overwrote existing output'
fi

ambient=$test_root/ambient-source
cp -R "$source_root" "$ambient"
printf '%s\n' ambient >"$ambient/ambient.bin"
ambient_output=$test_root/ambient-output
if "$repo_root/scripts/assemble-luna-kernel-root.sh" \
  "$ambient#sha256=$plan_sha" "$ambient_output" >/dev/null 2>&1; then
  lbf_fail 'kernel-root assembler accepted ambient material'
fi
[ ! -e "$ambient_output" ] || lbf_fail 'ambient failure left partial output'

ptx_source=$test_root/ptx-source
make_source "$ptx_source" \
  '{"schema_version":2,"runtime_jit":"forbidden"}' "$module_digest"
ptx_plan_sha=$(lbf_sha256_file "$ptx_source/kernel-root.plan.v1")
ptx_output=$test_root/ptx-output
if "$repo_root/scripts/assemble-luna-kernel-root.sh" \
  "$ptx_source#sha256=$ptx_plan_sha" "$ptx_output" >/dev/null 2>&1; then
  lbf_fail 'kernel-root assembler accepted JIT vocabulary'
fi
[ ! -e "$ptx_output" ] || lbf_fail 'JIT rejection left partial output'

wrong_path=$test_root/wrong-path-source
make_source "$wrong_path" '{"schema_version":2}' \
  'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
wrong_plan_sha=$(lbf_sha256_file "$wrong_path/kernel-root.plan.v1")
if "$repo_root/scripts/assemble-luna-kernel-root.sh" \
  "$wrong_path#sha256=$wrong_plan_sha" "$test_root/wrong-path-output" \
  >/dev/null 2>&1; then
  lbf_fail 'kernel-root assembler accepted a non-content-addressed module path'
fi

printf '%s\n' 'Luna deployment kernel-root assembly gate passed'
