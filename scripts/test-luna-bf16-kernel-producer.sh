#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/luna-bf16-kernel-producer-common.sh"

test_root=$(realpath -- "$(mktemp -d /tmp/lunaflux-bf16-producer-test.XXXXXX)")
cleanup() {
  chmod -R u+w "$test_root" 2>/dev/null || true
  rm -rf -- "$test_root"
}
trap cleanup EXIT HUP INT TERM
mkdir "$test_root/toolchain" "$test_root/candidates" "$test_root/candidates/sources" \
  "$test_root/candidates/recipes"
cp "$repo_root/scripts/fixtures/luna-cuda-aot/fake-nvcc.sh" "$test_root/toolchain/nvcc"
cp "$repo_root/scripts/fixtures/luna-cuda-aot/fake-ptxas.sh" "$test_root/toolchain/ptxas"
chmod 555 "$test_root/toolchain/nvcc" "$test_root/toolchain/ptxas"

driver_report=$test_root/driver.txt
"$repo_root/scripts/inspect-luna-cuda-aot-driver.sh" \
  "$test_root/toolchain/nvcc" >"$driver_report"
driver_identity=$(sed -n '6s/^driver_identity_sha256=//p' "$driver_report")
{
  printf '%s\n' 'schema=test-approved-complete-cuda-toolchain-v1'
  printf 'driver_identity_sha256=%s\n' "$driver_identity"
} >"$test_root/toolchain.manifest"
toolchain_sha=$(lbf_sha256_file "$test_root/toolchain.manifest")

cat >"$test_root/candidates/candidate-set.v1" <<'EOF'
schema=lunaflux-bf16-candidate-set.v1
target=sm_120
operation_count=9
candidate=embedding,0,embedding_lookup
candidate=norm,1,rms_norm
candidate=qkv,2,qkv_projection
candidate=rope,3,positioned_rotary
candidate=attention,4,paged_attention
candidate=dense,5,dense_projection
candidate=residual,6,residual_add
candidate=mlp,7,gated_mlp
candidate=head,8,language_model_head
EOF

write_recipe() {
  key=$1
  operation_id=$2
  family=$3
  schema=$4
  source_sha=$5
  recipe=$test_root/candidates/recipes/$key.recipe
  {
    printf 'schema=%s\n' "$schema"
    if [ "$family" != paged_attention ]; then
      printf 'family=%s\n' "$family"
    else
      printf '%s\n' 'mode=mixed-prefill-decode'
    fi
    printf 'operation_id=%s\n' "$operation_id"
    printf 'entry_point_id=%s\n' "$((operation_id + 1))"
    printf 'function_symbol=lunaflux_test_%s\n' "$key"
    printf 'source_sha256=%s\n' "$source_sha"
    printf 'toolchain_sha256=%s\n' "$toolchain_sha"
    printf '%s\n' 'compiler_version=13.1.0'
    printf '%s\n' 'target=sm_120'
    printf '%s\n' 'output=cubin'
    if [ "$family" = paged_attention ]; then
      printf '%s\n' 'language_standard=c++17'
    fi
    printf '%s\n' 'optimization=3'
    printf '%s\n' 'fmad=false'
    printf '%s\n' 'reassociate=false'
    printf '%s\n' 'max_registers=128'
    printf '%s\n' 'grid=1,1,1'
    printf '%s\n' 'block=256,1,1'
    printf '%s\n' 'manifest_bindable=false'
    printf '%s\n' 'operands=fixture'
  } >"$recipe"
}

while IFS=, read -r key operation_id family; do
  source_file=$test_root/candidates/sources/$key.cu
  printf 'extern "C" __global__ void lunaflux_test_%s(void) {}\n' "$key" >"$source_file"
  source_sha=$(lbf_sha256_file "$source_file")
  case "$family" in
    embedding_lookup|rms_norm|positioned_rotary|residual_add)
      schema=lunaflux-luna-cuda-pointwise-aot-recipe-v1
      ;;
    qkv_projection|dense_projection|gated_mlp|language_model_head)
      schema=lunaflux-luna-cuda-projection-aot-recipe-v1
      ;;
    paged_attention) schema=lunaflux-paged-attention-cuda-aot-recipe-v1 ;;
  esac
  write_recipe "$key" "$operation_id" "$family" "$schema" "$source_sha"
done <<'EOF'
embedding,0,embedding_lookup
norm,1,rms_norm
qkv,2,qkv_projection
rope,3,positioned_rotary
attention,4,paged_attention
dense,5,dense_projection
residual,6,residual_add
mlp,7,gated_mlp
head,8,language_model_head
EOF

write_candidate_inventory() {
  output=$1
  find "$test_root/candidates" -type f -print | sed "s#^$test_root/candidates/##" |
    LC_ALL=C sort | while IFS= read -r relative; do
      printf '%s  %s\n' \
        "$(lbf_sha256_file "$test_root/candidates/$relative")" "$relative"
    done >"$output"
}

inventory=$test_root/candidates.files.sha256
write_candidate_inventory "$inventory"
inventory_sha=$(lbf_sha256_file "$inventory")
output=$test_root/compiled
"$repo_root/scripts/build-luna-bf16-kernel-set.sh" \
  "$test_root/toolchain/nvcc" \
  "$test_root/toolchain.manifest#sha256=$toolchain_sha" \
  "$test_root/candidates" \
  "$inventory#sha256=$inventory_sha" \
  "$output" >"$test_root/build.out"
"$repo_root/scripts/verify-luna-bf16-kernel-set.sh" "$output" \
  >"$test_root/verify.out"
grep -qx 'operation_count=9' "$output/compiled-set.v1"
grep -qx 'module_count=1' "$output/compiled-set.v1"
[ "$(find "$output/receipts" -type f -name '*.receipt.v1' | wc -l | tr -d ' ')" -eq 9 ]
[ "$(find "$output/sha256" -type f -name '*.cubin' | wc -l | tr -d ' ')" -eq 1 ]

if "$repo_root/scripts/build-luna-bf16-kernel-set.sh" \
  "$test_root/toolchain/nvcc" \
  "$test_root/toolchain.manifest#sha256=$toolchain_sha" \
  "$test_root/candidates" \
  "$inventory#sha256=$inventory_sha" \
  "$output" >/dev/null 2>&1; then
  lbf_fail 'producer overwrote an existing output'
fi

cp "$output/compiled-set.v1" "$test_root/compiled-set.original"
chmod u+w "$output/receipts/embedding.receipt.v1"
printf '%s\n' 'tamper' >>"$output/receipts/embedding.receipt.v1"
if "$repo_root/scripts/verify-luna-bf16-kernel-set.sh" "$output" >/dev/null 2>&1; then
  lbf_fail 'verifier accepted a substituted receipt'
fi
sed '$d' "$output/receipts/embedding.receipt.v1" >"$test_root/receipt.restored"
cp "$test_root/receipt.restored" "$output/receipts/embedding.receipt.v1"

printf '%s\n' 'ambient' >"$test_root/candidates/ambient.bin"
ambient_output=$test_root/ambient-output
if "$repo_root/scripts/build-luna-bf16-kernel-set.sh" \
  "$test_root/toolchain/nvcc" \
  "$test_root/toolchain.manifest#sha256=$toolchain_sha" \
  "$test_root/candidates" \
  "$inventory#sha256=$inventory_sha" \
  "$ambient_output" >/dev/null 2>&1; then
  lbf_fail 'producer accepted an ambient candidate file'
fi
[ ! -e "$ambient_output" ] || lbf_fail 'failed producer left partial output'
rm "$test_root/candidates/ambient.bin"

nondeterministic_output=$test_root/nondeterministic-output
if FAKE_LUNA_CUDA_NONDETERMINISTIC=1 \
  "$repo_root/scripts/build-luna-bf16-kernel-set.sh" \
  "$test_root/toolchain/nvcc" \
  "$test_root/toolchain.manifest#sha256=$toolchain_sha" \
  "$test_root/candidates" \
  "$inventory#sha256=$inventory_sha" \
  "$nondeterministic_output" >/dev/null 2>&1; then
  lbf_fail 'producer accepted nondeterministic compiler output'
fi
[ ! -e "$nondeterministic_output" ] ||
  lbf_fail 'nondeterministic failure left partial output'

sed 's/^function_symbol=lunaflux_test_embedding$/function_symbol=lunaflux_missing_symbol/' \
  "$test_root/candidates/recipes/embedding.recipe" >"$test_root/recipe.substituted"
cp "$test_root/recipe.substituted" \
  "$test_root/candidates/recipes/embedding.recipe"
write_candidate_inventory "$inventory"
inventory_sha=$(lbf_sha256_file "$inventory")
symbol_output=$test_root/symbol-output
if "$repo_root/scripts/build-luna-bf16-kernel-set.sh" \
  "$test_root/toolchain/nvcc" \
  "$test_root/toolchain.manifest#sha256=$toolchain_sha" \
  "$test_root/candidates" \
  "$inventory#sha256=$inventory_sha" \
  "$symbol_output" >/dev/null 2>&1; then
  lbf_fail 'producer accepted a recipe/source symbol substitution'
fi
[ ! -e "$symbol_output" ] || lbf_fail 'symbol rejection left partial output'

printf '%s\n' 'Luna BF16 offline kernel producer gate passed'
