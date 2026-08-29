#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
TZ=UTC
export LC_ALL TZ
umask 077

fail() {
  printf 'Qwen3 BF16 physical campaign rejected: %s\n' "$1" >&2
  exit 1
}

usage() {
  printf '%s\n' \
    "usage: $0 ABSOLUTE_NVCC ABSOLUTE_COMPUTE_SANITIZER ABSOLUTE_TOOLCHAIN_MANIFEST#sha256=HEX ABSOLUTE_QWEN_MODEL_ROOT CONFIG_SHA256 MODEL_CONTENT_SHA256 TOKENIZER_SHA256 NUMERIC_LOCATOR NUMERIC_ARTIFACT_SHA256 ROUTE_MANIFEST_SHA256 ABSOLUTE_REFERENCE_CORPUS#sha256=HEX ABSOLUTE_NEW_KERNEL_OUTPUT ABSOLUTE_NEW_EVIDENCE_OUTPUT EXPECTED_GPU_UUID EXPECTED_GPU_PCI [CYCLES]" >&2
  exit 2
}

[[ $# -eq 15 || $# -eq 16 ]] || usage
nvcc=$1
sanitizer=$2
toolchain_argument=$3
model_root=$4
config_sha=$5
model_content_sha=$6
tokenizer_sha=$7
numeric_locator=$8
artifact_sha=$9
route_sha=${10}
corpus_argument=${11}
kernel_output=${12}
evidence_output=${13}
expected_uuid=${14}
expected_pci=${15}
cycles=${16:-2}

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
. "$repo_root/scripts/immutable-evidence-directory.sh"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

is_sha256() {
  [[ $1 =~ ^[0-9a-f]{64}$ ]]
}

require_tool() {
  local tool=$1 label=$2
  [[ $tool == /* && -f $tool && ! -L $tool && -x $tool ]] ||
    fail "$label is not an absolute executable regular file"
  [[ $(realpath -- "$tool") == "$tool" ]] || fail "$label path is not canonical"
}

require_new_output() {
  local output=$1 label=$2 parent name
  [[ $output == /* ]] || fail "$label is not absolute"
  parent=${output%/*}
  name=${output##*/}
  [[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] ||
    fail "$label basename is invalid"
  [[ -d $parent && ! -L $parent && $(realpath -- "$parent") == "$parent" ]] ||
    fail "$label parent is not canonical"
  [[ $output == "$parent/$name" && ! -e $output && ! -L $output ]] ||
    fail "$label is not a new canonical path"
}

require_tool "$nvcc" NVCC
require_tool "$sanitizer" compute-sanitizer
case $toolchain_argument in /*#sha256=*) ;; *) usage ;; esac
toolchain_manifest=${toolchain_argument%#sha256=*}
toolchain_sha=${toolchain_argument##*#sha256=}
[[ -f $toolchain_manifest && ! -L $toolchain_manifest &&
   $(realpath -- "$toolchain_manifest") == "$toolchain_manifest" ]] ||
  fail 'toolchain manifest is not a canonical regular file'
is_sha256 "$toolchain_sha" || fail 'toolchain manifest digest is invalid'
[[ $(sha256_file "$toolchain_manifest") == "$toolchain_sha" ]] ||
  fail 'toolchain manifest digest mismatch'
[[ -d $model_root && ! -L $model_root && $(realpath -- "$model_root") == "$model_root" ]] ||
  fail 'Qwen model root is not canonical'
for digest in "$config_sha" "$model_content_sha" "$tokenizer_sha" "$artifact_sha" "$route_sha"; do
  is_sha256 "$digest" || fail 'a Qwen authority digest is invalid'
done
[[ $numeric_locator =~ ^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$ &&
   $numeric_locator != /* && $numeric_locator != *..* ]] ||
  fail 'numeric locator is not a bounded approved relative locator'
case $corpus_argument in /*#sha256=*) ;; *) usage ;; esac
corpus_path=${corpus_argument%#sha256=*}
corpus_sha=${corpus_argument##*#sha256=}
[[ -f $corpus_path && ! -L $corpus_path && $(realpath -- "$corpus_path") == "$corpus_path" ]] ||
  fail 'reference corpus is not a canonical regular file'
is_sha256 "$corpus_sha" || fail 'reference corpus digest is invalid'
[[ $(sha256_file "$corpus_path") == "$corpus_sha" ]] ||
  fail 'reference corpus digest mismatch'
[[ $expected_uuid =~ ^GPU-[0-9a-fA-F-]{36}$ ]] || fail 'GPU UUID is invalid'
[[ $expected_pci =~ ^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$ ]] ||
  fail 'GPU PCI identity is invalid'
[[ $cycles =~ ^[1-9][0-9]?$ ]] && (( cycles <= 16 )) ||
  fail 'cycles are outside 1..16'
require_new_output "$kernel_output" 'kernel output'
require_new_output "$evidence_output" 'evidence output'

"$nvcc" --version > /tmp/lunaflux-qwen3-nvcc-version.$$ 2>&1
grep -F 'release 13.1,' /tmp/lunaflux-qwen3-nvcc-version.$$ >/dev/null ||
  fail 'Qwen campaign requires CUDA 13.1 NVCC'
rm -f -- /tmp/lunaflux-qwen3-nvcc-version.$$
driver_report=$(mktemp /tmp/lunaflux-qwen3-driver.XXXXXX)
trap 'rm -f -- "$driver_report"' EXIT HUP INT TERM
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" >"$driver_report"
driver_identity=$(sed -n 's/^driver_identity_sha256=//p' "$driver_report")
approved_driver=$(sed -n 's/^driver_identity_sha256=//p' "$toolchain_manifest")
[[ $driver_identity == "$approved_driver" && $(grep -c '^driver_identity_sha256=' "$toolchain_manifest") == 1 ]] ||
  fail 'NVCC driver identity is not approved by the toolchain manifest'
compiler_version=$(sed -n 's/^compiler_version=//p' "$driver_report")
IFS=. read -r compiler_major compiler_minor compiler_patch <<<"$compiler_version"
[[ $compiler_major =~ ^[0-9]+$ && $compiler_minor =~ ^[0-9]+$ &&
   $compiler_patch =~ ^[0-9]+$ ]] || fail 'compiler version is not canonical'

command -v nvidia-smi >/dev/null 2>&1 || fail 'nvidia-smi is unavailable'
gpu_identity=$(nvidia-smi --id=0 --query-gpu=uuid,pci.bus_id --format=csv,noheader,nounits)
[[ $gpu_identity == "$expected_uuid, $expected_pci" ]] ||
  fail 'device 0 UUID/PCI identity differs before admission'

evidence_parent=${evidence_output%/*}
evidence_name=${evidence_output##*/}
stage=$(mktemp -d "$evidence_parent/.${evidence_name}.partial.XXXXXX")
stage=$(realpath -- "$stage")
published=0
cleanup() {
  rm -f -- "$driver_report"
  if [[ $published -ne 1 && -d ${stage:-} ]]; then
    chmod -R u+rwX "$stage" 2>/dev/null || true
    rm -rf -- "$stage"
  fi
}
trap cleanup EXIT HUP INT TERM
mkdir -p "$stage/measurements"
cp "$driver_report" "$stage/measurements/driver.v1"
cp "$toolchain_manifest" "$stage/measurements/toolchain.v1"
printf '%s\n' "$gpu_identity" >"$stage/measurements/device-identity.txt"

moon build --target native --deny-warn --warn-list +73 \
  tests/qwen3_bf16_physical \
  >"$stage/measurements/moon-build.stdout" \
  2>"$stage/measurements/moon-build.stderr"
[[ ! -s $stage/measurements/moon-build.stderr ]] || fail 'MoonBit build emitted stderr'
probe=$repo_root/_build/native/debug/build/tests/qwen3_bf16_physical/qwen3_bf16_physical.exe
[[ -x $probe && -f $probe && ! -L $probe ]] || fail 'Qwen physical probe is missing'
cp "$probe" "$stage/qwen3_bf16_physical.exe"
chmod 0555 "$stage/qwen3_bf16_physical.exe"

candidate_output=$stage/candidate-export
"$stage/qwen3_bf16_physical.exe" export "$candidate_output" "$model_root" \
  "$config_sha" "$model_content_sha" "$tokenizer_sha" "$corpus_path" "$corpus_sha" \
  "$toolchain_sha" "$compiler_major" "$compiler_minor" "$compiler_patch" \
  >"$stage/measurements/export.stdout" 2>"$stage/measurements/export.stderr"
[[ ! -s $stage/measurements/export.stderr ]] || fail 'Qwen candidate export emitted stderr'
grep -Fx 'qkv_independent_widths=passed' "$stage/measurements/export.stdout" >/dev/null
grep -Fx 'qk_rms_norm_capability=11' "$stage/measurements/export.stdout" >/dev/null
layer_count=$(sed -n 's/^layer_count=//p' "$stage/measurements/export.stdout")
[[ $layer_count =~ ^[1-9][0-9]*$ ]] || fail 'Qwen layer count is not canonical'
candidate_root=$candidate_output/candidate-root
candidate_inventory=$candidate_output/candidate.files.sha256
candidate_inventory_sha=$(sha256_file "$candidate_inventory")
[[ $(grep -c ',qkv_projection$' "$candidate_root/candidate-set.v1") == "$layer_count" &&
   $(grep -c ',qk_rms_norm$' "$candidate_root/candidate-set.v1") == "$layer_count" ]] ||
  fail 'candidate set QKV/QK RMS counts do not match the Qwen layer count'

scripts/build-luna-bf16-kernel-set.sh "$nvcc" "$toolchain_argument" \
  "$candidate_root" "$candidate_inventory#sha256=$candidate_inventory_sha" \
  "$kernel_output" \
  >"$stage/measurements/offline-build.stdout" \
  2>"$stage/measurements/offline-build.stderr"
[[ ! -s $stage/measurements/offline-build.stderr ]] || fail 'offline AOT build emitted stderr'
scripts/verify-luna-bf16-kernel-set.sh "$kernel_output" \
  >"$stage/measurements/compiled-verify.stdout" \
  2>"$stage/measurements/compiled-verify.stderr"
[[ ! -s $stage/measurements/compiled-verify.stderr ]] || fail 'compiled-set verifier emitted stderr'
compiled_set=$kernel_output/compiled-set.v1
[[ $(grep -c ',qkv_projection,' "$compiled_set") == "$layer_count" &&
   $(grep -c ',qk_rms_norm,' "$compiled_set") == "$layer_count" ]] ||
  fail 'compiled QKV/QK RMS counts do not match the Qwen layer count'
qkv_module_shas=$(sed -n 's/^operation=[0-9]*,[^,]*,qkv_projection,\([^,]*\),.*$/\1/p' "$compiled_set")
qk_module_shas=$(sed -n 's/^operation=[0-9]*,[^,]*,qk_rms_norm,\([^,]*\),.*$/\1/p' "$compiled_set")
[[ $(printf '%s\n' "$qkv_module_shas" | wc -l | tr -d ' ') == "$layer_count" &&
   $(printf '%s\n' "$qk_module_shas" | wc -l | tr -d ' ') == "$layer_count" ]] ||
  fail 'compiled Qwen family module inventory is incomplete'
[[ $(printf '%s\n' "$qkv_module_shas" | LC_ALL=C sort -u | wc -l | tr -d ' ') == 1 &&
   $(printf '%s\n' "$qk_module_shas" | LC_ALL=C sort -u | wc -l | tr -d ' ') == 1 ]] ||
  fail 'same-shape Qwen families did not collapse to reusable module digests'
qkv_module_sha=$(printf '%s\n' "$qkv_module_shas" | sed -n '1p')
qk_module_sha=$(printf '%s\n' "$qk_module_shas" | sed -n '1p')
is_sha256 "$qkv_module_sha" && is_sha256 "$qk_module_sha" ||
  fail 'compiled Qwen module identities are malformed'
qkv_cubin=$kernel_output/sha256/$qkv_module_sha.cubin
qk_cubin=$kernel_output/sha256/$qk_module_sha.cubin
[[ $(sha256_file "$qkv_cubin") == "$qkv_module_sha" &&
   $(sha256_file "$qk_cubin") == "$qk_module_sha" ]] ||
  fail 'compiled Qwen module bytes do not match their identities'

"$sanitizer" --tool memcheck --leak-check full --error-exitcode 99 \
  --log-file "$stage/measurements/lowerings-memcheck.log" \
  "$stage/qwen3_bf16_physical.exe" probe "$model_root" "$config_sha" \
  "$model_content_sha" "$toolchain_sha" "$compiler_major" "$compiler_minor" \
  "$compiler_patch" "$qkv_cubin" "$qk_cubin" "$expected_uuid" \
  "$expected_pci" "$cycles" \
  >"$stage/measurements/lowerings.stdout" \
  2>"$stage/measurements/lowerings.stderr"
[[ ! -s $stage/measurements/lowerings.stderr ]] || fail 'Qwen lowering probe emitted stderr'
grep -F 'outcome=qwen3-bf16-lowerings-pass ' "$stage/measurements/lowerings.stdout" >/dev/null
grep -F 'ERROR SUMMARY: 0 errors' "$stage/measurements/lowerings-memcheck.log" >/dev/null

"$sanitizer" --tool memcheck --leak-check full --error-exitcode 99 \
  --log-file "$stage/measurements/materialize-memcheck.log" \
  "$stage/qwen3_bf16_physical.exe" materialize "$model_root" "$config_sha" \
  "$model_content_sha" "$numeric_locator" "$artifact_sha" "$route_sha" \
  "$expected_uuid" "$expected_pci" \
  >"$stage/measurements/materialize.stdout" \
  2>"$stage/measurements/materialize.stderr"
[[ ! -s $stage/measurements/materialize.stderr ]] || fail 'Qwen materializer emitted stderr'
grep -F 'outcome=qwen3-numeric-materialization-pass ' "$stage/measurements/materialize.stdout" >/dev/null
grep -F 'resources=root0,context0,allocation0,pending0,cleanup0' \
  "$stage/measurements/materialize.stdout" >/dev/null
grep -F 'ERROR SUMMARY: 0 errors' "$stage/measurements/materialize-memcheck.log" >/dev/null

cat >"$stage/SCOPE.txt" <<'EOF'
schema=lunaflux-qwen3-bf16-physical-scope.v1
validated=qwen-tokenizer-reference-corpus
validated=qwen-bf16-candidate-completeness
validated=qwen-independent-width-qkv-cuda
validated=qwen-capability-11-qk-rms-norm-cuda
validated=qwen-canonical-numeric-allocation-copy-close
native_framed_c1_serving=separate-qwen-v12-physical-campaign-required
openai_sse_benchmark=blocked-authenticated-qwen-token-id-sse-bridge-unavailable
standard_openai_responses_profile_satisfies_benchmark_adapter=false
benchmark_capacity=c1-correctness-only
release_bind_max_batch_rows_native=1
release_bind_max_query_rows_native=1
benchmark_c32=separate-authenticated-release-profile-required
release_bind_max_batch_rows_benchmark=32
release_bind_max_query_rows_benchmark=32
engine_server_gpu_concurrency=exactly-one-required
promotion_authority=absent
EOF

lunaflux_prepare_evidence_manifest "$stage" || fail 'evidence inventory failed'
cat >"$stage/RESULT.txt" <<EOF
schema=lunaflux-qwen3-bf16-physical-result.v1
outcome=passed-lower-level
files_sha256=$lunaflux_evidence_manifest_sha256
toolchain_sha256=$toolchain_sha
driver_identity_sha256=$driver_identity
config_sha256=$config_sha
model_content_sha256=$model_content_sha
tokenizer_sha256=$tokenizer_sha
numeric_artifact_sha256=$artifact_sha
route_manifest_sha256=$route_sha
reference_corpus_sha256=$corpus_sha
qkv_module_sha256=$qkv_module_sha
qk_rms_norm_module_sha256=$qk_module_sha
layer_count=$layer_count
full_serving=runner-wired-not-executed
promotion_authority=absent
EOF
mv "$stage" "$evidence_output"
stage=
lunaflux_seal_evidence_directory "$evidence_output" \
  "$evidence_output/qwen3_bf16_physical.exe" || fail 'evidence sealing failed'
published=1
printf 'outcome=qwen3-bf16-physical-lower-level-pass evidence=%s kernels=%s full_serving=blocked\n' \
  "$evidence_output" "$kernel_output"
