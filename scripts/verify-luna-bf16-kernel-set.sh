#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/luna-bf16-kernel-producer-common.sh"

[ "$#" -eq 1 ] ||
  lbf_fail 'usage: verify-luna-bf16-kernel-set.sh ABSOLUTE_COMPILED_SET_ROOT'
root=$1
lbf_require_absolute_directory "$root"

scratch=$(mktemp -d /tmp/lunaflux-bf16-producer-verify.XXXXXX) ||
  lbf_fail 'could not create private verifier scratch'
trap 'rm -rf -- "$scratch"' EXIT HUP INT TERM

top_entries=$(find "$root" -mindepth 1 -maxdepth 1 -print |
  sed "s#^$root/##" | LC_ALL=C sort)
[ "$top_entries" = "compiled-set.v1
files.sha256
receipts
sha256" ] || lbf_fail 'compiled set has unexpected top-level entries'
if find "$root" -type l -print | grep -q .; then
  lbf_fail 'compiled set contains a symbolic link'
fi
if find "$root" ! -type d ! -type f -print | grep -q .; then
  lbf_fail 'compiled set contains a special filesystem object'
fi
if find "$root/sha256" -type f ! -name '*.cubin' -print | grep -q .; then
  lbf_fail 'compiled module inventory contains non-CUBIN material'
fi
if find "$root/receipts" -type f ! -name '*.receipt.v1' -print | grep -q .; then
  lbf_fail 'receipt inventory contains an unsupported file'
fi

inventory=$root/files.sha256
lbf_require_newline "$inventory" 'compiled-set inventory'
declared=$scratch/declared
actual=$scratch/actual
lbf_inventory_paths "$inventory" >"$declared"
LC_ALL=C sort "$declared" >"$scratch/sorted"
cmp -s "$declared" "$scratch/sorted" ||
  lbf_fail 'compiled-set inventory is not sorted'
[ -z "$(uniq -d "$declared" | sed -n '1p')" ] ||
  lbf_fail 'compiled-set inventory has duplicate paths'
find "$root" -type f ! -path "$inventory" -print | sed "s#^$root/##" |
  LC_ALL=C sort >"$actual"
cmp -s "$actual" "$declared" ||
  lbf_fail 'compiled-set inventory is not the exact file set'
while IFS= read -r line; do
  lbf_validate_inventory_line "$line"
  digest=${line%%  *}
  relative=${line#*  }
  [ "$(lbf_sha256_file "$root/$relative")" = "$digest" ] ||
    lbf_fail "compiled-set payload digest mismatch: $relative"
done <"$inventory"

record=$root/compiled-set.v1
lbf_require_newline "$record" 'compiled-set record'
[ "$(sed -n '1p' "$record")" = 'schema=lunaflux-bf16-compiled-kernel-set.v1' ] ||
  lbf_fail 'unsupported compiled-set schema'
record_value() {
  rv_line=$(sed -n "$1p" "$record")
  case "$rv_line" in "$2="*) printf '%s\n' "${rv_line#*=}" ;; *)
    lbf_fail "compiled-set field $2 is not canonical"
    ;;
  esac
}
candidate_set_sha=$(record_value 2 candidate_set_sha256)
candidate_inventory_sha=$(record_value 3 candidate_inventory_sha256)
toolchain_sha=$(record_value 4 toolchain_sha256)
driver_identity_sha=$(record_value 5 driver_identity_sha256)
target=$(record_value 6 target)
operation_count=$(record_value 7 operation_count)
module_count=$(record_value 8 module_count)
for digest in "$candidate_set_sha" "$candidate_inventory_sha" "$toolchain_sha" "$driver_identity_sha"; do
  lbf_is_sha256 "$digest" || lbf_fail 'compiled-set identity digest is invalid'
done
lbf_is_target "$target" || lbf_fail 'compiled-set target is invalid'
lbf_is_uint "$operation_count" && [ "$operation_count" -ge 9 ] ||
  lbf_fail 'compiled-set operation count is invalid'
lbf_is_uint "$module_count" && [ "$module_count" -ge 1 ] ||
  lbf_fail 'compiled-set module count is invalid'
[ "$(wc -l <"$record" | tr -d ' ')" -eq "$((operation_count + 8))" ] ||
  lbf_fail 'compiled-set line count does not match operation_count'
actual_module_count=$(find "$root/sha256" -type f -name '*.cubin' -print |
  wc -l | tr -d ' ')
[ "$actual_module_count" -eq "$module_count" ] ||
  lbf_fail 'compiled-set module count does not match its exact inventory'

operations=$scratch/operations
: >"$operations"
operation_id=0
while [ "$operation_id" -lt "$operation_count" ]; do
  line_number=$((operation_id + 9))
  line=$(sed -n "${line_number}p" "$record")
  case "$line" in operation=*) value=${line#operation=} ;; *)
    lbf_fail "compiled-set operation line $line_number is not canonical"
    ;;
  esac
  old_ifs=$IFS
  IFS=,
  set -- $value
  IFS=$old_ifs
  [ "$#" -eq 5 ] || lbf_fail 'compiled-set operation row has the wrong field count'
  [ "$1" = "$operation_id" ] ||
    lbf_fail 'compiled-set operations are not contiguous and ordered'
  key=$2
  family=$3
  artifact_sha=$4
  receipt_sha=$5
  lbf_is_key "$key" || lbf_fail "compiled-set candidate key is invalid: $key"
  lbf_is_family "$family" || lbf_fail "compiled-set family is invalid: $family"
  lbf_is_sha256 "$artifact_sha" || lbf_fail 'compiled-set artifact digest is invalid'
  lbf_is_sha256 "$receipt_sha" || lbf_fail 'compiled-set receipt digest is invalid'
  if awk -F, -v key="$key" '$1 == key { found=1 } END { exit !found }' "$operations"; then
    lbf_fail "compiled-set candidate key is duplicated: $key"
  fi
  printf '%s,%s,%s,%s,%s\n' "$key" "$operation_id" "$family" "$artifact_sha" "$receipt_sha" >>"$operations"

  module=$root/sha256/$artifact_sha.cubin
  receipt=$root/receipts/$key.receipt.v1
  [ -f "$module" ] && [ ! -L "$module" ] ||
    lbf_fail "compiled-set module is missing: $key"
  [ "$(lbf_sha256_file "$module")" = "$artifact_sha" ] ||
    lbf_fail "compiled-set module digest mismatch: $key"
  [ -f "$receipt" ] && [ ! -L "$receipt" ] ||
    lbf_fail "compiled-set receipt is missing: $key"
  [ "$(lbf_sha256_file "$receipt")" = "$receipt_sha" ] ||
    lbf_fail "compiled-set receipt digest mismatch: $key"
  lbf_require_newline "$receipt" "compile receipt $key"
  [ "$(wc -l <"$receipt" | tr -d ' ')" -eq 13 ] ||
    lbf_fail "compile receipt has the wrong field count: $key"
  receipt_value() {
    rr_line=$(sed -n "$1p" "$receipt")
    case "$rr_line" in "$2="*) printf '%s\n' "${rr_line#*=}" ;; *)
      lbf_fail "compile receipt field $2 is not canonical: $key"
      ;;
    esac
  }
  [ "$(receipt_value 1 schema)" = lunaflux-bf16-offline-compile-receipt.v1 ] ||
    lbf_fail "compile receipt schema is unsupported: $key"
  [ "$(receipt_value 2 candidate_set_sha256)" = "$candidate_set_sha" ] ||
    lbf_fail "compile receipt candidate-set mismatch: $key"
  [ "$(receipt_value 3 operation_id)" = "$operation_id" ] ||
    lbf_fail "compile receipt operation mismatch: $key"
  [ "$(receipt_value 4 family)" = "$family" ] ||
    lbf_fail "compile receipt family mismatch: $key"
  receipt_source_sha=$(receipt_value 5 source_sha256)
  receipt_recipe_sha=$(receipt_value 6 recipe_sha256)
  lbf_is_sha256 "$receipt_source_sha" || lbf_fail "receipt source digest is invalid: $key"
  lbf_is_sha256 "$receipt_recipe_sha" || lbf_fail "receipt recipe digest is invalid: $key"
  [ "$(receipt_value 7 toolchain_sha256)" = "$toolchain_sha" ] ||
    lbf_fail "compile receipt toolchain mismatch: $key"
  [ "$(receipt_value 8 driver_identity_sha256)" = "$driver_identity_sha" ] ||
    lbf_fail "compile receipt driver mismatch: $key"
  [ "$(receipt_value 9 target)" = "$target" ] ||
    lbf_fail "compile receipt target mismatch: $key"
  [ "$(receipt_value 10 first_artifact_sha256)" = "$artifact_sha" ] ||
    lbf_fail "compile receipt first-build mismatch: $key"
  [ "$(receipt_value 11 second_artifact_sha256)" = "$artifact_sha" ] ||
    lbf_fail "compile receipt second-build mismatch: $key"
  [ "$(receipt_value 12 module_relative_path)" = "sha256/$artifact_sha.cubin" ] ||
    lbf_fail "compile receipt module path mismatch: $key"
  [ "$(receipt_value 13 deterministic)" = 1 ] ||
    lbf_fail "compile receipt does not prove deterministic output: $key"
  operation_id=$((operation_id + 1))
done

for required_family in embedding_lookup rms_norm positioned_rotary residual_add \
  qkv_projection dense_projection gated_mlp language_model_head paged_attention; do
  awk -F, -v family="$required_family" '$3 == family { found=1 } END { exit !found }' "$operations" ||
    lbf_fail "compiled set is missing BF16 family: $required_family"
done

if grep -E -i 'nvrtc|runtime[_-]?jit|developer[_-]?jit|\.ptx' "$record" "$root"/receipts/* \
  >/dev/null 2>&1; then
  lbf_fail 'compiled-set control evidence contains PTX or JIT vocabulary'
fi
printf '%s\n' 'Luna BF16 compiled kernel-set verification passed'
