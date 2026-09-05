#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL
umask 077

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/luna-bf16-kernel-producer-common.sh"

[ "$#" -eq 2 ] ||
  lbf_fail 'usage: assemble-luna-kernel-root.sh ABSOLUTE_PLAN_SOURCE_ROOT#sha256=PLAN_HEX ABSOLUTE_NEW_OUTPUT'
source_argument=$1
output=$2
case "$source_argument" in /*#sha256=*) ;; *)
  lbf_fail 'plan source must be absolute and independently digest pinned'
  ;;
esac
source_root=${source_argument%#sha256=*}
expected_plan_sha=${source_argument##*#sha256=}
lbf_require_absolute_directory "$source_root"
lbf_is_sha256 "$expected_plan_sha" || lbf_fail 'plan digest is invalid'
plan=$source_root/kernel-root.plan.v1
inventory=$source_root/kernel.files.sha256
payload=$source_root/payload
lbf_require_absolute_file "$plan"
lbf_require_absolute_file "$inventory"
lbf_require_absolute_directory "$payload"
[ "$(lbf_sha256_file "$plan")" = "$expected_plan_sha" ] ||
  lbf_fail 'plan digest does not match its bytes'

case "$output" in /*) ;; *) lbf_fail 'output path must be absolute' ;; esac
case "$output" in /|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
  lbf_fail 'output path is not a safe canonical absolute path'
  ;;
esac
[ ! -e "$output" ] && [ ! -L "$output" ] ||
  lbf_fail 'refusing to overwrite an existing output'
output_parent=$(CDPATH= cd -- "$(dirname -- "$output")" && pwd -P)
[ "$output_parent/$(basename -- "$output")" = "$output" ] ||
  lbf_fail 'output parent is not canonical'

scratch=$(mktemp -d /tmp/lunaflux-kernel-root-assemble.XXXXXX) ||
  lbf_fail 'could not create private verifier scratch'
publish_container=$(mktemp -d "$output_parent/.lunaflux-kernel-root.XXXXXX") ||
  lbf_fail 'could not create atomic publication scratch'
published=0
cleanup() {
  rm -rf -- "$scratch"
  if [ "$published" -ne 1 ]; then
    chmod -R u+w "$publish_container" 2>/dev/null || true
    rm -rf -- "$publish_container"
  fi
}
trap cleanup EXIT HUP INT TERM

lbf_require_newline "$plan" 'kernel-root plan'
[ "$(wc -l <"$plan" | tr -d ' ')" -eq 13 ] ||
  lbf_fail 'kernel-root plan must contain exactly 13 fields'
plan_value() {
  pv_line=$(sed -n "$1p" "$plan")
  case "$pv_line" in "$2="*) printf '%s\n' "${pv_line#*=}" ;; *)
    lbf_fail "kernel-root plan field $2 is not canonical"
    ;;
  esac
}
[ "$(plan_value 1 schema)" = lunaflux-bf16-kernel-root-plan.v1 ] ||
  lbf_fail 'unsupported kernel-root plan schema'
manifest_relative=$(plan_value 2 execution_manifest_relative)
manifest_sha=$(plan_value 3 execution_manifest_sha256)
[ "$(plan_value 4 manifest_schema_version)" = 2 ] ||
  lbf_fail 'kernel-root manifest schema must be v2'
[ "$(plan_value 5 catalog_version)" = 3 ] ||
  lbf_fail 'kernel-root catalog must be v3'
operation_count=$(plan_value 6 operation_count)
module_count=$(plan_value 7 module_count)
entry_point_count=$(plan_value 8 entry_point_count)
total_module_bytes=$(plan_value 9 total_module_bytes)
inventory_sha=$(plan_value 10 files_inventory_sha256)
identity_sha=$(plan_value 11 kernel_root_identity_sha256)
[ "$(plan_value 12 exact_inventory)" = 1 ] || lbf_fail 'exact inventory is required'
[ "$(plan_value 13 compiler_jit_free)" = 1 ] || lbf_fail 'compiler/JIT-free root is required'
[ "$manifest_relative" = lunaflux.execution.json ] ||
  lbf_fail 'execution manifest must use the fixed release locator'
for digest in "$manifest_sha" "$inventory_sha" "$identity_sha"; do
  lbf_is_sha256 "$digest" || lbf_fail 'kernel-root plan digest is invalid'
done
for count in "$operation_count" "$module_count" "$entry_point_count" "$total_module_bytes"; do
  lbf_is_uint "$count" || lbf_fail 'kernel-root plan count is invalid'
done
[ "$operation_count" -ge 9 ] && [ "$module_count" -ge 1 ] &&
  [ "$entry_point_count" -ge 9 ] || lbf_fail 'kernel-root plan is incomplete'
[ "$(lbf_sha256_file "$inventory")" = "$inventory_sha" ] ||
  lbf_fail 'kernel inventory digest mismatch'

lbf_require_newline "$inventory" 'kernel inventory'
declared=$scratch/declared
actual=$scratch/actual
lbf_inventory_paths "$inventory" >"$declared"
LC_ALL=C sort "$declared" >"$scratch/sorted"
cmp -s "$declared" "$scratch/sorted" || lbf_fail 'kernel inventory is not sorted'
[ -z "$(uniq -d "$declared" | sed -n '1p')" ] ||
  lbf_fail 'kernel inventory contains duplicate paths'
find "$payload" -type f -print | sed "s#^$payload/##" | LC_ALL=C sort >"$actual"
cmp -s "$actual" "$declared" || lbf_fail 'kernel inventory is not the exact payload set'
if find "$source_root" -type l -print | grep -q .; then
  lbf_fail 'kernel-root plan source contains a symbolic link'
fi
if find "$source_root" ! -type d ! -type f -print | grep -q .; then
  lbf_fail 'kernel-root plan source contains a special filesystem object'
fi
source_top=$(find "$source_root" -mindepth 1 -maxdepth 1 -print |
  sed "s#^$source_root/##" | LC_ALL=C sort)
[ "$source_top" = "kernel-root.plan.v1
kernel.files.sha256
payload" ] || lbf_fail 'kernel-root plan source contains ambient top-level entries'

observed_modules=0
observed_module_bytes=0
observed_json=0
observed_fused_runtime=0
while IFS= read -r line; do
  lbf_validate_inventory_line "$line"
  digest=${line%%  *}
  relative=${line#*  }
  file=$payload/$relative
  [ -f "$file" ] && [ ! -L "$file" ] ||
    lbf_fail "kernel payload is not a regular file: $relative"
  [ "$(lbf_sha256_file "$file")" = "$digest" ] ||
    lbf_fail "kernel payload digest mismatch: $relative"
  case "$relative" in
    lunaflux.execution.json)
      observed_json=$((observed_json + 1))
      [ "$digest" = "$manifest_sha" ] || lbf_fail 'execution manifest digest mismatch'
      ;;
    sha256/*.cubin)
      artifact=${relative#sha256/}
      artifact=${artifact%.cubin}
      [ "$artifact" = "$digest" ] || lbf_fail 'CUBIN path is not content addressed'
      observed_modules=$((observed_modules + 1))
      size=$(wc -c <"$file" | tr -d ' ')
      observed_module_bytes=$((observed_module_bytes + size))
      ;;
    fused-production.runtime.v1)
      observed_fused_runtime=$((observed_fused_runtime + 1))
      [ "$(sed -n '1p' "$file")" = \
        'schema=lunaflux-fused-production-runtime-source.v1' ] ||
        lbf_fail 'fused production runtime source schema is invalid'
      ;;
    reusable-fused-residual.runtime.v1)
      observed_fused_runtime=$((observed_fused_runtime + 1))
      [ "$(sed -n '1p' "$file")" = \
        'schema=lunaflux-reusable-fused-residual-rmsnorm-runtime.v1' ] ||
        lbf_fail 'reusable fused residual runtime schema is invalid'
      ;;
    reusable-fused-runtime-bundle.v3)
      observed_fused_runtime=$((observed_fused_runtime + 1))
      case "$(sed -n '1p' "$file")" in
        schema=lunaflux-reusable-fused-runtime-bundle.v3|schema=lunaflux-reusable-fused-runtime-bundle.v4) ;;
        *) lbf_fail 'reusable fused runtime bundle schema is invalid' ;;
      esac
      ;;
    *) lbf_fail "unsupported kernel-root payload: $relative" ;;
  esac
done <"$inventory"
[ "$observed_json" -eq 1 ] && [ "$observed_modules" -eq "$module_count" ] ||
  lbf_fail 'kernel-root manifest/module counts do not match exact payload'
[ "$observed_module_bytes" -eq "$total_module_bytes" ] ||
  lbf_fail 'kernel-root module byte total mismatch'
[ "$observed_fused_runtime" -le 1 ] ||
  lbf_fail 'kernel-root contains multiple fused runtime sources'
if grep -E -i 'nvrtc|runtime[_-]?jit|developer[_-]?jit|\.ptx|source[_-]?(path|code)' \
  "$payload/$manifest_relative" >/dev/null 2>&1; then
  lbf_fail 'execution manifest contains compiler, PTX, JIT, or source authority'
fi

{
  printf '%s\n' 'schema=lunaflux-bf16-kernel-root-identity.v1'
  printf 'execution_manifest_sha256=%s\n' "$manifest_sha"
  printf 'files_inventory_sha256=%s\n' "$inventory_sha"
} >"$scratch/identity"
[ "$(lbf_sha256_file "$scratch/identity")" = "$identity_sha" ] ||
  lbf_fail 'kernel-root identity digest mismatch'

stage=$publish_container/result
mkdir "$stage" "$stage/kernel-root"
cp -R "$payload/." "$stage/kernel-root/"
cp "$inventory" "$stage/kernel.files.sha256"
cp "$plan" "$stage/kernel-root.plan.v1"
find "$stage" -type f -exec chmod 444 {} \;
find "$stage" -type d -exec chmod 555 {} \;
chmod 755 "$stage"
if [ "${LUNA_KERNEL_ROOT_VERIFY_INNER:-0}" != 1 ]; then
  "$repo_root/scripts/verify-luna-kernel-root.sh" "$stage"
fi
mv "$stage" "$output"
rmdir "$publish_container"
published=1
trap - EXIT HUP INT TERM
rm -rf -- "$scratch"
printf '%s\n' "Luna deployment kernel root assembled: $output"
printf 'kernel_manifest_relative=%s\n' "$manifest_relative"
printf 'kernel_manifest_sha256=%s\n' "$manifest_sha"
printf 'kernel_inventory_sha256=%s\n' "$inventory_sha"
