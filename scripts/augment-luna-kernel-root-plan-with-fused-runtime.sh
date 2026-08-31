#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL
umask 077

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/luna-bf16-kernel-producer-common.sh"

[ "$#" -eq 3 ] || lbf_fail \
  'usage: augment-luna-kernel-root-plan-with-fused-runtime.sh ABSOLUTE_PLAN_SOURCE#sha256=HEX ABSOLUTE_FUSED_RUNTIME#sha256=HEX ABSOLUTE_NEW_OUTPUT'
source_argument=$1
runtime_argument=$2
output=$3
case "$source_argument" in /*#sha256=*) ;; *) lbf_fail 'source plan is not pinned' ;; esac
case "$runtime_argument" in /*#sha256=*) ;; *) lbf_fail 'fused runtime is not pinned' ;; esac
source_root=${source_argument%#sha256=*}
source_sha=${source_argument##*#sha256=}
runtime=${runtime_argument%#sha256=*}
runtime_sha=${runtime_argument##*#sha256=}
lbf_require_absolute_directory "$source_root"
lbf_require_absolute_file "$runtime"
lbf_is_sha256 "$source_sha" || lbf_fail 'source plan digest is invalid'
lbf_is_sha256 "$runtime_sha" || lbf_fail 'fused runtime digest is invalid'
[ "$(lbf_sha256_file "$source_root/kernel-root.plan.v1")" = "$source_sha" ] ||
  lbf_fail 'source plan digest mismatch'
[ "$(lbf_sha256_file "$runtime")" = "$runtime_sha" ] ||
  lbf_fail 'fused runtime digest mismatch'
case "$(sed -n '1p' "$runtime")" in
  schema=lunaflux-fused-production-runtime-source.v1)
    runtime_relative=fused-production.runtime.v1
    ;;
  schema=lunaflux-reusable-fused-residual-rmsnorm-runtime.v1)
    runtime_relative=reusable-fused-residual.runtime.v1
    ;;
  *) lbf_fail 'fused runtime schema is invalid' ;;
esac

case "$output" in /*) ;; *) lbf_fail 'output path must be absolute' ;; esac
case "$output" in /|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
  lbf_fail 'output path is not a safe canonical absolute path'
  ;;
esac
[ ! -e "$output" ] && [ ! -L "$output" ] || lbf_fail 'refusing to overwrite output'
output_parent=$(CDPATH= cd -- "$(dirname -- "$output")" && pwd -P)
[ "$output_parent/$(basename -- "$output")" = "$output" ] ||
  lbf_fail 'output parent is not canonical'

scratch=$(realpath -- "$(mktemp -d /tmp/lunaflux-fused-kernel-plan.XXXXXX)") ||
  lbf_fail 'could not create scratch directory'
cleanup() {
  chmod -R u+w "$scratch" 2>/dev/null || true
  rm -rf -- "$scratch"
}
trap cleanup EXIT HUP INT TERM
"$repo_root/scripts/assemble-luna-kernel-root.sh" "$source_argument" \
  "$scratch/original" >/dev/null

stage=$scratch/stage
mkdir "$stage" "$stage/payload"
cp -R "$source_root/payload/." "$stage/payload/"
cp "$runtime" "$stage/payload/$runtime_relative"
find "$stage/payload" -type f -print | sed "s#^$stage/payload/##" | LC_ALL=C sort |
  while IFS= read -r relative; do
    printf '%s  %s\n' "$(lbf_sha256_file "$stage/payload/$relative")" "$relative"
  done >"$stage/kernel.files.sha256"
inventory_sha=$(lbf_sha256_file "$stage/kernel.files.sha256")
plan=$source_root/kernel-root.plan.v1
field() {
  line=$(sed -n "$1p" "$plan")
  case "$line" in "$2="*) printf '%s\n' "${line#*=}" ;; *)
    lbf_fail "source plan field $2 is invalid" ;;
  esac
}
manifest_sha=$(field 3 execution_manifest_sha256)
{
  printf '%s\n' 'schema=lunaflux-bf16-kernel-root-identity.v1'
  printf 'execution_manifest_sha256=%s\n' "$manifest_sha"
  printf 'files_inventory_sha256=%s\n' "$inventory_sha"
} >"$scratch/identity"
identity_sha=$(lbf_sha256_file "$scratch/identity")
{
  sed -n '1,9p' "$plan"
  printf 'files_inventory_sha256=%s\n' "$inventory_sha"
  printf 'kernel_root_identity_sha256=%s\n' "$identity_sha"
  sed -n '12,13p' "$plan"
} >"$stage/kernel-root.plan.v1"
plan_sha=$(lbf_sha256_file "$stage/kernel-root.plan.v1")
mv "$stage" "$output"
trap - EXIT HUP INT TERM
chmod -R u+w "$scratch" 2>/dev/null || true
rm -rf -- "$scratch"
printf '%s\n' 'schema=lunaflux-fused-kernel-root-augmentation.v1'
printf 'parent_plan_sha256=%s\n' "$source_sha"
printf 'fused_runtime_sha256=%s\n' "$runtime_sha"
printf 'kernel_inventory_sha256=%s\n' "$inventory_sha"
printf 'kernel_plan_sha256=%s\n' "$plan_sha"
printf '%s\n' 'qualification_only=true'
