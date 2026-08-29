#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL
umask 077

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/release-bundle-common.sh"
BUNDLE_SCRATCH_DIR=$(mktemp -d /tmp/lunaflux-bundle-assemble.XXXXXX) ||
  bundle_fail 'could not create private assembly scratch'
export BUNDLE_SCRATCH_DIR
trap 'rm -rf "$BUNDLE_SCRATCH_DIR"' EXIT HUP INT TERM

usage() {
  printf '%s\n' \
    'usage: assemble-release-bundle.sh ABSOLUTE_INPUT#sha256=HEX ABSOLUTE_NEW_OUTPUT' >&2
  exit 2
}

[ "$#" -eq 2 ] || usage
input_argument=$1
output=$2

case "$input_argument" in /*#sha256=*) ;; *) usage ;; esac
input=${input_argument%#sha256=*}
input_digest=${input_argument##*#sha256=}
case "$input" in *#sha256=*) usage ;; esac
bundle_require_canonical_absolute_file "$input"
bundle_require_digest "$input" "$input_digest" 'assembly input'
bundle_require_newline_terminated "$input" 'assembly input'

case "$output" in /*) ;; *) bundle_fail 'output path must be absolute' ;; esac
case "$output" in /|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
  bundle_fail 'output path is not a safe canonical absolute path'
  ;;
esac
[ ! -e "$output" ] || bundle_fail 'refusing to overwrite an existing output'
output_parent=$(CDPATH= cd -- "$(dirname -- "$output")" && pwd -P)
[ "$output_parent/$(basename -- "$output")" = "$output" ] ||
  bundle_fail 'output parent is not canonical'

[ "$(wc -l < "$input" | tr -d ' ')" -eq 27 ] ||
  bundle_fail 'assembly input must contain exactly 27 newline-terminated lines'
[ -z "$(sed -n '28p' "$input")" ] || bundle_fail 'assembly input has trailing fields'
if grep -q "$(printf '\r')" "$input"; then
  bundle_fail 'assembly input contains carriage returns'
fi

spec_value() {
  line=$(sed -n "$1p" "$input")
  case "$line" in "$2="*) printf '%s\n' "${line#*=}" ;; *) bundle_fail "noncanonical input field $2" ;; esac
}

[ "$(spec_value 1 schema)" = 'lunaflux.deployment-assembly.v1' ] ||
  bundle_fail 'unsupported assembly schema'
base_image=$(spec_value 2 base_image)
case "$base_image" in *@sha256:*) ;; *) bundle_fail 'base image is not digest pinned' ;; esac
base_digest=${base_image##*@sha256:}
bundle_is_lower_sha256 "$base_digest" || bundle_fail 'base image digest is invalid'
architecture=$(spec_value 3 linux_architecture)
lunaflux_source=$(spec_value 4 lunaflux_source)
lunaflux_sha=$(spec_value 5 lunaflux_sha256)
worker_source=$(spec_value 6 worker_source)
worker_sha=$(spec_value 7 worker_sha256)
launch_source=$(spec_value 8 launch_source)
launch_sha=$(spec_value 9 launch_sha256)
model_root=$(spec_value 10 model_source_root)
model_inventory=$(spec_value 11 model_inventory_source)
model_inventory_sha=$(spec_value 12 model_inventory_sha256)
descriptor_relative=$(spec_value 13 runtime_descriptor_relative)
descriptor_sha=$(spec_value 14 runtime_descriptor_sha256)
policy_root=$(spec_value 15 policy_source_root)
policy_inventory=$(spec_value 16 policy_inventory_source)
policy_inventory_sha=$(spec_value 17 policy_inventory_sha256)
policy_relative=$(spec_value 18 instance_policy_relative)
policy_sha=$(spec_value 19 instance_policy_sha256)
kernel_root=$(spec_value 20 kernel_source_root)
kernel_inventory=$(spec_value 21 kernel_inventory_source)
kernel_inventory_sha=$(spec_value 22 kernel_inventory_sha256)
kernel_manifest_relative=$(spec_value 23 kernel_manifest_relative)
kernel_manifest_sha=$(spec_value 24 kernel_manifest_sha256)
library_root=$(spec_value 25 runtime_library_source_root)
library_inventory=$(spec_value 26 runtime_library_inventory_source)
library_inventory_sha=$(spec_value 27 runtime_library_inventory_sha256)

case "$architecture" in x86_64|aarch64) ;; *) bundle_fail 'unsupported Linux architecture' ;; esac
for file in "$lunaflux_source" "$worker_source" "$launch_source" \
  "$model_inventory" "$policy_inventory" "$kernel_inventory" \
  "$library_inventory"; do
  bundle_require_canonical_absolute_file "$file"
done
for root in "$model_root" "$policy_root" "$kernel_root"; do
  bundle_require_canonical_absolute_directory "$root"
done
bundle_require_digest "$lunaflux_source" "$lunaflux_sha" 'lunaflux executable'
bundle_require_digest "$worker_source" "$worker_sha" 'worker executable'
bundle_require_digest "$launch_source" "$launch_sha" 'launch file'
bundle_require_digest "$model_inventory" "$model_inventory_sha" 'model inventory'
bundle_require_digest "$policy_inventory" "$policy_inventory_sha" 'policy inventory'
bundle_require_digest "$kernel_inventory" "$kernel_inventory_sha" 'kernel inventory'
bundle_require_digest "$library_inventory" "$library_inventory_sha" 'library inventory'

bundle_is_strict_relative "$descriptor_relative" || bundle_fail 'invalid runtime descriptor locator'
bundle_is_strict_relative "$policy_relative" || bundle_fail 'invalid instance policy locator'
bundle_is_strict_relative "$kernel_manifest_relative" || bundle_fail 'invalid kernel manifest locator'
case "$descriptor_relative:$policy_relative:$kernel_manifest_relative" in
  *.json:*.json:*.json) ;;
  *) bundle_fail 'descriptor, policy, and kernel manifest locators must be JSON' ;;
esac
bundle_is_lower_sha256 "$descriptor_sha" || bundle_fail 'runtime descriptor digest is invalid'
bundle_is_lower_sha256 "$policy_sha" || bundle_fail 'instance policy digest is invalid'
bundle_is_lower_sha256 "$kernel_manifest_sha" || bundle_fail 'kernel manifest digest is invalid'

bundle_validate_inventory "$model_inventory" "$model_root" model
bundle_validate_inventory "$policy_inventory" "$policy_root" policy
bundle_validate_inventory "$kernel_inventory" "$kernel_root" kernel
bundle_require_digest "$model_root/$descriptor_relative" "$descriptor_sha" 'runtime descriptor'
bundle_require_digest "$policy_root/$policy_relative" "$policy_sha" 'instance policy'
bundle_require_digest "$kernel_root/$kernel_manifest_relative" "$kernel_manifest_sha" 'kernel manifest'

module_count=$(bundle_inventory_paths "$kernel_inventory" |
  sed -n '/\.cubin$/p;/\.fatbin$/p;/\.bin$/p' | wc -l | tr -d ' ')
[ "$module_count" -gt 0 ] || bundle_fail 'kernel inventory has no AOT module'
json_count=$(bundle_inventory_paths "$kernel_inventory" | sed -n '/\.json$/p' | wc -l | tr -d ' ')
[ "$json_count" -eq 1 ] || bundle_fail 'kernel inventory must contain exactly one JSON manifest'

if [ "$library_root" = none ]; then
  [ "$(sed -n '1p' "$library_inventory")" = none ] &&
    [ "$(wc -l < "$library_inventory" | tr -d ' ')" -eq 1 ] ||
    bundle_fail 'none library inventory must contain exactly literal none'
else
  bundle_require_canonical_absolute_directory "$library_root"
  bundle_validate_inventory "$library_inventory" "$library_root" library
fi

control_jsons=$(find "$model_root" "$policy_root" "$kernel_root" -type f -name '*.json' -print)
if grep -E -i 'nvrtc|runtime[_-]?jit|developer[_-]?jit|\.ptx|source[_-]?(path|code)' \
  "$launch_source" $control_jsons \
  >/dev/null 2>&1; then
  bundle_fail 'authenticated control material contains compiler or JIT vocabulary'
fi

mkdir "$output" || bundle_fail 'could not claim new output path'
complete=0
printf '%s\n' 'lunaflux-deployment-assembly-claim-v1' > "$output/.assembly-claim"
cleanup() {
  rm -rf "$BUNDLE_SCRATCH_DIR"
  if [ "$complete" -ne 1 ] && [ -f "$output/.assembly-claim" ] &&
    [ "$(sed -n '1p' "$output/.assembly-claim")" = 'lunaflux-deployment-assembly-claim-v1' ]; then
    chmod -R u+w "$output" 2>/dev/null || true
    rm -rf "$output"
  fi
}
trap cleanup EXIT HUP INT TERM

stage=$output/.stage
mkdir "$stage"
mkdir -p "$stage/launch-root" "$stage/model-root" "$stage/policy-root" \
  "$stage/oci-context/rootfs/opt/lunaflux/bin" \
  "$stage/oci-context/rootfs/opt/lunaflux/lib" \
  "$stage/oci-context/rootfs/opt/lunaflux/kernels" \
  "$stage/oci-context/rootfs/usr/share/licenses/lunaflux" \
  "$stage/oci-context/rootfs/var/lib/lunaflux/model" \
  "$stage/oci-context/metadata" "$stage/evidence"

cp "$launch_source" "$stage/launch-root/lunaflux.launch.json"
chmod 444 "$stage/launch-root/lunaflux.launch.json"
bundle_copy_inventory "$model_inventory" "$model_root" "$stage/model-root"
bundle_copy_inventory "$policy_inventory" "$policy_root" "$stage/policy-root"
bundle_copy_inventory "$kernel_inventory" "$kernel_root" \
  "$stage/oci-context/rootfs/opt/lunaflux/kernels"
cp "$lunaflux_source" "$stage/oci-context/rootfs/opt/lunaflux/bin/lunaflux"
cp "$worker_source" "$stage/oci-context/rootfs/opt/lunaflux/bin/lunaflux-device-worker"
chmod 555 "$stage/oci-context/rootfs/opt/lunaflux/bin/lunaflux" \
  "$stage/oci-context/rootfs/opt/lunaflux/bin/lunaflux-device-worker"
cp "$repo_root/LICENSE" \
  "$stage/oci-context/rootfs/usr/share/licenses/lunaflux/LICENSE"
chmod 444 "$stage/oci-context/rootfs/usr/share/licenses/lunaflux/LICENSE"
if [ "$library_root" != none ]; then
  bundle_copy_inventory "$library_inventory" "$library_root" \
    "$stage/oci-context/rootfs/opt/lunaflux/lib"
fi
printf '%s\n' 'external-read-only-model-root-required' \
  > "$stage/oci-context/rootfs/var/lib/lunaflux/model/.mount-contract"
chmod 444 "$stage/oci-context/rootfs/var/lib/lunaflux/model/.mount-contract"

cp "$model_inventory" "$stage/evidence/model.files.sha256"
cp "$policy_inventory" "$stage/evidence/policy.files.sha256"
cp "$kernel_inventory" "$stage/evidence/kernel.files.sha256"
cp "$library_inventory" "$stage/evidence/runtime-libraries.files.sha256"
for evidence_file in "$stage/evidence/"*.sha256; do chmod 444 "$evidence_file"; done

printf '%s\n' "$base_image" > "$stage/oci-context/metadata/base-image.ref"
printf '%s\n' "$architecture" > "$stage/oci-context/metadata/linux-architecture"
printf '%s\n' "$kernel_manifest_relative" \
  > "$stage/oci-context/metadata/kernel-manifest.relative"
printf '%s\n' "$kernel_manifest_sha" \
  > "$stage/oci-context/metadata/kernel-manifest.sha256"
if [ "$library_root" = none ]; then
  printf '%s\n' none > "$stage/oci-context/metadata/runtime-libraries.list"
else
  library_list=
  while IFS= read -r relative; do
    item=opt/lunaflux/lib/$relative
    if [ -z "$library_list" ]; then library_list=$item; else library_list=$library_list,$item; fi
  done <<EOF
$(bundle_inventory_paths "$library_inventory")
EOF
  printf '%s\n' "$library_list" > "$stage/oci-context/metadata/runtime-libraries.list"
fi
"$repo_root/scripts/assemble-oci-license-inventory.sh" \
  "$stage/oci-context/metadata/license-inventory.json"
: > "$stage/oci-context/metadata/artifacts.sha256"
bundle_write_file_inventory "$stage/oci-context/rootfs" \
  "$stage/oci-context/metadata/artifacts.sha256"
for metadata_file in "$stage/oci-context/metadata/"*; do chmod 444 "$metadata_file"; done
rootfs_inventory_sha=$(bundle_sha256_file "$stage/oci-context/metadata/artifacts.sha256")

printf '%s  %s\n' "$(bundle_sha256_file "$repo_root/scripts/assemble-release-bundle.sh")" \
  scripts/assemble-release-bundle.sh > "$stage/evidence/assembler.files.sha256"
printf '%s  %s\n' "$(bundle_sha256_file "$repo_root/scripts/assemble-oci-license-inventory.sh")" \
  scripts/assemble-oci-license-inventory.sh >> "$stage/evidence/assembler.files.sha256"
printf '%s  %s\n' "$(bundle_sha256_file "$repo_root/scripts/release-bundle-common.sh")" \
  scripts/release-bundle-common.sh >> "$stage/evidence/assembler.files.sha256"
printf '%s  %s\n' "$(bundle_sha256_file "$repo_root/scripts/verify-release-bundle.sh")" \
  scripts/verify-release-bundle.sh >> "$stage/evidence/assembler.files.sha256"
LC_ALL=C sort -o "$stage/evidence/assembler.files.sha256" \
  "$stage/evidence/assembler.files.sha256"
chmod 444 "$stage/evidence/assembler.files.sha256"
assembler_sha=$(bundle_sha256_file "$stage/evidence/assembler.files.sha256")

cat > "$stage/evidence/deployment-bundle.v1" <<EOF
schema=lunaflux.deployment-bundle.v1
base_image=$base_image
linux_architecture=$architecture
launch_file_sha256=$launch_sha
runtime_descriptor_relative=$descriptor_relative
runtime_descriptor_sha256=$descriptor_sha
instance_policy_relative=$policy_relative
instance_policy_sha256=$policy_sha
lunaflux_executable_sha256=$lunaflux_sha
worker_executable_sha256=$worker_sha
model_inventory_sha256=$model_inventory_sha
policy_inventory_sha256=$policy_inventory_sha
kernel_manifest_relative=$kernel_manifest_relative
kernel_manifest_sha256=$kernel_manifest_sha
kernel_inventory_sha256=$kernel_inventory_sha
runtime_library_inventory_sha256=$library_inventory_sha
rootfs_inventory_sha256=$rootfs_inventory_sha
assembly_input_sha256=$input_digest
assembler_sha256=$assembler_sha
exact_inventory=1
compiler_jit_free=1
EOF
chmod 444 "$stage/evidence/deployment-bundle.v1"

bundle_write_file_inventory "$stage" "$stage/bundle.files.sha256"
chmod 444 "$stage/bundle.files.sha256"
find "$stage" -type d -exec chmod 555 {} \;

"$repo_root/scripts/verify-release-bundle.sh" "$stage"
chmod 755 "$stage"
for entry in evidence launch-root model-root oci-context policy-root; do
  chmod 755 "$stage/$entry"
done
for entry in bundle.files.sha256 evidence launch-root model-root oci-context policy-root; do
  mv "$stage/$entry" "$output/$entry"
done
rmdir "$stage"
rm "$output/.assembly-claim"
find "$output" -type d -exec chmod 555 {} \;
chmod 555 "$output"
complete=1
trap - EXIT HUP INT TERM
rm -rf "$BUNDLE_SCRATCH_DIR"
printf '%s\n' "LunaFlux deployment bundle assembled: $output"
printf '%s\n' "evidence_sha256=$(bundle_sha256_file "$output/evidence/deployment-bundle.v1")"
