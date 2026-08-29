#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/release-bundle-common.sh"
BUNDLE_SCRATCH_DIR=$(mktemp -d /tmp/lunaflux-bundle-verify.XXXXXX) ||
  bundle_fail 'could not create private verifier scratch'
export BUNDLE_SCRATCH_DIR
trap 'rm -rf "$BUNDLE_SCRATCH_DIR"' EXIT HUP INT TERM

[ "$#" -eq 1 ] || {
  printf '%s\n' 'usage: verify-release-bundle.sh ABSOLUTE_BUNDLE_ROOT' >&2
  exit 2
}
bundle=$1
bundle_require_canonical_absolute_directory "$bundle"

top_entries=$(find "$bundle" -mindepth 1 -maxdepth 1 -print |
  sed "s#^$bundle/##" | LC_ALL=C sort)
[ "$top_entries" = "bundle.files.sha256
evidence
launch-root
model-root
oci-context
policy-root" ] || bundle_fail 'bundle has unexpected top-level entries'
if find "$bundle" -type l -print | grep -q .; then
  bundle_fail 'bundle contains a symbolic link'
fi
if find "$bundle" ! -type d ! -type f -print | grep -q .; then
  bundle_fail 'bundle contains a special filesystem object'
fi
for directory in $(find "$bundle" -type d -print); do bundle_require_mode "$directory" 555; done
for file in $(find "$bundle" -type f -print); do
  case "${file#"$bundle"/}" in
    oci-context/rootfs/opt/lunaflux/bin/lunaflux|oci-context/rootfs/opt/lunaflux/bin/lunaflux-device-worker)
      bundle_require_mode "$file" 555
      ;;
    *) bundle_require_mode "$file" 444 ;;
  esac
done

manifest=$bundle/bundle.files.sha256
[ -s "$manifest" ] || bundle_fail 'bundle.files.sha256 is empty'
bundle_require_newline_terminated "$manifest" 'bundle payload inventory'
actual=$BUNDLE_SCRATCH_DIR/bundle.actual
declared=$BUNDLE_SCRATCH_DIR/bundle.declared
find "$bundle" -type f ! -path "$manifest" -print | sed "s#^$bundle/##" |
  LC_ALL=C sort > "$actual"
bundle_inventory_paths "$manifest" > "$declared"
cmp -s "$actual" "$declared" || {
  rm -f "$actual" "$declared"
  bundle_fail 'bundle.files.sha256 is not the exact bundle payload inventory'
}
rm -f "$actual" "$declared"
while IFS= read -r line; do
  bundle_validate_inventory_line "$line"
  digest=${line%%  *}
  relative=${line#*  }
  [ "$(bundle_sha256_file "$bundle/$relative")" = "$digest" ] ||
    bundle_fail "bundle payload digest mismatch: $relative"
done < "$manifest"

evidence=$bundle/evidence/deployment-bundle.v1
[ "$(wc -l < "$evidence" | tr -d ' ')" -eq 21 ] ||
  bundle_fail 'deployment evidence must contain exactly 21 lines'
bundle_require_newline_terminated "$evidence" 'deployment evidence'
[ -z "$(sed -n '22p' "$evidence")" ] || bundle_fail 'deployment evidence has trailing fields'
[ "$(bundle_evidence_value "$evidence" 1 schema)" = 'lunaflux.deployment-bundle.v1' ] ||
  bundle_fail 'unsupported deployment evidence schema'
base_image=$(bundle_evidence_value "$evidence" 2 base_image)
case "$base_image" in *@sha256:*) ;; *) bundle_fail 'base image evidence is not digest pinned' ;; esac
base_digest=${base_image##*@sha256:}
bundle_is_lower_sha256 "$base_digest" || bundle_fail 'base image evidence digest is invalid'
architecture=$(bundle_evidence_value "$evidence" 3 linux_architecture)
launch_sha=$(bundle_evidence_value "$evidence" 4 launch_file_sha256)
descriptor_relative=$(bundle_evidence_value "$evidence" 5 runtime_descriptor_relative)
descriptor_sha=$(bundle_evidence_value "$evidence" 6 runtime_descriptor_sha256)
policy_relative=$(bundle_evidence_value "$evidence" 7 instance_policy_relative)
policy_sha=$(bundle_evidence_value "$evidence" 8 instance_policy_sha256)
lunaflux_sha=$(bundle_evidence_value "$evidence" 9 lunaflux_executable_sha256)
worker_sha=$(bundle_evidence_value "$evidence" 10 worker_executable_sha256)
model_inventory_sha=$(bundle_evidence_value "$evidence" 11 model_inventory_sha256)
policy_inventory_sha=$(bundle_evidence_value "$evidence" 12 policy_inventory_sha256)
kernel_manifest_relative=$(bundle_evidence_value "$evidence" 13 kernel_manifest_relative)
kernel_manifest_sha=$(bundle_evidence_value "$evidence" 14 kernel_manifest_sha256)
kernel_inventory_sha=$(bundle_evidence_value "$evidence" 15 kernel_inventory_sha256)
library_inventory_sha=$(bundle_evidence_value "$evidence" 16 runtime_library_inventory_sha256)
rootfs_inventory_sha=$(bundle_evidence_value "$evidence" 17 rootfs_inventory_sha256)
input_sha=$(bundle_evidence_value "$evidence" 18 assembly_input_sha256)
assembler_sha=$(bundle_evidence_value "$evidence" 19 assembler_sha256)
[ "$(bundle_evidence_value "$evidence" 20 exact_inventory)" = 1 ] || bundle_fail 'exact inventory policy is false'
[ "$(bundle_evidence_value "$evidence" 21 compiler_jit_free)" = 1 ] || bundle_fail 'compiler/JIT-free policy is false'

for digest in "$launch_sha" "$descriptor_sha" "$policy_sha" "$lunaflux_sha" \
  "$worker_sha" "$model_inventory_sha" "$policy_inventory_sha" \
  "$kernel_manifest_sha" "$kernel_inventory_sha" "$library_inventory_sha" \
  "$rootfs_inventory_sha" "$input_sha" "$assembler_sha"; do
  bundle_is_lower_sha256 "$digest" || bundle_fail 'deployment evidence contains an invalid digest'
done
bundle_is_strict_relative "$descriptor_relative" || bundle_fail 'invalid runtime descriptor locator'
bundle_is_strict_relative "$policy_relative" || bundle_fail 'invalid instance policy locator'
bundle_is_strict_relative "$kernel_manifest_relative" || bundle_fail 'invalid kernel manifest locator'

bundle_require_digest "$bundle/launch-root/lunaflux.launch.json" "$launch_sha" 'launch file'
bundle_require_digest "$bundle/model-root/$descriptor_relative" "$descriptor_sha" 'runtime descriptor'
bundle_require_digest "$bundle/policy-root/$policy_relative" "$policy_sha" 'instance policy'
bundle_require_digest "$bundle/oci-context/rootfs/opt/lunaflux/bin/lunaflux" "$lunaflux_sha" 'lunaflux executable'
bundle_require_digest "$bundle/oci-context/rootfs/opt/lunaflux/bin/lunaflux-device-worker" "$worker_sha" 'worker executable'
bundle_require_digest "$bundle/evidence/model.files.sha256" "$model_inventory_sha" 'model inventory'
bundle_require_digest "$bundle/evidence/policy.files.sha256" "$policy_inventory_sha" 'policy inventory'
bundle_require_digest "$bundle/evidence/kernel.files.sha256" "$kernel_inventory_sha" 'kernel inventory'
bundle_require_digest "$bundle/evidence/runtime-libraries.files.sha256" "$library_inventory_sha" 'library inventory'
bundle_require_digest "$bundle/evidence/assembler.files.sha256" "$assembler_sha" 'assembler inventory'
bundle_require_digest "$bundle/oci-context/metadata/artifacts.sha256" "$rootfs_inventory_sha" 'rootfs inventory'

bundle_validate_inventory "$bundle/evidence/model.files.sha256" "$bundle/model-root" model
bundle_validate_inventory "$bundle/evidence/policy.files.sha256" "$bundle/policy-root" policy
bundle_validate_inventory "$bundle/evidence/kernel.files.sha256" \
  "$bundle/oci-context/rootfs/opt/lunaflux/kernels" kernel
if [ "$(sed -n '1p' "$bundle/evidence/runtime-libraries.files.sha256")" = none ]; then
  bundle_require_newline_terminated \
    "$bundle/evidence/runtime-libraries.files.sha256" 'none library inventory'
  [ "$(wc -l < "$bundle/evidence/runtime-libraries.files.sha256" | tr -d ' ')" -eq 1 ] ||
    bundle_fail 'none library inventory contains extra lines'
  [ -z "$(find "$bundle/oci-context/rootfs/opt/lunaflux/lib" -type f -print)" ] ||
    bundle_fail 'library inventory says none but staged libraries exist'
else
  bundle_validate_inventory "$bundle/evidence/runtime-libraries.files.sha256" \
    "$bundle/oci-context/rootfs/opt/lunaflux/lib" library
fi

[ "$(sed -n '1p' "$bundle/oci-context/metadata/base-image.ref")" = "$base_image" ] ||
  bundle_fail 'base image evidence does not match OCI metadata'
[ "$(sed -n '1p' "$bundle/oci-context/metadata/linux-architecture")" = "$architecture" ] ||
  bundle_fail 'architecture evidence does not match OCI metadata'
[ "$(sed -n '1p' "$bundle/oci-context/metadata/kernel-manifest.relative")" = "$kernel_manifest_relative" ] ||
  bundle_fail 'kernel manifest locator evidence does not match OCI metadata'
[ "$(sed -n '1p' "$bundle/oci-context/metadata/kernel-manifest.sha256")" = "$kernel_manifest_sha" ] ||
  bundle_fail 'kernel manifest digest evidence does not match OCI metadata'

control_jsons=$(find "$bundle/model-root" "$bundle/policy-root" \
  "$bundle/oci-context/rootfs/opt/lunaflux/kernels" -type f -name '*.json' -print)
if grep -E -i 'nvrtc|runtime[_-]?jit|developer[_-]?jit|\.ptx|source[_-]?(path|code)' \
  "$bundle/launch-root/lunaflux.launch.json" $control_jsons \
  >/dev/null 2>&1; then
  bundle_fail 'authenticated control material contains compiler or JIT vocabulary'
fi

"$repo_root/scripts/verify-oci-context.sh" "$base_image" "$bundle/oci-context" >/dev/null
rm -rf "$BUNDLE_SCRATCH_DIR"
trap - EXIT HUP INT TERM
printf '%s\n' 'LunaFlux deployment bundle is exact, launch-bound, and compiler/JIT-free.'
