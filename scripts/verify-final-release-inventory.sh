#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/final-release-inventory-common.sh"

[ "$#" -eq 2 ] || {
  printf '%s\n' \
    'usage: verify-final-release-inventory.sh ABSOLUTE_INVENTORY ABSOLUTE_AUTHENTICATOR#sha256=HEX' >&2
  exit 2
}
root=$1
authenticator_argument=$2
case "$authenticator_argument" in /*#sha256=*) ;; *) final_inventory_fail 'authenticator argument is not digest suffixed' ;; esac
authenticator=${authenticator_argument%#sha256=*}
authenticator_digest=${authenticator_argument##*#sha256=}
final_inventory_require_directory "$root"
final_inventory_require_file "$authenticator" 'external authenticator'
final_inventory_require_digest "$authenticator" "$authenticator_digest" 'external authenticator'
[ -x "$authenticator" ] || final_inventory_fail 'external authenticator is not executable'

FINAL_INVENTORY_SCRATCH=$(mktemp -d /tmp/lunaflux-final-release-verify.XXXXXX) ||
  final_inventory_fail 'could not create private verifier scratch'
FINAL_INVENTORY_SCRATCH=$(CDPATH= cd -- "$FINAL_INVENTORY_SCRATCH" && pwd -P)
export FINAL_INVENTORY_SCRATCH
trap 'rm -rf "$FINAL_INVENTORY_SCRATCH"' EXIT HUP INT TERM
authenticator_exec=$FINAL_INVENTORY_SCRATCH/external-authenticator
cp "$authenticator" "$authenticator_exec"
chmod 500 "$authenticator_exec"
final_inventory_require_digest "$authenticator_exec" "$authenticator_digest" \
  'private external authenticator copy'

top=$(find "$root" -mindepth 1 -maxdepth 1 -print | sed "s#^$root/##" | LC_ALL=C sort)
[ "$top" = "approvals
artifacts
final-release.v1
inventory.files.sha256
payload.files.sha256
tools
tools.files.sha256" ] || final_inventory_fail 'inventory has unexpected top-level entries'
if find "$root" -type l -print | grep -q .; then
  final_inventory_fail 'inventory contains a symbolic link'
fi
if find "$root" ! -type d ! -type f -print | grep -q .; then
  final_inventory_fail 'inventory contains a special filesystem object'
fi
for fir_directory in $(find "$root" -type d -print); do
  [ "$(final_inventory_mode "$fir_directory")" = 555 ] ||
    final_inventory_fail 'inventory directory is not mode 555'
done
for fir_file in $(find "$root" -type f -print); do
  [ "$(final_inventory_mode "$fir_file")" = 444 ] ||
    final_inventory_fail 'inventory file is not mode 444'
  [ "$(final_inventory_links "$fir_file")" = 1 ] ||
    final_inventory_fail 'inventory file has a hard-link alias'
done

manifest=$root/final-release.v1
[ "$(wc -l < "$manifest" | tr -d ' ')" -eq 22 ] ||
  final_inventory_fail 'final release manifest must contain exactly 22 lines'
final_inventory_require_newline "$manifest" 'final release manifest'
[ "$(final_inventory_field "$manifest" 1 schema)" = 'lunaflux.final-release-inventory.v1' ] ||
  final_inventory_fail 'unsupported final release inventory schema'
subject=$(final_inventory_field "$manifest" 2 release_subject_sha256)
final_inventory_is_sha256 "$subject" || final_inventory_fail 'release subject digest is invalid'
oci_image=$(final_inventory_field "$manifest" 3 oci_image)
final_inventory_require_image "$oci_image"

payload_expected=$FINAL_INVENTORY_SCRATCH/payload-paths
cat > "$payload_expected" <<'EOF'
approvals/kernel-manifest.json.approval
approvals/license-inventory.json.approval
approvals/oci-digest.txt.approval
approvals/provenance.json.approval
approvals/rootfs-scan.json.approval
approvals/runtime-contracts.json.approval
approvals/sbom.json.approval
approvals/source-identity.txt.approval
artifacts/kernel-manifest.json
artifacts/license-inventory.json
artifacts/oci-digest.txt
artifacts/provenance.json
artifacts/rootfs-scan.json
artifacts/runtime-contracts.json
artifacts/sbom.json
artifacts/source-identity.txt
EOF
final_inventory_validate_files "$root" "$root/payload.files.sha256" "$payload_expected"

tools_expected=$FINAL_INVENTORY_SCRATCH/tool-paths
cat > "$tools_expected" <<'EOF'
tools/Containerfile
tools/assemble-final-release-inventory.sh
tools/build-oci-image.sh
tools/external-authenticator
tools/final-release-inventory-common.sh
tools/verify-final-release-inventory.sh
tools/verify-oci-context.sh
tools/verify-release-bundle.sh
EOF
final_inventory_validate_files "$root" "$root/tools.files.sha256" "$tools_expected"

all_expected=$FINAL_INVENTORY_SCRATCH/all-paths
find "$root" -type f ! -path "$root/inventory.files.sha256" -print |
  sed "s#^$root/##" | LC_ALL=C sort > "$all_expected"
final_inventory_validate_files "$root" "$root/inventory.files.sha256" "$all_expected"

oci_digest_sha=$(final_inventory_field "$manifest" 4 oci_digest_sha256)
oci_digest_approval_sha=$(final_inventory_field "$manifest" 5 oci_digest_approval_sha256)
sbom_sha=$(final_inventory_field "$manifest" 6 sbom_sha256)
sbom_approval_sha=$(final_inventory_field "$manifest" 7 sbom_approval_sha256)
license_inventory_sha=$(final_inventory_field "$manifest" 8 license_inventory_sha256)
license_inventory_approval_sha=$(final_inventory_field "$manifest" 9 license_inventory_approval_sha256)
provenance_sha=$(final_inventory_field "$manifest" 10 provenance_sha256)
provenance_approval_sha=$(final_inventory_field "$manifest" 11 provenance_approval_sha256)
kernel_manifest_sha=$(final_inventory_field "$manifest" 12 kernel_manifest_sha256)
kernel_manifest_approval_sha=$(final_inventory_field "$manifest" 13 kernel_manifest_approval_sha256)
rootfs_scan_sha=$(final_inventory_field "$manifest" 14 rootfs_scan_sha256)
rootfs_scan_approval_sha=$(final_inventory_field "$manifest" 15 rootfs_scan_approval_sha256)
runtime_contracts_sha=$(final_inventory_field "$manifest" 16 runtime_contracts_sha256)
runtime_contracts_approval_sha=$(final_inventory_field "$manifest" 17 runtime_contracts_approval_sha256)
source_identity_sha=$(final_inventory_field "$manifest" 18 source_identity_sha256)
source_identity_approval_sha=$(final_inventory_field "$manifest" 19 source_identity_approval_sha256)
payload_inventory_sha=$(final_inventory_field "$manifest" 20 payload_inventory_sha256)
tools_inventory_sha=$(final_inventory_field "$manifest" 21 tools_inventory_sha256)
recorded_authenticator_sha=$(final_inventory_field "$manifest" 22 external_authenticator_sha256)
[ "$recorded_authenticator_sha" = "$authenticator_digest" ] ||
  final_inventory_fail 'external authenticator identity does not match the assembled inventory'
final_inventory_require_digest "$root/tools/external-authenticator" "$authenticator_digest" \
  'recorded external authenticator'
final_inventory_require_digest "$root/payload.files.sha256" "$payload_inventory_sha" \
  'payload inventory'
final_inventory_require_digest "$root/tools.files.sha256" "$tools_inventory_sha" 'tool inventory'

verify_role() {
  vr_role=$1
  vr_digest=$2
  vr_approval_digest=$3
  final_inventory_require_digest "$root/artifacts/$vr_role" "$vr_digest" "$vr_role artifact"
  final_inventory_require_digest "$root/approvals/$vr_role.approval" \
    "$vr_approval_digest" "$vr_role approval"
  env -i LC_ALL=C PATH=/usr/bin:/bin "$authenticator_exec" verify \
    "$vr_role" "$subject" "$root/artifacts/$vr_role" "$vr_digest" \
    "$root/approvals/$vr_role.approval" "$vr_approval_digest" "$oci_image" ||
    final_inventory_fail "external authenticator rejected $vr_role"
  final_inventory_require_digest "$root/artifacts/$vr_role" "$vr_digest" \
    "$vr_role artifact after authentication"
  final_inventory_require_digest "$root/approvals/$vr_role.approval" \
    "$vr_approval_digest" "$vr_role approval after authentication"
}

verify_role oci-digest.txt "$oci_digest_sha" "$oci_digest_approval_sha"
[ "$(wc -l < "$root/artifacts/oci-digest.txt" | tr -d ' ')" -eq 1 ] &&
  [ "$(sed -n '1p' "$root/artifacts/oci-digest.txt")" = "image=$oci_image" ] ||
  final_inventory_fail 'OCI digest artifact does not bind the manifest image'
verify_role sbom.json "$sbom_sha" "$sbom_approval_sha"
verify_role license-inventory.json "$license_inventory_sha" "$license_inventory_approval_sha"
verify_role provenance.json "$provenance_sha" "$provenance_approval_sha"
verify_role kernel-manifest.json "$kernel_manifest_sha" "$kernel_manifest_approval_sha"
verify_role rootfs-scan.json "$rootfs_scan_sha" "$rootfs_scan_approval_sha"
verify_role runtime-contracts.json "$runtime_contracts_sha" "$runtime_contracts_approval_sha"
verify_role source-identity.txt "$source_identity_sha" "$source_identity_approval_sha"
[ "$(wc -l < "$root/artifacts/source-identity.txt" | tr -d ' ')" -eq 3 ] ||
  final_inventory_fail 'source identity must contain exactly three lines'
[ "$(final_inventory_field "$root/artifacts/source-identity.txt" 1 schema)" = \
    'lunaflux.source-identity.v1' ] ||
  final_inventory_fail 'source identity schema is invalid'
source_archive_sha=$(final_inventory_field \
  "$root/artifacts/source-identity.txt" 2 source_archive_sha256)
source_inventory_sha=$(final_inventory_field \
  "$root/artifacts/source-identity.txt" 3 source_inventory_sha256)
final_inventory_is_sha256 "$source_archive_sha" ||
  final_inventory_fail 'source archive identity is invalid'
final_inventory_is_sha256 "$source_inventory_sha" ||
  final_inventory_fail 'source inventory identity is invalid'
[ "$source_archive_sha" != "$source_inventory_sha" ] ||
  final_inventory_fail 'source archive and inventory identities collapsed'

rm -rf "$FINAL_INVENTORY_SCRATCH"
trap - EXIT HUP INT TERM
printf '%s\n' 'LunaFlux final release inventory is exact and externally reauthenticated.'
