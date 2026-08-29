#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL
umask 077

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/final-release-inventory-common.sh"

usage() {
  printf '%s\n' \
    'usage: assemble-final-release-inventory.sh ABSOLUTE_INPUT#sha256=HEX ABSOLUTE_AUTHENTICATOR#sha256=HEX ABSOLUTE_NEW_OUTPUT' >&2
  exit 2
}

[ "$#" -eq 3 ] || usage
input_argument=$1
authenticator_argument=$2
output=$3
case "$input_argument" in /*#sha256=*) ;; *) usage ;; esac
case "$authenticator_argument" in /*#sha256=*) ;; *) usage ;; esac
input=${input_argument%#sha256=*}
input_digest=${input_argument##*#sha256=}
authenticator=${authenticator_argument%#sha256=*}
authenticator_digest=${authenticator_argument##*#sha256=}

final_inventory_require_file "$input" 'assembly input'
final_inventory_require_digest "$input" "$input_digest" 'assembly input'
final_inventory_require_newline "$input" 'assembly input'
final_inventory_require_file "$authenticator" 'external authenticator'
final_inventory_require_digest "$authenticator" "$authenticator_digest" 'external authenticator'
final_inventory_require_size "$authenticator" 16777216 'external authenticator'
[ -x "$authenticator" ] || final_inventory_fail 'external authenticator is not executable'

case "$output" in /*) ;; *) final_inventory_fail 'output path is not absolute' ;; esac
case "$output" in /|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
  final_inventory_fail 'output path is not a safe canonical path'
  ;;
esac
[ ! -e "$output" ] || final_inventory_fail 'refusing to overwrite an existing output'
output_parent=$(CDPATH= cd -- "$(dirname -- "$output")" && pwd -P)
[ "$output_parent/$(basename -- "$output")" = "$output" ] ||
  final_inventory_fail 'output parent is not canonical'

[ "$(wc -l < "$input" | tr -d ' ')" -eq 35 ] ||
  final_inventory_fail 'assembly input must contain exactly 35 lines'
[ -z "$(sed -n '36p' "$input")" ] || final_inventory_fail 'assembly input has trailing fields'
if grep -q "$(printf '\r')" "$input"; then
  final_inventory_fail 'assembly input contains carriage returns'
fi

[ "$(final_inventory_field "$input" 1 schema)" = 'lunaflux.final-release-assembly.v1' ] ||
  final_inventory_fail 'unsupported assembly schema'
subject=$(final_inventory_field "$input" 2 release_subject_sha256)
final_inventory_is_sha256 "$subject" || final_inventory_fail 'release subject digest is invalid'
oci_image=$(final_inventory_field "$input" 3 oci_image)
final_inventory_require_image "$oci_image"

oci_digest_source=$(final_inventory_field "$input" 4 oci_digest_source)
oci_digest_sha=$(final_inventory_field "$input" 5 oci_digest_sha256)
oci_digest_approval=$(final_inventory_field "$input" 6 oci_digest_approval_source)
oci_digest_approval_sha=$(final_inventory_field "$input" 7 oci_digest_approval_sha256)
sbom_source=$(final_inventory_field "$input" 8 sbom_source)
sbom_sha=$(final_inventory_field "$input" 9 sbom_sha256)
sbom_approval=$(final_inventory_field "$input" 10 sbom_approval_source)
sbom_approval_sha=$(final_inventory_field "$input" 11 sbom_approval_sha256)
license_inventory_source=$(final_inventory_field "$input" 12 license_inventory_source)
license_inventory_sha=$(final_inventory_field "$input" 13 license_inventory_sha256)
license_inventory_approval=$(final_inventory_field "$input" 14 license_inventory_approval_source)
license_inventory_approval_sha=$(final_inventory_field "$input" 15 license_inventory_approval_sha256)
provenance_source=$(final_inventory_field "$input" 16 provenance_source)
provenance_sha=$(final_inventory_field "$input" 17 provenance_sha256)
provenance_approval=$(final_inventory_field "$input" 18 provenance_approval_source)
provenance_approval_sha=$(final_inventory_field "$input" 19 provenance_approval_sha256)
kernel_manifest_source=$(final_inventory_field "$input" 20 kernel_manifest_source)
kernel_manifest_sha=$(final_inventory_field "$input" 21 kernel_manifest_sha256)
kernel_manifest_approval=$(final_inventory_field "$input" 22 kernel_manifest_approval_source)
kernel_manifest_approval_sha=$(final_inventory_field "$input" 23 kernel_manifest_approval_sha256)
rootfs_scan_source=$(final_inventory_field "$input" 24 rootfs_scan_source)
rootfs_scan_sha=$(final_inventory_field "$input" 25 rootfs_scan_sha256)
rootfs_scan_approval=$(final_inventory_field "$input" 26 rootfs_scan_approval_source)
rootfs_scan_approval_sha=$(final_inventory_field "$input" 27 rootfs_scan_approval_sha256)
runtime_contracts_source=$(final_inventory_field "$input" 28 runtime_contracts_source)
runtime_contracts_sha=$(final_inventory_field "$input" 29 runtime_contracts_sha256)
runtime_contracts_approval=$(final_inventory_field "$input" 30 runtime_contracts_approval_source)
runtime_contracts_approval_sha=$(final_inventory_field "$input" 31 runtime_contracts_approval_sha256)
source_identity_source=$(final_inventory_field "$input" 32 source_identity_source)
source_identity_sha=$(final_inventory_field "$input" 33 source_identity_sha256)
source_identity_approval=$(final_inventory_field "$input" 34 source_identity_approval_source)
source_identity_approval_sha=$(final_inventory_field "$input" 35 source_identity_approval_sha256)

stage=$(mktemp -d "$output_parent/.lunaflux-final-release-stage.XXXXXX") ||
  final_inventory_fail 'could not create output-adjacent private stage'
FINAL_INVENTORY_SCRATCH=$(mktemp -d /tmp/lunaflux-final-release-assemble.XXXXXX) || {
  rmdir "$stage"
  final_inventory_fail 'could not create private verifier scratch'
}
FINAL_INVENTORY_SCRATCH=$(CDPATH= cd -- "$FINAL_INVENTORY_SCRATCH" && pwd -P)
export FINAL_INVENTORY_SCRATCH
cleanup() {
  chmod -R u+w "$stage" "$FINAL_INVENTORY_SCRATCH" 2>/dev/null || true
  rm -rf "$stage" "$FINAL_INVENTORY_SCRATCH"
  if [ -n "${claimed_output:-}" ]; then
    chmod -R u+w "$claimed_output" 2>/dev/null || true
    rm -rf "$claimed_output"
  fi
}
trap cleanup EXIT HUP INT TERM
mkdir "$stage/artifacts" "$stage/approvals" "$stage/tools"
authenticator_exec=$FINAL_INVENTORY_SCRATCH/external-authenticator
cp "$authenticator" "$authenticator_exec"
chmod 500 "$authenticator_exec"
final_inventory_require_digest "$authenticator_exec" "$authenticator_digest" \
  'private external authenticator copy'

seen=$FINAL_INVENTORY_SCRATCH/seen-digests
: > "$seen"
payload_paths=$FINAL_INVENTORY_SCRATCH/payload-paths
: > "$payload_paths"

copy_and_authenticate() {
  fia_role=$1
  fia_source=$2
  fia_digest=$3
  fia_approval=$4
  fia_approval_digest=$5
  final_inventory_require_file "$fia_source" "$fia_role artifact"
  final_inventory_require_digest "$fia_source" "$fia_digest" "$fia_role artifact"
  final_inventory_require_size "$fia_source" 268435456 "$fia_role artifact"
  final_inventory_require_file "$fia_approval" "$fia_role approval"
  final_inventory_require_digest "$fia_approval" "$fia_approval_digest" "$fia_role approval"
  final_inventory_require_size "$fia_approval" 1048576 "$fia_role approval"
  for fia_seen in "$fia_digest" "$fia_approval_digest"; do
    if grep -F -x "$fia_seen" "$seen" >/dev/null 2>&1; then
      final_inventory_fail 'artifact and approval digests must be globally unique'
    fi
    printf '%s\n' "$fia_seen" >> "$seen"
  done
  fia_artifact=artifacts/$fia_role
  fia_approval_path=approvals/$fia_role.approval
  cp "$fia_source" "$stage/$fia_artifact"
  cp "$fia_approval" "$stage/$fia_approval_path"
  chmod 444 "$stage/$fia_artifact" "$stage/$fia_approval_path"
  final_inventory_require_digest "$stage/$fia_artifact" "$fia_digest" "$fia_role staged artifact"
  final_inventory_require_digest "$stage/$fia_approval_path" "$fia_approval_digest" "$fia_role staged approval"
  env -i LC_ALL=C PATH=/usr/bin:/bin "$authenticator_exec" verify \
    "$fia_role" "$subject" "$stage/$fia_artifact" "$fia_digest" \
    "$stage/$fia_approval_path" "$fia_approval_digest" "$oci_image" ||
    final_inventory_fail "external authenticator rejected $fia_role"
  final_inventory_require_digest "$stage/$fia_artifact" "$fia_digest" "$fia_role artifact after authentication"
  final_inventory_require_digest "$stage/$fia_approval_path" "$fia_approval_digest" "$fia_role approval after authentication"
  printf '%s\n%s\n' "$fia_artifact" "$fia_approval_path" >> "$payload_paths"
}

copy_and_authenticate oci-digest.txt "$oci_digest_source" "$oci_digest_sha" \
  "$oci_digest_approval" "$oci_digest_approval_sha"
[ "$(wc -l < "$stage/artifacts/oci-digest.txt" | tr -d ' ')" -eq 1 ] &&
  [ "$(sed -n '1p' "$stage/artifacts/oci-digest.txt")" = "image=$oci_image" ] ||
  final_inventory_fail 'OCI digest artifact does not exactly bind the selected image'
copy_and_authenticate sbom.json "$sbom_source" "$sbom_sha" "$sbom_approval" "$sbom_approval_sha"
copy_and_authenticate license-inventory.json "$license_inventory_source" "$license_inventory_sha" \
  "$license_inventory_approval" "$license_inventory_approval_sha"
copy_and_authenticate provenance.json "$provenance_source" "$provenance_sha" \
  "$provenance_approval" "$provenance_approval_sha"
copy_and_authenticate kernel-manifest.json "$kernel_manifest_source" "$kernel_manifest_sha" \
  "$kernel_manifest_approval" "$kernel_manifest_approval_sha"
copy_and_authenticate rootfs-scan.json "$rootfs_scan_source" "$rootfs_scan_sha" \
  "$rootfs_scan_approval" "$rootfs_scan_approval_sha"
copy_and_authenticate runtime-contracts.json "$runtime_contracts_source" "$runtime_contracts_sha" \
  "$runtime_contracts_approval" "$runtime_contracts_approval_sha"
copy_and_authenticate source-identity.txt "$source_identity_source" "$source_identity_sha" \
  "$source_identity_approval" "$source_identity_approval_sha"
[ "$(wc -l < "$stage/artifacts/source-identity.txt" | tr -d ' ')" -eq 3 ] ||
  final_inventory_fail 'source identity must contain exactly three lines'
[ "$(final_inventory_field "$stage/artifacts/source-identity.txt" 1 schema)" = \
    'lunaflux.source-identity.v1' ] ||
  final_inventory_fail 'source identity schema is invalid'
source_archive_sha=$(final_inventory_field \
  "$stage/artifacts/source-identity.txt" 2 source_archive_sha256)
source_inventory_sha=$(final_inventory_field \
  "$stage/artifacts/source-identity.txt" 3 source_inventory_sha256)
final_inventory_is_sha256 "$source_archive_sha" ||
  final_inventory_fail 'source archive identity is invalid'
final_inventory_is_sha256 "$source_inventory_sha" ||
  final_inventory_fail 'source inventory identity is invalid'
[ "$source_archive_sha" != "$source_inventory_sha" ] ||
  final_inventory_fail 'source archive and inventory identities collapsed'

LC_ALL=C sort -o "$payload_paths" "$payload_paths"
final_inventory_write_paths "$stage" "$payload_paths" "$stage/payload.files.sha256"
chmod 444 "$stage/payload.files.sha256"
payload_inventory_sha=$(final_inventory_sha256 "$stage/payload.files.sha256")

tools_paths=$FINAL_INVENTORY_SCRATCH/tool-paths
: > "$tools_paths"
copy_tool() {
  fit_source=$1
  fit_name=$2
  final_inventory_require_file "$fit_source" "release tool $fit_name"
  cp "$fit_source" "$stage/tools/$fit_name"
  chmod 444 "$stage/tools/$fit_name"
  printf '%s\n' "tools/$fit_name" >> "$tools_paths"
}
copy_tool "$authenticator_exec" external-authenticator
copy_tool "$repo_root/deploy/oci/Containerfile" Containerfile
copy_tool "$repo_root/scripts/assemble-final-release-inventory.sh" assemble-final-release-inventory.sh
copy_tool "$repo_root/scripts/build-oci-image.sh" build-oci-image.sh
copy_tool "$repo_root/scripts/final-release-inventory-common.sh" final-release-inventory-common.sh
copy_tool "$repo_root/scripts/verify-final-release-inventory.sh" verify-final-release-inventory.sh
copy_tool "$repo_root/scripts/verify-oci-context.sh" verify-oci-context.sh
copy_tool "$repo_root/scripts/verify-release-bundle.sh" verify-release-bundle.sh
LC_ALL=C sort -o "$tools_paths" "$tools_paths"
final_inventory_write_paths "$stage" "$tools_paths" "$stage/tools.files.sha256"
chmod 444 "$stage/tools.files.sha256"
tools_inventory_sha=$(final_inventory_sha256 "$stage/tools.files.sha256")

cat > "$stage/final-release.v1" <<EOF
schema=lunaflux.final-release-inventory.v1
release_subject_sha256=$subject
oci_image=$oci_image
oci_digest_sha256=$oci_digest_sha
oci_digest_approval_sha256=$oci_digest_approval_sha
sbom_sha256=$sbom_sha
sbom_approval_sha256=$sbom_approval_sha
license_inventory_sha256=$license_inventory_sha
license_inventory_approval_sha256=$license_inventory_approval_sha
provenance_sha256=$provenance_sha
provenance_approval_sha256=$provenance_approval_sha
kernel_manifest_sha256=$kernel_manifest_sha
kernel_manifest_approval_sha256=$kernel_manifest_approval_sha
rootfs_scan_sha256=$rootfs_scan_sha
rootfs_scan_approval_sha256=$rootfs_scan_approval_sha
runtime_contracts_sha256=$runtime_contracts_sha
runtime_contracts_approval_sha256=$runtime_contracts_approval_sha
source_identity_sha256=$source_identity_sha
source_identity_approval_sha256=$source_identity_approval_sha
payload_inventory_sha256=$payload_inventory_sha
tools_inventory_sha256=$tools_inventory_sha
external_authenticator_sha256=$authenticator_digest
EOF
chmod 444 "$stage/final-release.v1"

all_paths=$FINAL_INVENTORY_SCRATCH/all-paths
find "$stage" -type f -print | sed "s#^$stage/##" | LC_ALL=C sort > "$all_paths"
final_inventory_write_paths "$stage" "$all_paths" "$stage/inventory.files.sha256"
chmod 444 "$stage/inventory.files.sha256"
find "$stage" -type d -exec chmod 555 {} \;

"$repo_root/scripts/verify-final-release-inventory.sh" \
  "$stage" "$authenticator#sha256=$authenticator_digest" >/dev/null
find "$stage" -type d -exec chmod 700 {} \;
mkdir "$output" || final_inventory_fail 'could not claim the new output atomically'
claimed_output=$output
for publication_entry in approvals artifacts final-release.v1 \
  inventory.files.sha256 payload.files.sha256 tools tools.files.sha256; do
  mv "$stage/$publication_entry" "$output/$publication_entry"
done
rmdir "$stage"
find "$output" -type d -exec chmod 555 {} \;
"$repo_root/scripts/verify-final-release-inventory.sh" \
  "$output" "$authenticator#sha256=$authenticator_digest" >/dev/null
claimed_output=
rm -rf "$FINAL_INVENTORY_SCRATCH"
trap - EXIT HUP INT TERM
printf '%s\n' 'LunaFlux final release inventory assembled; external approvals remain external authority.'
