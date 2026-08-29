#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
assembler=$repo_root/scripts/assemble-final-release-inventory.sh
verifier=$repo_root/scripts/verify-final-release-inventory.sh

fail() {
  printf '%s\n' "LunaFlux final release inventory gate failed: $1" >&2
  exit 1
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

assert_assembly_rejected() {
  label=$1
  recipe=$2
  authenticator_argument=$3
  output=$4
  recipe_digest=$(sha256_file "$recipe")
  if "$assembler" "$recipe#sha256=$recipe_digest" \
    "$authenticator_argument" "$output" >/dev/null 2>&1; then
    fail "assembler passed hostile fixture: $label"
  fi
  [ ! -e "$output" ] || fail "failed assembly retained requested output: $label"
}

assert_verifier_rejected() {
  label=$1
  root=$2
  authenticator_argument=$3
  if "$verifier" "$root" "$authenticator_argument" >/dev/null 2>&1; then
    fail "verifier passed hostile fixture: $label"
  fi
}

sh -n "$repo_root/scripts/final-release-inventory-common.sh"
sh -n "$assembler"
sh -n "$verifier"

fixture=$(mktemp -d /tmp/lunaflux-final-release-gate.XXXXXX)
fixture=$(CDPATH= cd -- "$fixture" && pwd -P)
trap 'chmod -R u+w "$fixture" 2>/dev/null || true; rm -rf "$fixture"' EXIT HUP INT TERM
mkdir "$fixture/sources" "$fixture/approvals"

# This authenticator is deliberately fixture-only. Its digest is selected by
# this test, not by a deployment trust store, so it can never be release proof.
authenticator=$fixture/fixture-authenticator
cat > "$authenticator" <<'EOF'
#!/bin/sh
set -eu
[ "$#" -eq 8 ] && [ "$1" = verify ] || exit 90
role=$2
subject=$3
artifact=$4
artifact_sha=$5
approval=$6
approval_sha=$7
image=$8
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}
[ "$(sha256_file "$artifact")" = "$artifact_sha" ] || exit 91
[ "$(sha256_file "$approval")" = "$approval_sha" ] || exit 92
expected=$(mktemp /tmp/lunaflux-fixture-approval.XXXXXX)
trap 'rm -f "$expected"' EXIT HUP INT TERM
cat > "$expected" <<EOT
schema=lunaflux.fixture-external-approval.v1
role=$role
subject_sha256=$subject
artifact_sha256=$artifact_sha
oci_image=$image
approved=1
EOT
cmp -s "$expected" "$approval"
EOF
chmod 755 "$authenticator"
authenticator_sha=$(sha256_file "$authenticator")
authenticator_argument=$authenticator#sha256=$authenticator_sha

subject=1111111111111111111111111111111111111111111111111111111111111111
image_digest=2222222222222222222222222222222222222222222222222222222222222222
oci_image=registry.invalid/lunaflux/final@sha256:$image_digest
roles='oci-digest.txt sbom.json license-inventory.json provenance.json kernel-manifest.json rootfs-scan.json runtime-contracts.json source-identity.txt'

for role in $roles; do
  case "$role" in
    oci-digest.txt) printf '%s\n' "image=$oci_image" > "$fixture/sources/$role" ;;
    source-identity.txt)
      printf '%s\n' \
        'schema=lunaflux.source-identity.v1' \
        'source_archive_sha256=3333333333333333333333333333333333333333333333333333333333333333' \
        'source_inventory_sha256=4444444444444444444444444444444444444444444444444444444444444444' \
        > "$fixture/sources/$role"
      ;;
    *) printf '{"schema":"lunaflux.fixture.%s","passed":true}\n' "$role" > "$fixture/sources/$role" ;;
  esac
  artifact_sha=$(sha256_file "$fixture/sources/$role")
  cat > "$fixture/approvals/$role.approval" <<EOF
schema=lunaflux.fixture-external-approval.v1
role=$role
subject_sha256=$subject
artifact_sha256=$artifact_sha
oci_image=$oci_image
approved=1
EOF
done

recipe=$fixture/assembly-input.v1
cat > "$recipe" <<EOF
schema=lunaflux.final-release-assembly.v1
release_subject_sha256=$subject
oci_image=$oci_image
oci_digest_source=$fixture/sources/oci-digest.txt
oci_digest_sha256=$(sha256_file "$fixture/sources/oci-digest.txt")
oci_digest_approval_source=$fixture/approvals/oci-digest.txt.approval
oci_digest_approval_sha256=$(sha256_file "$fixture/approvals/oci-digest.txt.approval")
sbom_source=$fixture/sources/sbom.json
sbom_sha256=$(sha256_file "$fixture/sources/sbom.json")
sbom_approval_source=$fixture/approvals/sbom.json.approval
sbom_approval_sha256=$(sha256_file "$fixture/approvals/sbom.json.approval")
license_inventory_source=$fixture/sources/license-inventory.json
license_inventory_sha256=$(sha256_file "$fixture/sources/license-inventory.json")
license_inventory_approval_source=$fixture/approvals/license-inventory.json.approval
license_inventory_approval_sha256=$(sha256_file "$fixture/approvals/license-inventory.json.approval")
provenance_source=$fixture/sources/provenance.json
provenance_sha256=$(sha256_file "$fixture/sources/provenance.json")
provenance_approval_source=$fixture/approvals/provenance.json.approval
provenance_approval_sha256=$(sha256_file "$fixture/approvals/provenance.json.approval")
kernel_manifest_source=$fixture/sources/kernel-manifest.json
kernel_manifest_sha256=$(sha256_file "$fixture/sources/kernel-manifest.json")
kernel_manifest_approval_source=$fixture/approvals/kernel-manifest.json.approval
kernel_manifest_approval_sha256=$(sha256_file "$fixture/approvals/kernel-manifest.json.approval")
rootfs_scan_source=$fixture/sources/rootfs-scan.json
rootfs_scan_sha256=$(sha256_file "$fixture/sources/rootfs-scan.json")
rootfs_scan_approval_source=$fixture/approvals/rootfs-scan.json.approval
rootfs_scan_approval_sha256=$(sha256_file "$fixture/approvals/rootfs-scan.json.approval")
runtime_contracts_source=$fixture/sources/runtime-contracts.json
runtime_contracts_sha256=$(sha256_file "$fixture/sources/runtime-contracts.json")
runtime_contracts_approval_source=$fixture/approvals/runtime-contracts.json.approval
runtime_contracts_approval_sha256=$(sha256_file "$fixture/approvals/runtime-contracts.json.approval")
source_identity_source=$fixture/sources/source-identity.txt
source_identity_sha256=$(sha256_file "$fixture/sources/source-identity.txt")
source_identity_approval_source=$fixture/approvals/source-identity.txt.approval
source_identity_approval_sha256=$(sha256_file "$fixture/approvals/source-identity.txt.approval")
EOF

recipe_sha=$(sha256_file "$recipe")
valid=$fixture/valid
second=$fixture/second
"$assembler" "$recipe#sha256=$recipe_sha" "$authenticator_argument" "$valid" >/dev/null
"$verifier" "$valid" "$authenticator_argument" >/dev/null
"$assembler" "$recipe#sha256=$recipe_sha" "$authenticator_argument" "$second" >/dev/null
"$verifier" "$second" "$authenticator_argument" >/dev/null
diff -r "$valid" "$second" >/dev/null || fail 'identical approved inputs were not deterministic'

if "$assembler" "$recipe#sha256=$recipe_sha" "$authenticator_argument" "$valid" \
  >/dev/null 2>&1; then
  fail 'assembler overwrote an existing inventory'
fi
"$verifier" "$valid" "$authenticator_argument" >/dev/null ||
  fail 'overwrite attempt changed the completed inventory'

substitution=$fixture/substitution-input.v1
cp "$recipe" "$substitution"
printf '%s\n' substituted >> "$fixture/sources/sbom.json"
assert_assembly_rejected substituted-artifact "$substitution" \
  "$authenticator_argument" "$fixture/substitution-output"
sed -n '1p' "$fixture/sources/sbom.json" > "$fixture/sources/sbom.restored"
mv "$fixture/sources/sbom.restored" "$fixture/sources/sbom.json"

symlink_recipe=$fixture/symlink-input.v1
cp "$recipe" "$symlink_recipe"
ln -s "$fixture/sources/sbom.json" "$fixture/sources/sbom-link.json"
sed "s#sbom_source=$fixture/sources/sbom.json#sbom_source=$fixture/sources/sbom-link.json#" \
  "$recipe" > "$symlink_recipe"
assert_assembly_rejected symlink-source "$symlink_recipe" \
  "$authenticator_argument" "$fixture/symlink-output"

hardlink_recipe=$fixture/hardlink-input.v1
ln "$fixture/sources/sbom.json" "$fixture/sources/sbom-hardlink.json"
sed "s#sbom_source=$fixture/sources/sbom.json#sbom_source=$fixture/sources/sbom-hardlink.json#" \
  "$recipe" > "$hardlink_recipe"
assert_assembly_rejected hardlink-source "$hardlink_recipe" \
  "$authenticator_argument" "$fixture/hardlink-output"
rm "$fixture/sources/sbom-hardlink.json"

replay_approval=$fixture/approvals/replayed.approval
artifact_sha=$(sha256_file "$fixture/sources/sbom.json")
cat > "$replay_approval" <<EOF
schema=lunaflux.fixture-external-approval.v1
role=sbom.json
subject_sha256=9999999999999999999999999999999999999999999999999999999999999999
artifact_sha256=$artifact_sha
oci_image=$oci_image
approved=1
EOF
replay_recipe=$fixture/replay-input.v1
sed \
  -e "s#sbom_approval_source=$fixture/approvals/sbom.json.approval#sbom_approval_source=$replay_approval#" \
  -e "s#sbom_approval_sha256=.*#sbom_approval_sha256=$(sha256_file "$replay_approval")#" \
  "$recipe" > "$replay_recipe"
assert_assembly_rejected cross-subject-replay "$replay_recipe" \
  "$authenticator_argument" "$fixture/replay-output"

fake_bin=$fixture/fake-bin
mkdir "$fake_bin"
cp_state=$fixture/fake-cp-state
cat > "$fake_bin/cp" <<EOF
#!/bin/sh
count=0
[ ! -f "$cp_state" ] || count=\$(sed -n '1p' "$cp_state")
count=\$((count + 1))
printf '%s\n' "\$count" > "$cp_state"
[ "\$count" -ne 3 ] || exit 73
exec /bin/cp "\$@"
EOF
chmod 755 "$fake_bin/cp"
partial=$fixture/partial-output
if PATH="$fake_bin:$PATH" "$assembler" "$recipe#sha256=$recipe_sha" \
  "$authenticator_argument" "$partial" >/dev/null 2>&1; then
  fail 'injected partial-copy failure unexpectedly succeeded'
fi
[ ! -e "$partial" ] || fail 'injected partial-copy failure published an output'
[ -z "$(find "$fixture" -maxdepth 1 -name '.lunaflux-final-release-stage.*' -print)" ] ||
  fail 'injected partial-copy failure retained a stage'

fake_mv_bin=$fixture/fake-mv-bin
mkdir "$fake_mv_bin"
mv_state=$fixture/fake-mv-state
cat > "$fake_mv_bin/mv" <<EOF
#!/bin/sh
count=0
[ ! -f "$mv_state" ] || count=\$(sed -n '1p' "$mv_state")
count=\$((count + 1))
printf '%s\n' "\$count" > "$mv_state"
[ "\$count" -ne 3 ] || exit 74
exec /bin/mv "\$@"
EOF
chmod 755 "$fake_mv_bin/mv"
partial_publication=$fixture/partial-publication-output
if PATH="$fake_mv_bin:$PATH" "$assembler" "$recipe#sha256=$recipe_sha" \
  "$authenticator_argument" "$partial_publication" >/dev/null 2>&1; then
  fail 'injected partial-publication failure unexpectedly succeeded'
fi
[ ! -e "$partial_publication" ] ||
  fail 'injected partial-publication failure retained the claimed output'
[ -z "$(find "$fixture" -maxdepth 1 -name '.lunaflux-final-release-stage.*' -print)" ] ||
  fail 'injected partial-publication failure retained a stage'

changed=$fixture/changed
cp -R "$valid" "$changed"
chmod 755 "$changed/artifacts"
chmod 644 "$changed/artifacts/sbom.json"
printf '%s\n' substituted >> "$changed/artifacts/sbom.json"
chmod 444 "$changed/artifacts/sbom.json"
chmod 555 "$changed/artifacts"
assert_verifier_rejected substituted-output "$changed" "$authenticator_argument"

linked=$fixture/linked
cp -R "$valid" "$linked"
chmod 755 "$linked/artifacts"
rm "$linked/artifacts/sbom.json"
ln -s "$linked/artifacts/provenance.json" "$linked/artifacts/sbom.json"
chmod 555 "$linked/artifacts"
assert_verifier_rejected symlink-output "$linked" "$authenticator_argument"

alternate_authenticator=$fixture/alternate-authenticator
cp "$authenticator" "$alternate_authenticator"
printf '%s\n' '# distinct unapproved tool identity' >> "$alternate_authenticator"
chmod 755 "$alternate_authenticator"
alternate_argument=$alternate_authenticator#sha256=$(sha256_file "$alternate_authenticator")
assert_verifier_rejected authenticator-substitution "$valid" "$alternate_argument"

printf '%s\n' 'LunaFlux final release inventory deterministic and hostile-input gates passed.'
