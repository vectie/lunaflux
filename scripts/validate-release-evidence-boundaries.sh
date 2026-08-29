#!/bin/sh

set -eu

package=release/evidence

if rg -n 'LunaNexa|lunanexa|moongate|python|pytorch|tvm' "$package" \
  --glob '*.mbt' --glob 'moon.pkg'; then
  printf '%s\n' 'release evidence imports or names a foreign product/runtime' >&2
  exit 1
fi

imports=$(sed -n '/^import {/,/^}/p' "$package/moon.pkg" | \
  rg '"' | sed 's/^[[:space:]]*//')
expected='"moonbitlang/core/encoding/utf8",
"moonbitlang/x/crypto",'
[ "$imports" = "$expected" ] || {
  printf '%s\n' 'release evidence dependency surface changed' >&2
  exit 1
}

for role in \
  PhysicalTarget ReferenceCorrectness NativeLeak StructuralBoundary \
  Performance SecurityScan OciImage Sbom BuildProvenance LicenseInventory \
  KernelManifest RuntimeBuild ExternalAdapterBuild DirectPath \
  ExternalAdapterPath OperationalLifecycle; do
  count=$(rg -n "^[[:space:]]*$role$" "$package/enums.mbt" | wc -l | tr -d ' ')
  [ "$count" -eq 1 ] || {
    printf 'release evidence role is missing or duplicated: %s\n' "$role" >&2
    exit 1
  }
done

for role in SecurityReviewer OperationsReviewer PerformanceReviewer; do
  count=$(rg -n "^[[:space:]]*$role$" "$package/enums.mbt" | wc -l | tr -d ' ')
  [ "$count" -eq 1 ] || {
    printf 'release reviewer role is missing or duplicated: %s\n' "$role" >&2
    exit 1
  }
done

if ! rg -q 'identity\.is_none|identity is Some|identity\.is_some' \
    "$package/entries.mbt" || \
  ! rg -q 'attestation is Some' "$package/entries.mbt"; then
  printf '%s\n' 'terminal reviewer identity/attestation binding is missing' >&2
  exit 1
fi

if rg -n '^(let|const)[[:space:]]' "$package" --glob '*.mbt'; then
  printf '%s\n' 'release evidence must not add package-global state' >&2
  exit 1
fi

if ! rg -q '^pub struct VerifiedReleaseCampaign \{' \
    "$package/verification.mbt" || \
  ! rg -q '^[[:space:]]+priv campaign_digest : ReleaseCampaignDigest$' \
    "$package/verification.mbt" || \
  ! rg -q '^[[:space:]]+priv status : ReleaseEvidenceStatus$' \
    "$package/verification.mbt" || \
  ! rg -U -q \
    '^pub fn verify_release_campaign\((?s:.*?)\) -> VerifiedReleaseCampaign raise ReleaseEvidenceError \{' \
    "$package/verification.mbt" || \
  rg -n '^pub fn VerifiedReleaseCampaign::(new|make|create|from_)' \
    "$package/verification.mbt"; then
  printf '%s\n' 'verified release verdict is no longer an opaque capability' >&2
  exit 1
fi

if rg -n '^pub fn ReleaseCampaignRecord::(is_accepted|accepted|verdict)' \
    "$package" --glob '*.mbt'; then
  printf '%s\n' 'a declared campaign state leaked as a verified verdict' >&2
  exit 1
fi

if ! rg -q 'priv locator : String' "$package/verification.mbt" || \
  ! rg -q 'artifact\.locator\(\) != locator' "$package/verification.mbt" || \
  ! rg -q 'digests\.contains\(artifact\.digest\(\)\)' \
    "$package/canonical_writer.mbt" || \
  ! rg -q 'digests\.contains\(attestation\.digest\(\)\)' \
    "$package/canonical_writer.mbt"; then
  printf '%s\n' 'exact locator or cross-role content uniqueness binding is missing' >&2
  exit 1
fi

interface="$package/pkg.generated.mbti"
if [ -f "$interface" ]; then
  if ! rg -U -q \
      'pub struct VerifiedReleaseCampaign \{\n  // private fields\n\}' \
      "$interface" || \
    ! rg -q '^pub fn VerifiedReleaseCampaign::campaign_digest\(' \
      "$interface" || \
    ! rg -q '^pub fn VerifiedReleaseCampaign::status\(' "$interface" || \
    ! rg -q '^pub fn VerifiedReleaseCampaign::is_accepted\(' "$interface" || \
    rg -n '^pub fn VerifiedReleaseCampaign::(new|make|create|from_)' \
      "$interface"; then
    printf '%s\n' 'generated release verdict interface is not opaque and exact' >&2
    exit 1
  fi
fi

for file in "$package"/*.mbt; do
  lines=$(wc -l < "$file" | tr -d ' ')
  [ "$lines" -le 500 ] || {
    printf 'release evidence file exceeds 500 lines: %s (%s)\n' "$file" "$lines" >&2
    exit 1
  }
done

printf '%s\n' 'release evidence boundaries are narrow and complete.'
