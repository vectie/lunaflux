#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)
cd "$repo_root"

fail() {
  printf '%s\n' "FP8 release authority boundary: $1" >&2
  exit 1
}

package=kernels/fp8_release_authority
interface="$package/pkg.generated.mbti"

for anchor in \
  '@approved_fs.ApprovedRoot' \
  '@approved_fs.ApprovedRelativeLocator' \
  '@luna_capability_manifest.LunaAdmittedFp8ReleaseManifestV2' \
  '@fp8_runtime_recipe.Fp8RuntimeRecipeV2' \
  'read_immutable_snapshot' \
  'module_bytes != artifact_module.module_bytes()' \
  'release_sha256(module_bytes) != module_digest'; do
  rg -F -q "$anchor" "$package" --glob '*.mbt' || \
    fail "missing trust-chain invariant: $anchor"
done

if rg -n 'CompileReceipt|CompileEvidence|compile_only|Compiler' \
  "$package" --glob '*.mbt' --glob 'moon.pkg'; then
  fail 'caller-produced compiler evidence crossed executable admission'
fi

rg -F -x -q \
  'pub fn admit_v2(@approved_fs.ApprovedRoot, @approved_fs.ApprovedRelativeLocator, @luna_capability_manifest.LunaAdmittedFp8ReleaseManifestV2, @fp8_runtime_recipe.Fp8RuntimeRecipeV2, Fp8ReleaseAuthorityLimits) -> Fp8ReleaseAuthorityV2 raise Fp8ReleaseAuthorityError' \
  "$interface" || fail 'release admission API drifted'

rg -F -x -q \
  'pub fn admit_reusable_paged_v3(@approved_fs.ApprovedRoot, @approved_fs.ApprovedRelativeLocator, @luna_capability_manifest.LunaAdmittedFp8ReusablePagedReleaseManifestV3, @fp8_runtime_recipe.Fp8ReusablePagedRecipeV3, Fp8ReleaseAuthorityLimits) -> Fp8ReusablePagedEnvelopeAuthorityV3 raise Fp8ReleaseAuthorityError' \
  "$interface" || fail 'reusable paged v3 release admission API drifted'

for source in "$package"/*.mbt; do
  lines=$(wc -l < "$source" | tr -d ' ')
  [ "$lines" -lt 500 ] || fail "$source exceeds the 499-line budget"
done

moon check "$package" --target native --deny-warn --warn-list +73
moon test "$package" --target native --deny-warn --warn-list +73
printf '%s\n' 'FP8 release authority boundary validation passed.'
