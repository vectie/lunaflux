#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
package_dir=$repo_root/release/deployment_bundle
assembler=$repo_root/scripts/assemble-release-bundle.sh
verifier=$repo_root/scripts/verify-release-bundle.sh
common=$repo_root/scripts/release-bundle-common.sh
materializer=$repo_root/scripts/materialize-release-bundle.sh
recovery=$repo_root/scripts/recover-release-materialization.sh
recovery_common=$repo_root/scripts/release-materialization-recovery-common.sh
materialized_verifier=$repo_root/scripts/verify-materialized-release-bundle.sh
materialized_gate=$repo_root/scripts/validate-release-materialization.sh

fail() {
  printf '%s\n' "LunaFlux deployment-bundle boundary gate failed: $1" >&2
  exit 1
}

[ -f "$package_dir/moon.pkg" ] || fail 'focused MoonBit package is missing'
grep -F -x 'supported_targets = "native"' "$package_dir/moon.pkg" >/dev/null ||
  fail 'deployment-bundle package is not native-only'
for import in moonbitlang/core/encoding/utf8 moonbitlang/x/crypto \
  vectie/lunaflux/release/evidence; do
  grep -F "\"$import\"" "$package_dir/moon.pkg" >/dev/null ||
    fail "required focused import is missing: $import"
done
if grep -E 'vectie/lunaflux/(deploy|runtime|engine|device|kernels|model|scheduler)/' \
  "$package_dir/moon.pkg" >/dev/null 2>&1; then
  fail 'inert bundle evidence imports runtime or execution implementation'
fi
if rg -n 'LunaNexa|MoonGate|MoonSuite|python|pytorch|torch|tvm' \
  "$package_dir" "$assembler" "$verifier" "$common" "$materializer" \
  "$recovery" "$recovery_common" "$materialized_verifier" \
  "$materialized_gate" >/dev/null 2>&1; then
  fail 'bundle implementation crosses a product or forbidden-runtime boundary'
fi

for script in "$assembler" "$verifier" "$common" "$materializer" \
  "$recovery" "$recovery_common" "$materialized_verifier" \
  "$materialized_gate"; do sh -n "$script"; done
grep -F '[ ! -e "$output" ]' "$assembler" >/dev/null ||
  fail 'assembler does not refuse an existing output'
grep -F 'mkdir "$output"' "$assembler" >/dev/null ||
  fail 'assembler does not atomically claim a new output directory'
grep -F '"$repo_root/scripts/verify-release-bundle.sh" "$stage"' "$assembler" >/dev/null ||
  fail 'assembler does not verify the complete stage before transfer'
transfer_line=$(grep -n '^for entry in bundle.files.sha256 evidence launch-root model-root oci-context policy-root; do$' \
  "$assembler" | cut -d: -f1)
claim_remove_line=$(grep -n '^rm "\$output/.assembly-claim"$' "$assembler" | cut -d: -f1)
[ -n "$transfer_line" ] && [ -n "$claim_remove_line" ] &&
  [ "$claim_remove_line" -gt "$transfer_line" ] ||
  fail 'assembly cleanup claim is removed before staged transfer completes'
grep -F 'bundle_validate_inventory "$model_inventory" "$model_root" model' \
  "$assembler" >/dev/null || fail 'model exact-inventory gate is missing'
grep -F 'bundle_validate_inventory "$kernel_inventory" "$kernel_root" kernel' \
  "$assembler" >/dev/null || fail 'kernel exact-inventory gate is missing'
grep -F '"$repo_root/scripts/verify-oci-context.sh" "$base_image" "$bundle/oci-context"' \
  "$verifier" >/dev/null || fail 'bundle verifier does not compose the OCI verifier'

grep -F 'mkdir "$output"' "$materializer" >/dev/null ||
  fail 'materializer does not exclusively claim a new output'
grep -F '"$tool" validate-materialized-release' "$materializer" >/dev/null ||
  fail 'materializer does not invoke the typed semantic preflight command'
semantic_line=$(grep -n '^if ! "\$tool" validate-materialized-release' \
  "$materializer" | cut -d: -f1)
install_line=$(grep -n '^mv "\$stage" "\$output/.bundle"$' \
  "$materializer" | cut -d: -f1)
recovery_line=$(grep -n '^"\$repo_root/scripts/recover-release-materialization.sh" "\$output"' \
  "$materializer" | cut -d: -f1)
[ -n "$semantic_line" ] && [ -n "$install_line" ] &&
  [ -n "$recovery_line" ] && [ "$semantic_line" -lt "$install_line" ] &&
  [ "$install_line" -lt "$recovery_line" ] ||
  fail 'semantic preflight or publication can occur outside the cleanup claim'
grep -F 'source_target_binding=1' "$materializer" >/dev/null &&
  grep -F 'no_overwrite=1' "$materializer" >/dev/null ||
  fail 'materialization transaction evidence lost typed/no-overwrite flags'
grep -F '"$repo_root/scripts/verify-materialized-release-bundle.sh" "$stage"' \
  "$materializer" >/dev/null ||
  fail 'materializer does not verify semantic evidence before publication'
grep -F '"$repo_root/scripts/recover-release-materialization.sh" "$output"' \
  "$materializer" >/dev/null ||
  fail 'materializer does not use the explicit resumable publication protocol'
grep -F 'materialization_validate_split "$stage"' "$recovery" >/dev/null &&
  grep -F 'published entries are not an exact prefix' "$recovery_common" >/dev/null ||
  fail 'recovery does not authenticate an exact prefix publication state'
grep -F 'owner_uid=$materialization_owner_uid' "$recovery_common" >/dev/null &&
  grep -F 'output_path=$output' "$recovery_common" >/dev/null ||
  fail 'recovery claim does not bind current uid and canonical output'
if grep -F 'rm -rf "$output"' "$materializer" "$recovery" \
  "$recovery_common" >/dev/null 2>&1; then
  fail 'materialization cleanup can recursively delete an output namespace'
fi
grep -F 'rmdir "$output"' "$recovery" >/dev/null ||
  fail 'empty CLAIMED recovery is not limited to empty-directory removal'
grep -F '"$repo_root/scripts/verify-release-bundle.sh" "$bundle"' \
  "$materialized_verifier" >/dev/null ||
  fail 'materialized verifier no longer composes the exact bundle verifier'

for file in "$package_dir"/*.mbt "$assembler" "$verifier" "$common" \
  "$materializer" "$recovery" "$recovery_common" "$materialized_verifier" \
  "$materialized_gate"; do
  lines=$(wc -l < "$file" | tr -d ' ')
  [ "$lines" -lt 500 ] || fail "file exceeds the 500-line cohesion budget: $file"
done

printf '%s\n' 'LunaFlux deployment-bundle dependency and static boundaries are valid.'
