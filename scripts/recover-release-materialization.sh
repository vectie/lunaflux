#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL
umask 077

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/release-bundle-common.sh"
. "$repo_root/scripts/release-materialization-recovery-common.sh"

[ "$#" -eq 1 ] || {
  printf '%s\n' 'usage: recover-release-materialization.sh ABSOLUTE_CLAIMED_OUTPUT' >&2
  exit 2
}
output=$1
bundle_require_canonical_absolute_directory "$output"
materialization_owner_uid=$(id -u)
BUNDLE_SCRATCH_DIR=$(mktemp -d "$(dirname -- "$output")/.lunaflux-recovery.XXXXXX") ||
  bundle_fail 'could not create same-filesystem recovery scratch'
export BUNDLE_SCRATCH_DIR
cleanup_recovery_scratch() {
  find "$BUNDLE_SCRATCH_DIR" -type d -exec chmod u+w {} \; 2>/dev/null || true
  rm -rf "$BUNDLE_SCRATCH_DIR"
}
trap cleanup_recovery_scratch EXIT HUP INT TERM

claim=$output/.materialization-claim
materialization_require_owned_tree
materialization_read_claim "$claim"

top_entries=$(find "$output" -mindepth 1 -maxdepth 1 -print |
  sed "s#^$output/##" | LC_ALL=C sort)
case "$top_entries" in
  .materialization-claim)
    rm "$claim"
    rmdir "$output"
    cleanup_recovery_scratch
    trap - EXIT HUP INT TERM
    printf '%s\n' 'LunaFlux empty materialization claim recovered; output released.'
    exit 0
    ;;
esac

for top_entry in $top_entries; do
  case "$top_entry" in
    .bundle|.materialization-claim|.materialization-prepared|bundle.files.sha256|evidence|launch-root|model-root|oci-context|policy-root) ;;
    *) bundle_fail "materialization transaction has an unknown top-level entry: $top_entry" ;;
  esac
done

stage=$output/.bundle
if [ -e "$stage" ]; then
  [ -d "$stage" ] && [ ! -L "$stage" ] ||
    bundle_fail 'materialization stage is not a real directory'
else
  mkdir "$BUNDLE_SCRATCH_DIR/empty-stage"
  stage=$BUNDLE_SCRATCH_DIR/empty-stage
fi
materialization_validate_split "$stage"

prepared=$output/.materialization-prepared
if [ -e "$prepared" ]; then
  materialization_read_prepared "$prepared"
  [ "$prepared_manifest_sha" = "$split_manifest_sha" ] &&
    [ "$prepared_deployment_sha" = "$split_deployment_sha" ] &&
    [ "$prepared_semantic_sha" = "$split_semantic_sha" ] ||
    bundle_fail 'prepared record does not bind the exact split bundle'
elif [ "$materialization_prefix_count" -eq 0 ] ||
  [ "$materialization_prefix_count" -eq 6 ]; then
  prepared_next=$BUNDLE_SCRATCH_DIR/materialization-prepared
  materialization_write_prepared "$prepared_next" "$BUNDLE_SCRATCH_DIR/split-view"
  mv "$prepared_next" "$prepared"
else
  bundle_fail 'partial publication has no authenticated prepared record'
fi

for entry in $materialization_order; do
  [ -e "$output/$entry" ] || mv "$stage/$entry" "$output/$entry"
done
if [ -e "$output/.bundle" ]; then rmdir "$output/.bundle"; fi
"$repo_root/scripts/verify-materialized-release-bundle.sh" \
  "$BUNDLE_SCRATCH_DIR/split-view" >/dev/null
rm "$prepared"
rm "$claim"
find "$output" -type d -exec chmod 555 {} \;
chmod 555 "$output"
"$repo_root/scripts/verify-materialized-release-bundle.sh" "$output" >/dev/null
cleanup_recovery_scratch
trap - EXIT HUP INT TERM
printf '%s\n' "LunaFlux release materialization recovered and verified: $output"
