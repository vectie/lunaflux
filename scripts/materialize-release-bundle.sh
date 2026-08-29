#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL
umask 077

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/release-bundle-common.sh"
. "$repo_root/scripts/release-materialization-recovery-common.sh"

usage() {
  printf '%s\n' \
    'usage: materialize-release-bundle.sh ABSOLUTE_INPUT#sha256=HEX ABSOLUTE_NEW_OUTPUT ABSOLUTE_PREFLIGHT_TOOL#sha256=HEX' >&2
  exit 2
}

[ "$#" -eq 3 ] || usage
input_argument=$1
output=$2
tool_argument=$3
case "$input_argument" in /*#sha256=*) ;; *) usage ;; esac
input=${input_argument%#sha256=*}
input_sha=${input_argument##*#sha256=}
case "$input" in *#sha256=*) usage ;; esac
bundle_require_canonical_absolute_file "$input"
bundle_require_digest "$input" "$input_sha" 'deployment assembly input'
case "$tool_argument" in /*#sha256=*) ;; *) usage ;; esac
tool=${tool_argument%#sha256=*}
tool_sha=${tool_argument##*#sha256=}
case "$tool" in *#sha256=*) usage ;; esac
bundle_require_canonical_absolute_file "$tool"
bundle_require_digest "$tool" "$tool_sha" 'semantic preflight tool'
[ -x "$tool" ] || bundle_fail 'semantic preflight tool is not executable'

case "$output" in /*) ;; *) bundle_fail 'output path must be absolute' ;; esac
case "$output" in /|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
  bundle_fail 'output path is not a safe canonical absolute path'
  ;;
esac
[ ! -e "$output" ] || bundle_fail 'refusing to overwrite an existing output'
output_parent=$(CDPATH= cd -- "$(dirname -- "$output")" && pwd -P)
[ "$output_parent/$(basename -- "$output")" = "$output" ] ||
  bundle_fail 'output parent is not canonical'
BUNDLE_SCRATCH_DIR=$(mktemp -d "$output_parent/.lunaflux-materialize.XXXXXX") ||
  bundle_fail 'could not create same-filesystem materialization scratch'
export BUNDLE_SCRATCH_DIR

mkdir "$output" || bundle_fail 'could not claim new output path'
materialization_owner_uid=$(id -u)
claim=$output/.materialization-claim
materializer_sha=$(bundle_sha256_file "$repo_root/scripts/materialize-release-bundle.sh")
materialization_write_claim "$claim"
installed=0
cleanup() {
  find "$BUNDLE_SCRATCH_DIR" -type d -exec chmod u+w {} \; 2>/dev/null || true
  rm -rf "$BUNDLE_SCRATCH_DIR"
  if [ "$installed" -eq 0 ] && [ -d "$output" ] && [ ! -L "$output" ] &&
    [ "$(find "$output" -mindepth 1 -maxdepth 1 -print | wc -l | tr -d ' ')" -eq 1 ]; then
    materialization_read_claim "$claim"
    rm "$claim"
    rmdir "$output"
  fi
}
trap cleanup EXIT HUP INT TERM

stage=$BUNDLE_SCRATCH_DIR/bundle
"$repo_root/scripts/assemble-release-bundle.sh" "$input_argument" "$stage" >/dev/null
deployment=$stage/evidence/deployment-bundle.v1
deployment_sha=$(bundle_sha256_file "$deployment")
stdout=$BUNDLE_SCRATCH_DIR/preflight.stdout
stderr=$BUNDLE_SCRATCH_DIR/preflight.stderr
if ! "$tool" validate-materialized-release \
  "$stage#sha256=$deployment_sha" >"$stdout" 2>"$stderr"; then
  bundle_fail 'semantic materialization preflight failed'
fi
[ ! -s "$stderr" ] || bundle_fail 'semantic materialization preflight wrote stderr'
[ "$(wc -l < "$stdout" | tr -d ' ')" -eq 22 ] ||
  bundle_fail 'semantic materialization preflight output is noncanonical'
[ -z "$(sed -n '21p' "$stdout")" ] ||
  bundle_fail 'semantic materialization preflight separator is noncanonical'
digest_line=$(sed -n '22p' "$stdout")
case "$digest_line" in release_materialization_preflight_sha256=*) ;;
  *) bundle_fail 'semantic materialization preflight digest line is missing' ;;
esac
semantic_sha=${digest_line#*=}
bundle_is_lower_sha256 "$semantic_sha" ||
  bundle_fail 'semantic materialization preflight digest is invalid'
semantic=$stage/evidence/release-materialization-preflight.v1
chmod 755 "$stage" "$stage/evidence"
sed -n '1,20p' "$stdout" > "$semantic"
bundle_require_newline_terminated "$semantic" 'semantic preflight evidence'
[ "$(bundle_sha256_file "$semantic")" = "$semantic_sha" ] ||
  bundle_fail 'semantic materialization preflight digest does not match output'
transaction=$stage/evidence/materialization.v1
cat > "$transaction" <<EOF
schema=lunaflux.release-materialization.v1
deployment_bundle_sha256=$deployment_sha
semantic_preflight_tool_sha256=$tool_sha
materializer_sha256=$materializer_sha
semantic_preflight_evidence_sha256=$semantic_sha
source_target_binding=1
semantic_join=1
no_overwrite=1
EOF
chmod 444 "$semantic" "$transaction"
manifest=$stage/bundle.files.sha256
chmod 644 "$manifest"
bundle_write_file_inventory "$stage" "$manifest"
chmod 444 "$manifest"
find "$stage" -type d -exec chmod 555 {} \;
"$repo_root/scripts/verify-materialized-release-bundle.sh" "$stage" >/dev/null

chmod 755 "$stage"
for entry in evidence launch-root model-root oci-context policy-root; do
  chmod 755 "$stage/$entry"
done
mv "$stage" "$output/.bundle"
installed=1
stage=$output/.bundle
claim_input_sha=$input_sha
claim_tool_sha=$tool_sha
claim_materializer_sha=$materializer_sha
prepared_next=$BUNDLE_SCRATCH_DIR/materialization-prepared
materialization_write_prepared "$prepared_next" "$stage"
mv "$prepared_next" "$output/.materialization-prepared"
"$repo_root/scripts/recover-release-materialization.sh" "$output" >/dev/null
trap - EXIT HUP INT TERM
find "$BUNDLE_SCRATCH_DIR" -type d -exec chmod u+w {} \; 2>/dev/null || true
rm -rf "$BUNDLE_SCRATCH_DIR" 2>/dev/null || true
printf '%s\n' "LunaFlux release bundle materialized and preflighted: $output"
printf '%s\n' "semantic_preflight_sha256=$semantic_sha"
