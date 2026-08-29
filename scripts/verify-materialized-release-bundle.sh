#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/release-bundle-common.sh"
BUNDLE_SCRATCH_DIR=$(mktemp -d /tmp/lunaflux-materialized-verify.XXXXXX) ||
  bundle_fail 'could not create private materialized-verifier scratch'
export BUNDLE_SCRATCH_DIR
trap 'rm -rf "$BUNDLE_SCRATCH_DIR"' EXIT HUP INT TERM

[ "$#" -eq 1 ] || {
  printf '%s\n' 'usage: verify-materialized-release-bundle.sh ABSOLUTE_BUNDLE_ROOT' >&2
  exit 2
}
bundle=$1
"$repo_root/scripts/verify-release-bundle.sh" "$bundle" >/dev/null

semantic=$bundle/evidence/release-materialization-preflight.v1
transaction=$bundle/evidence/materialization.v1
[ -f "$semantic" ] && [ ! -L "$semantic" ] ||
  bundle_fail 'materialized bundle has no semantic preflight evidence'
[ -f "$transaction" ] && [ ! -L "$transaction" ] ||
  bundle_fail 'materialized bundle has no transaction evidence'
bundle_require_mode "$semantic" 444
bundle_require_mode "$transaction" 444
bundle_require_newline_terminated "$semantic" 'semantic preflight evidence'
bundle_require_newline_terminated "$transaction" 'materialization transaction evidence'

[ "$(wc -l < "$semantic" | tr -d ' ')" -eq 20 ] ||
  bundle_fail 'semantic preflight evidence must contain exactly 20 lines'
[ -z "$(sed -n '21p' "$semantic")" ] ||
  bundle_fail 'semantic preflight evidence has trailing fields'
[ "$(bundle_evidence_value "$semantic" 1 schema)" = \
  lunaflux-release-materialization-preflight.v1 ] ||
  bundle_fail 'unsupported semantic preflight evidence schema'
runtime_recipe=$(bundle_evidence_value "$semantic" 2 runtime_recipe)
case "$runtime_recipe" in
  dense_llama_paged_aot_v5|dense_llama_i8_paged_aot_v6) ;;
  *) bundle_fail 'semantic preflight has unsupported runtime recipe' ;;
esac
deployment_sha=$(bundle_evidence_value "$semantic" 3 deployment_bundle_sha256)
launch_sha=$(bundle_evidence_value "$semantic" 4 launch_sha256)
descriptor_sha=$(bundle_evidence_value "$semantic" 5 runtime_descriptor_sha256)
policy_sha=$(bundle_evidence_value "$semantic" 6 instance_policy_sha256)
tokenizer_sha=$(bundle_evidence_value "$semantic" 7 tokenizer_sha256)
worker_sha=$(bundle_evidence_value "$semantic" 8 worker_executable_sha256)
model_content_sha=$(bundle_evidence_value "$semantic" 9 model_content_sha256)
model_plan_sha=$(bundle_evidence_value "$semantic" 10 model_plan_sha256)
bootstrap_sha=$(bundle_evidence_value "$semantic" 11 bootstrap_sha256)
bootstrap_source_sha=$(bundle_evidence_value "$semantic" 12 bootstrap_source_sha256)
for digest in "$deployment_sha" "$launch_sha" "$descriptor_sha" "$policy_sha" \
  "$tokenizer_sha" "$worker_sha" "$model_content_sha" "$model_plan_sha" \
  "$bootstrap_sha" "$bootstrap_source_sha"; do
  bundle_is_lower_sha256 "$digest" ||
    bundle_fail 'semantic preflight evidence contains an invalid digest'
done
for line in 13 14 15; do
  value=$(sed -n "${line}p" "$semantic")
  case "$value" in
    device_ordinal=*|compute_major=*|compute_minor=*) ;;
    *) bundle_fail 'semantic preflight device fields are noncanonical' ;;
  esac
  number=${value#*=}
  case "$number" in ''|*[!0-9]*) bundle_fail 'semantic device value is invalid' ;; esac
done
[ "$(bundle_evidence_value "$semantic" 16 source_target_binding)" = 1 ] &&
  [ "$(bundle_evidence_value "$semantic" 17 semantic_join)" = 1 ] &&
  [ "$(bundle_evidence_value "$semantic" 18 filesystem_authority_closed)" = 1 ] &&
  [ "$(bundle_evidence_value "$semantic" 19 device_opened)" = 0 ] &&
  [ "$(bundle_evidence_value "$semantic" 20 compiler_jit_authority)" = 0 ] ||
  bundle_fail 'semantic preflight authority flags are invalid'
if grep -q '/' "$semantic" || grep -E -i -q 'nvrtc|runtime[_-]?jit|developer[_-]?jit|\.ptx' "$semantic"; then
  bundle_fail 'semantic preflight evidence contains path or JIT payload'
fi

deployment=$bundle/evidence/deployment-bundle.v1
[ "$(bundle_sha256_file "$deployment")" = "$deployment_sha" ] ||
  bundle_fail 'semantic evidence does not bind this deployment bundle'
[ "$(bundle_evidence_value "$deployment" 4 launch_file_sha256)" = "$launch_sha" ] ||
  bundle_fail 'semantic launch digest disagrees with deployment evidence'
[ "$(bundle_evidence_value "$deployment" 6 runtime_descriptor_sha256)" = "$descriptor_sha" ] ||
  bundle_fail 'semantic descriptor digest disagrees with deployment evidence'
[ "$(bundle_evidence_value "$deployment" 8 instance_policy_sha256)" = "$policy_sha" ] ||
  bundle_fail 'semantic policy digest disagrees with deployment evidence'
[ "$(bundle_evidence_value "$deployment" 10 worker_executable_sha256)" = "$worker_sha" ] ||
  bundle_fail 'semantic worker digest disagrees with deployment evidence'

[ "$(wc -l < "$transaction" | tr -d ' ')" -eq 8 ] ||
  bundle_fail 'materialization transaction evidence must contain exactly 8 lines'
[ -z "$(sed -n '9p' "$transaction")" ] ||
  bundle_fail 'materialization transaction evidence has trailing fields'
[ "$(bundle_evidence_value "$transaction" 1 schema)" = \
  lunaflux.release-materialization.v1 ] ||
  bundle_fail 'unsupported materialization transaction schema'
[ "$(bundle_evidence_value "$transaction" 2 deployment_bundle_sha256)" = "$deployment_sha" ] ||
  bundle_fail 'transaction does not bind deployment evidence'
tool_sha=$(bundle_evidence_value "$transaction" 3 semantic_preflight_tool_sha256)
materializer_sha=$(bundle_evidence_value "$transaction" 4 materializer_sha256)
semantic_sha=$(bundle_evidence_value "$transaction" 5 semantic_preflight_evidence_sha256)
for digest in "$tool_sha" "$materializer_sha" "$semantic_sha"; do
  bundle_is_lower_sha256 "$digest" ||
    bundle_fail 'transaction evidence contains an invalid digest'
done
[ "$(bundle_sha256_file "$semantic")" = "$semantic_sha" ] ||
  bundle_fail 'transaction semantic evidence digest mismatch'
[ "$(bundle_evidence_value "$transaction" 6 source_target_binding)" = 1 ] &&
  [ "$(bundle_evidence_value "$transaction" 7 semantic_join)" = 1 ] &&
  [ "$(bundle_evidence_value "$transaction" 8 no_overwrite)" = 1 ] ||
  bundle_fail 'materialization transaction policy is invalid'

rm -rf "$BUNDLE_SCRATCH_DIR"
trap - EXIT HUP INT TERM
printf '%s\n' 'LunaFlux materialized release bundle is exact and semantically preflighted.'
