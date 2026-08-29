#!/bin/sh

# Fail-closed helpers shared by explicit release-materialization recovery and
# the materializer that creates its authenticated transaction claim.

materialization_order='bundle.files.sha256 evidence launch-root model-root oci-context policy-root'
materialization_order_csv='bundle.files.sha256,evidence,launch-root,model-root,oci-context,policy-root'

materialization_require_owned_regular() {
  mrr_path=$1
  mrr_mode=$2
  [ -f "$mrr_path" ] && [ ! -L "$mrr_path" ] ||
    bundle_fail "transaction metadata is not a regular non-symlink file: $mrr_path"
  bundle_require_mode "$mrr_path" "$mrr_mode"
  [ "$(bundle_file_owner_uid "$mrr_path")" = "$materialization_owner_uid" ] ||
    bundle_fail "transaction metadata has the wrong owner: $mrr_path"
  [ "$(bundle_file_link_count "$mrr_path")" = 1 ] ||
    bundle_fail "transaction metadata has an ambiguous hard-link identity: $mrr_path"
}

materialization_write_claim() {
  mwc_claim=$1
  cat > "$mwc_claim" <<EOF
schema=lunaflux.release-materialization-claim.v2
state=claimed
output_path=$output
owner_uid=$materialization_owner_uid
assembly_input_sha256=$input_sha
semantic_preflight_tool_sha256=$tool_sha
materializer_sha256=$materializer_sha
publication_order=$materialization_order_csv
EOF
  chmod 400 "$mwc_claim"
}

materialization_read_claim() {
  mrc_claim=$1
  materialization_require_owned_regular "$mrc_claim" 400
  [ "$(wc -l < "$mrc_claim" | tr -d ' ')" -eq 8 ] ||
    bundle_fail 'materialization claim must contain exactly 8 lines'
  [ "$(bundle_evidence_value "$mrc_claim" 1 schema)" = \
    lunaflux.release-materialization-claim.v2 ] ||
    bundle_fail 'unsupported materialization claim schema'
  [ "$(bundle_evidence_value "$mrc_claim" 2 state)" = claimed ] ||
    bundle_fail 'materialization claim state is invalid'
  [ "$(bundle_evidence_value "$mrc_claim" 3 output_path)" = "$output" ] ||
    bundle_fail 'materialization claim is bound to another output'
  [ "$(bundle_evidence_value "$mrc_claim" 4 owner_uid)" = \
    "$materialization_owner_uid" ] ||
    bundle_fail 'materialization claim is bound to another owner'
  claim_input_sha=$(bundle_evidence_value "$mrc_claim" 5 assembly_input_sha256)
  claim_tool_sha=$(bundle_evidence_value "$mrc_claim" 6 semantic_preflight_tool_sha256)
  claim_materializer_sha=$(bundle_evidence_value "$mrc_claim" 7 materializer_sha256)
  for mrc_digest in "$claim_input_sha" "$claim_tool_sha" "$claim_materializer_sha"; do
    bundle_is_lower_sha256 "$mrc_digest" ||
      bundle_fail 'materialization claim contains an invalid digest'
  done
  [ "$(bundle_evidence_value "$mrc_claim" 8 publication_order)" = \
    "$materialization_order_csv" ] ||
    bundle_fail 'materialization claim publication order is invalid'
}

materialization_write_prepared() {
  mwp_file=$1
  mwp_bundle=$2
  mwp_manifest_sha=$(bundle_sha256_file "$mwp_bundle/bundle.files.sha256")
  mwp_deployment_sha=$(bundle_sha256_file "$mwp_bundle/evidence/deployment-bundle.v1")
  mwp_semantic_sha=$(bundle_sha256_file "$mwp_bundle/evidence/release-materialization-preflight.v1")
  cat > "$mwp_file" <<EOF
schema=lunaflux.release-materialization-prepared.v1
output_path=$output
owner_uid=$materialization_owner_uid
bundle_inventory_sha256=$mwp_manifest_sha
deployment_bundle_sha256=$mwp_deployment_sha
semantic_preflight_evidence_sha256=$mwp_semantic_sha
semantic_preflight_tool_sha256=$claim_tool_sha
materializer_sha256=$claim_materializer_sha
EOF
  chmod 400 "$mwp_file"
}

materialization_read_prepared() {
  mrp_file=$1
  materialization_require_owned_regular "$mrp_file" 400
  [ "$(wc -l < "$mrp_file" | tr -d ' ')" -eq 8 ] ||
    bundle_fail 'prepared record must contain exactly 8 lines'
  [ "$(bundle_evidence_value "$mrp_file" 1 schema)" = \
    lunaflux.release-materialization-prepared.v1 ] ||
    bundle_fail 'unsupported prepared record schema'
  [ "$(bundle_evidence_value "$mrp_file" 2 output_path)" = "$output" ] ||
    bundle_fail 'prepared record is bound to another output'
  [ "$(bundle_evidence_value "$mrp_file" 3 owner_uid)" = \
    "$materialization_owner_uid" ] ||
    bundle_fail 'prepared record is bound to another owner'
  prepared_manifest_sha=$(bundle_evidence_value "$mrp_file" 4 bundle_inventory_sha256)
  prepared_deployment_sha=$(bundle_evidence_value "$mrp_file" 5 deployment_bundle_sha256)
  prepared_semantic_sha=$(bundle_evidence_value "$mrp_file" 6 semantic_preflight_evidence_sha256)
  prepared_tool_sha=$(bundle_evidence_value "$mrp_file" 7 semantic_preflight_tool_sha256)
  prepared_materializer_sha=$(bundle_evidence_value "$mrp_file" 8 materializer_sha256)
  for mrp_digest in "$prepared_manifest_sha" "$prepared_deployment_sha" \
    "$prepared_semantic_sha" "$prepared_tool_sha" "$prepared_materializer_sha"; do
    bundle_is_lower_sha256 "$mrp_digest" ||
      bundle_fail 'prepared record contains an invalid digest'
  done
  [ "$prepared_tool_sha" = "$claim_tool_sha" ] &&
    [ "$prepared_materializer_sha" = "$claim_materializer_sha" ] ||
    bundle_fail 'prepared record does not bind the exact claim'
}

materialization_require_owned_tree() {
  if find "$output" ! -user "$materialization_owner_uid" -print | grep -q .; then
    bundle_fail 'materialization transaction contains an object owned by another uid'
  fi
  if find "$output" -type l -print | grep -q .; then
    bundle_fail 'materialization transaction contains a symbolic link'
  fi
  if find "$output" ! -type d ! -type f -print | grep -q .; then
    bundle_fail 'materialization transaction contains a special filesystem object'
  fi
}

materialization_make_split_view() {
  msv_stage=$1
  msv_view=$2
  mkdir "$msv_view"
  for msv_entry in $materialization_order; do
    if [ -e "$output/$msv_entry" ]; then
      msv_source=$output/$msv_entry
    else
      msv_source=$msv_stage/$msv_entry
    fi
    if [ -d "$msv_source" ]; then
      find "$msv_source" -type d -print | while IFS= read -r msv_directory; do
        msv_relative=${msv_directory#"$msv_source"}
        case "$msv_relative" in '') ;; /*) ;;
          *) bundle_fail 'split bundle directory has an invalid relative path' ;;
        esac
        mkdir -p "$msv_view/$msv_entry$msv_relative"
      done
      find "$msv_source" -type f -print | while IFS= read -r msv_file; do
        msv_relative=${msv_file#"$msv_source"/}
        bundle_is_strict_relative "$msv_relative" ||
          bundle_fail 'split bundle file has an invalid relative path'
        msv_target=$msv_view/$msv_entry/$msv_relative
        if [ "$msv_entry" = oci-context ]; then
          cp "$msv_file" "$msv_target" ||
            bundle_fail 'could not copy the private OCI verification view'
          chmod "$(bundle_file_mode "$msv_file")" "$msv_target" ||
            bundle_fail 'could not preserve private OCI split-view mode'
        else
          ln "$msv_file" "$msv_target" ||
            bundle_fail 'could not construct a private split-verification view'
        fi
      done
    else
      msv_target=$msv_view/$msv_entry
      ln "$msv_source" "$msv_target" ||
        bundle_fail 'could not construct a private split-verification view'
    fi
  done
  find "$msv_view" -type d -exec chmod 555 {} \;
}

materialization_validate_split() {
  mvs_stage=$1
  mvs_missing=0
  materialization_prefix_count=0
  for mvs_entry in $materialization_order; do
    mvs_at_output=0
    mvs_at_stage=0
    [ ! -e "$output/$mvs_entry" ] || mvs_at_output=1
    [ ! -e "$mvs_stage/$mvs_entry" ] || mvs_at_stage=1
    [ "$mvs_at_output" -ne "$mvs_at_stage" ] ||
      bundle_fail "publication entry is missing or duplicated: $mvs_entry"
    if [ "$mvs_at_output" -eq 1 ]; then
      [ "$mvs_missing" -eq 0 ] ||
        bundle_fail 'published entries are not an exact prefix'
      materialization_prefix_count=$((materialization_prefix_count + 1))
    else
      mvs_missing=1
    fi
  done
  mvs_view=$BUNDLE_SCRATCH_DIR/split-view
  materialization_make_split_view "$mvs_stage" "$mvs_view"
  "$repo_root/scripts/verify-materialized-release-bundle.sh" "$mvs_view" >/dev/null
  split_manifest_sha=$(bundle_sha256_file "$mvs_view/bundle.files.sha256")
  split_deployment_sha=$(bundle_sha256_file "$mvs_view/evidence/deployment-bundle.v1")
  split_semantic_sha=$(bundle_sha256_file "$mvs_view/evidence/release-materialization-preflight.v1")
  split_transaction=$mvs_view/evidence/materialization.v1
  split_tool_sha=$(bundle_evidence_value "$split_transaction" 3 semantic_preflight_tool_sha256)
  split_materializer_sha=$(bundle_evidence_value "$split_transaction" 4 materializer_sha256)
  [ "$split_tool_sha" = "$claim_tool_sha" ] &&
    [ "$split_materializer_sha" = "$claim_materializer_sha" ] ||
    bundle_fail 'split bundle transaction does not bind the exact claim'
}
