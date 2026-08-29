#!/bin/sh

# Hostile verifier case that recomputes every self-contained seal after
# granting a non-trial helper one credential descriptor. The independently
# supplied verifier must still reject the stored invocation authority.
assert_recomputed_invocation_scope_rejected() {
  voi_fixture=$1
  voi_valid=$2
  voi_verifier=$3
  voi_campaign_argument=$4
  voi_driver_argument=$5
  voi_correctness_argument=$6
  voi_identity_argument=$7
  voi_supervisor_argument=$8
  voi_root=$voi_fixture/scope-substituted
  cp -R "$voi_valid" "$voi_root"
  chmod -R u+w "$voi_root"
  voi_invocation=$voi_root/process/preflight.invocation.v1
  voi_receipt=$voi_root/process/preflight.supervisor.v1
  sed 's/^credential_scope=none$/credential_scope=fd-3/' \
    "$voi_invocation" > "$voi_fixture/scope-invocation.changed"
  mv "$voi_fixture/scope-invocation.changed" "$voi_invocation"
  voi_invocation_sha=$(sha256_file "$voi_invocation")
  sed "s/^invocation_sha256=.*/invocation_sha256=$voi_invocation_sha/" \
    "$voi_receipt" > "$voi_fixture/scope-receipt.changed"
  mv "$voi_fixture/scope-receipt.changed" "$voi_receipt"
  voi_process_paths=$voi_fixture/scope-process-paths
  find "$voi_root/process" -type f -print | sed "s#^$voi_root/##" | \
    LC_ALL=C sort > "$voi_process_paths"
  : > "$voi_root/process.files.sha256"
  while IFS= read -r voi_relative; do
    printf '%s  %s\n' "$(sha256_file "$voi_root/$voi_relative")" \
      "$voi_relative" >> "$voi_root/process.files.sha256"
  done < "$voi_process_paths"
  voi_process_sha=$(sha256_file "$voi_root/process.files.sha256")
  sed "s/^process_inventory_sha256=.*/process_inventory_sha256=$voi_process_sha/" \
    "$voi_root/comparison-handoff.v1" > "$voi_fixture/scope-handoff.changed"
  mv "$voi_fixture/scope-handoff.changed" "$voi_root/comparison-handoff.v1"
  voi_all_paths=$voi_fixture/scope-all-paths
  find "$voi_root" -type f ! -path "$voi_root/campaign.files.sha256" -print | \
    sed "s#^$voi_root/##" | LC_ALL=C sort > "$voi_all_paths"
  : > "$voi_root/campaign.files.sha256"
  while IFS= read -r voi_relative; do
    printf '%s  %s\n' "$(sha256_file "$voi_root/$voi_relative")" \
      "$voi_relative" >> "$voi_root/campaign.files.sha256"
  done < "$voi_all_paths"
  find "$voi_root" -type f -exec chmod 444 {} \;
  find "$voi_root" -type d -exec chmod 555 {} \;
  if "$voi_verifier" "$voi_root" "$voi_campaign_argument" \
    "$voi_driver_argument" "$voi_correctness_argument" \
    "$voi_identity_argument" "$voi_supervisor_argument" \
    3</dev/null 4</dev/null 5</dev/null >/dev/null 2>&1; then
    fail 'verifier accepted recomputed non-trial credential authority'
  fi
}
