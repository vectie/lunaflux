///| Snapshot and strictly admit one digest-pinned instance policy.

///|
/// The independently caller-owned policy root is identity-bound for this call,
/// remains caller-owned, and never escapes in the returned admission.
pub fn load(
  policy_root : @approved_fs.ApprovedRoot,
  policy_root_label : StringView,
  policy_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : InstancePolicyDigest,
  limits : InstancePolicyFileLimits,
) -> InstancePolicyAdmission raise InstancePolicyFileError {
  policy_root.require_absolute_identity(policy_root_label) catch {
    error => raise map_fs_error(error, BindRoot, limits.max_policy_bytes())
  }
  load_bound_policy(policy_root, policy_locator, expected_digest, limits)
}

///|
fn load_bound_policy(
  policy_root : @approved_fs.ApprovedRoot,
  policy_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : InstancePolicyDigest,
  limits : InstancePolicyFileLimits,
) -> InstancePolicyAdmission raise InstancePolicyFileError {
  let bytes = read_policy_snapshot(
    policy_root, policy_locator, expected_digest, limits,
  )
  parse_policy(bytes, expected_digest, limits)
}

///|
/// Loads a policy from a pinned materialization source. The view has already
/// authenticated the actual no-follow source identity; its target label is
/// retained by the outer launch join and is not reopened here.
pub fn load_materialized(
  policy : @approved_fs.ApprovedRootMaterializationView,
  policy_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : InstancePolicyDigest,
  limits : InstancePolicyFileLimits,
) -> InstancePolicyAdmission raise InstancePolicyFileError {
  load_bound_policy(
    policy.source_root(),
    policy_locator,
    expected_digest,
    limits,
  )
}
