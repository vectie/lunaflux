///|
priv struct MaterializedBundleAdmission {
  root_label : String
  evidence : @deployment_bundle.DeploymentBundleEvidence
}

///|
fn parse_materialized_bundle_argument(
  argument : StringView,
) -> (String, @deployment_bundle.DeploymentBundleDigest) raise RuntimeInstanceError {
  let marker = "#sha256="
  if argument.length() < marker.length() + 66 {
    raise fail(DeploymentBundleAdmission)
  }
  let mut split = -1
  for index in 0..<(argument.length() - marker.length() + 1) {
    if argument[index:index + marker.length()].to_owned() == marker {
      if split >= 0 {
        raise fail(DeploymentBundleAdmission)
      }
      split = index
    }
  }
  if split <= 0 || split + marker.length() + 64 != argument.length() {
    raise fail(DeploymentBundleAdmission)
  }
  let root_label = argument[0:split].to_owned()
  let digest = @deployment_bundle.DeploymentBundleDigest::from_sha256(
    argument[split + marker.length():],
  ) catch {
    _ => raise fail(DeploymentBundleAdmission)
  }
  (root_label, digest)
}

///|
fn load_materialized_bundle(
  argument : StringView,
  limits : @deployment_bundle.DeploymentBundleLimits,
) -> MaterializedBundleAdmission raise RuntimeInstanceError {
  let (root_label, expected_digest) = parse_materialized_bundle_argument(
    argument,
  )
  let root = @approved_fs.ApprovedRoot::open_absolute(root_label) catch {
    _ => raise fail(DeploymentBundleAdmission)
  }
  let locator = @approved_fs.ApprovedRelativeLocator::new(
    "evidence/deployment-bundle.v1",
  ) catch {
    _ => {
      close_root(root, Cleanup)
      raise fail(DeploymentBundleAdmission)
    }
  }
  let file = root.open_file(locator) catch {
    _ => {
      close_root(root, Cleanup)
      raise fail(DeploymentBundleAdmission)
    }
  }
  let read : Result[Bytes, RuntimeInstanceError] = Ok(
    file.read_immutable_snapshot(
      maximum_bytes=Int64::from_int(limits.maximum_bytes()),
    ),
  ) catch {
    _ => Err(fail(DeploymentBundleAdmission))
  }
  let file_close = Ref(true)
  file.close() catch {
    _ => file_close.val = false
  }
  let root_close = Ref(true)
  root.close() catch {
    _ => root_close.val = false
  }
  let bytes = match read {
    Err(primary) => raise primary
    Ok(_) if !file_close.val || !root_close.val =>
      raise fail(DeploymentBundleAdmission)
    Ok(value) => value
  }
  let evidence = @deployment_bundle.admit_deployment_bundle_evidence(
    bytes,
    expected_digest~,
    limits~,
  ) catch {
    _ => raise fail(DeploymentBundleAdmission)
  }
  { root_label, evidence }
}
