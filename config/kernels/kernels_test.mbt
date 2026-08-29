///|
fn digest() -> String {
  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}

///|
test "kernel config retains only strict digest-pinned AOT policy" {
  let config = @kernels.new(
    "manifests/llama-sm90.json",
    digest(),
    DeploymentApprovedAotOnly,
  )
  assert_eq(config.expected_manifest_digest().as_hex(), digest())
  assert_eq(config.admission_policy(), DeploymentApprovedAotOnly)
  assert_true(config.permits_runtime_readiness())
  let inspection = @kernels.new(
    "manifests/llama-sm90.json",
    digest(),
    OfflineInspectionOnly,
  )
  assert_false(inspection.permits_runtime_readiness())
}

///|
test "kernel config rejects ambient paths and malformed digests" {
  for
    locator in [
      "", "/absolute/manifest.json", "../manifest.json", "a/./manifest.json", "a//manifest.json",
      "a/../manifest.json",
    ] {
    try @kernels.new(locator, digest(), DeploymentApprovedAotOnly) catch {
      Invalid(field=ManifestLocator, issue=InvalidFormat) => ()
      _ => fail("wrong strict locator failure")
    } noraise {
      _ => fail("ambient kernel manifest locator was accepted")
    }
  }
  for
    digest_value in [
      "", "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef",
    ] {
    try
      @kernels.new("manifest.json", digest_value, DeploymentApprovedAotOnly)
    catch {
      Invalid(field=ExpectedManifestDigest, issue=InvalidFormat) => ()
      _ => fail("wrong kernel manifest digest failure")
    } noraise {
      _ => fail("malformed kernel manifest digest was accepted")
    }
  }
}
