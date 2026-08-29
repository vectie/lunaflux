///|
fn fixture_evidence() -> Bytes {
  let text =
    #|schema=lunaflux.deployment-bundle.v1
    #|base_image=registry.invalid/lunaflux/cuda@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    #|linux_architecture=x86_64
    #|launch_file_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    #|runtime_descriptor_relative=runtime/descriptor.json
    #|runtime_descriptor_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    #|instance_policy_relative=instance/policy.json
    #|instance_policy_sha256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
    #|lunaflux_executable_sha256=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
    #|worker_executable_sha256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
    #|model_inventory_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    #|policy_inventory_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    #|kernel_manifest_relative=execution-i8-v4.json
    #|kernel_manifest_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    #|kernel_inventory_sha256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
    #|runtime_library_inventory_sha256=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
    #|rootfs_inventory_sha256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
    #|assembly_input_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    #|assembler_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    #|exact_inventory=1
    #|compiler_jit_free=1
    #|
  @utf8.encode(text)
}

///|
fn fixture_limits() -> DeploymentBundleLimits raise DeploymentBundleError {
  DeploymentBundleLimits::new(
    maximum_bytes=8192,
    maximum_base_image_bytes=512,
    maximum_locator_bytes=256,
  )
}

///|
test "canonical evidence is digest-pinned and release-artifact compatible" {
  let bytes = fixture_evidence()
  let digest = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
  let admitted = admit_deployment_bundle_evidence(
    bytes,
    expected_digest=DeploymentBundleDigest::from_sha256(digest),
    limits=fixture_limits(),
  )
  assert_eq(admitted.architecture(), X86_64)
  assert_eq(admitted.runtime_descriptor_relative(), "runtime/descriptor.json")
  assert_eq(admitted.kernel_manifest_relative(), "execution-i8-v4.json")
  assert_eq(admitted.canonical_bytes(), bytes)
  let release_limits = @release_evidence.ReleaseCampaignLimits::new(
    max_canonical_bytes=65536,
    max_locator_bytes=256,
    max_artifact_bytes=65536L,
  )
  let artifact = admitted.as_release_artifact(
    "evidence/deployment-bundle.v1", release_limits,
  )
  assert_eq(artifact.digest().as_hex(), digest)
  assert_eq(artifact.byte_length(), Int64::from_int(bytes.length()))
}

///|
test "substitution and noncanonical evidence fail closed" {
  let bytes = fixture_evidence()
  let digest = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
  let wrong = DeploymentBundleDigest::from_sha256(
    "123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0",
  )
  try
    admit_deployment_bundle_evidence(
      bytes,
      expected_digest=wrong,
      limits=fixture_limits(),
    )
  catch {
    DigestMismatch => ()
    _ => fail("unexpected substitution error")
  } noraise {
    _ => fail("expected substituted evidence to fail")
  }
  let extra = Bytes::from_array(
    bytes.to_array() + @utf8.encode("extra=1\n").to_array(),
  )
  let extra_digest = @crypto.bytes_to_hex_string(@crypto.sha256(extra))
  try
    admit_deployment_bundle_evidence(
      extra,
      expected_digest=DeploymentBundleDigest::from_sha256(extra_digest),
      limits=fixture_limits(),
    )
  catch {
    NonCanonicalEvidence => ()
    _ => fail("unexpected noncanonical error")
  } noraise {
    _ => fail("expected extra field to fail")
  }
  assert_true(digest.length() == 64)
}

///|
test "short and mutable-tag base references fail without partial admission" {
  let original = fixture_evidence()
  for
    hostile in [
      "x", "registry.invalid/cuda:latest@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ] {
    let original_text = @utf8.decode(original)
    let lines = original_text.split("\n").to_array()
    lines[1] = "base_image=\{hostile}"
    let changed = @utf8.encode(lines.join("\n"))
    let changed_digest = @crypto.bytes_to_hex_string(@crypto.sha256(changed))
    try
      admit_deployment_bundle_evidence(
        changed,
        expected_digest=DeploymentBundleDigest::from_sha256(changed_digest),
        limits=fixture_limits(),
      )
    catch {
      InvalidField(BaseImage) => ()
      _ => fail("unexpected base-image error")
    } noraise {
      _ => fail("expected hostile base image to fail")
    }
  }
}
