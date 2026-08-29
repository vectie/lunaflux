///|
fn source_digest_c() -> String {
  "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
}

///|
fn source_digest_d() -> String {
  "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
}

///|
fn fp8_v9_source_fixture() -> @worker_wire.EncodedBootstrapSource raise {
  let approval = @worker_wire.WorkerBootstrapLunaApproval::from_untrusted_wire_claim(
    manifest_digest_sha256=source_digest_a(),
    key_identity_sha256=source_digest_b(),
    signature_digest_sha256=source_digest_c(),
    approved_source_sha256=source_digest_d(),
  )
  @worker_wire.admit_fp8_reusable_bootstrap_source_v9(
    @worker_wire.WorkerBootstrapRoots::new(
      model_root="/srv/lunaflux/models",
      kernel_root="/srv/lunaflux/kernels",
    ),
    @worker_wire.WorkerFp8BootstrapModelSource::new(
      "config.json",
      "weights/fp8.numeric",
      source_digest_a(),
      source_digest_b(),
    ),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator="manifests/fp8-v5.json",
      manifest_digest_sha256=source_digest_c(),
      luna_approval=Some(approval),
    ),
    @worker_wire.WorkerFp8BootstrapExecution::new(
      8,
      9,
      25769803776L,
      true,
      true,
      16,
      4096,
      source_digest_d(),
    ),
    @worker_wire.WorkerBootstrapCeilings::new(
      max_model_config_bytes=1048576,
      max_weight_file_bytes=8589934592L,
      max_weight_arena_bytes=8589934592L,
      max_activation_arena_bytes=1073741824L,
      max_kv_arena_bytes=4294967296L,
      max_execution_manifest_bytes=1048576L,
      max_module_bytes=67108864L,
      max_total_module_bytes=268435456L,
    ),
  )
}

///|
test "approved reusable FP8 v9 source roundtrips exact authority identities" {
  let encoded = fp8_v9_source_fixture()
  let transport = FixedArray::make(encoded.capacity(), b'\x00')
  encoded.copy_to(transport, destination_offset=0)
  let decoded = @worker_wire.decode_bootstrap_source(
    transport,
    encoded.length(),
  )
  assert_eq(decoded.recipe(), DenseLlamaFp8ReusablePagedAotApprovedV9)
  assert_eq(decoded.digest(), encoded.digest())
  guard decoded.fp8_model_source() is Some(model) &&
    decoded.fp8_execution() is Some(execution) &&
    decoded.kernel().luna_approval() is Some(approval) else {
    fail("FP8 v9 evidence was dropped")
  }
  assert_eq(model.numeric_weight_artifact_digest(), source_digest_b())
  assert_eq(execution.reusable_recipe_digest(), source_digest_d())
  assert_eq(approval.manifest_digest(), source_digest_a())
  assert_eq(approval.approved_source(), source_digest_d())
}

///|
test "reusable FP8 v9 admits sm120 and rejects unsupported or unapproved input" {
  let blackwell = @worker_wire.WorkerFp8BootstrapExecution::new(
    12,
    0,
    25769803776L,
    true,
    true,
    16,
    4096,
    source_digest_d(),
  )
  let base = fp8_v9_source_fixture()
  guard base.fp8_model_source() is Some(model) else {
    fail("FP8 fixture lost typed model evidence")
  }
  let encoded = @worker_wire.admit_fp8_reusable_bootstrap_source_v9(
    base.roots(),
    model,
    base.kernel(),
    blackwell,
    base.ceilings(),
  )
  let bytes = FixedArray::make(encoded.capacity(), b'\x00')
  encoded.copy_to(bytes, destination_offset=0)
  let decoded = @worker_wire.decode_fp8_reusable_bootstrap_source_v9(
    bytes,
    encoded.length(),
  )
  guard decoded.fp8_execution() is Some(execution) else {
    fail("sm120 FP8 execution facts were dropped")
  }
  assert_eq(execution.compute_major(), 12)
  assert_eq(execution.compute_minor(), 0)
  try
    @worker_wire.WorkerFp8BootstrapExecution::new(
      10,
      0,
      25769803776L,
      true,
      true,
      16,
      4096,
      source_digest_d(),
    )
  catch {
    InvalidBootstrapSource(Recipe) => ()
    _ => fail("unsupported FP8 target reported an unrelated error")
  } noraise {
    _ => fail("unsupported FP8 target entered reusable source")
  }
  let approved = base
  try
    @worker_wire.admit_fp8_reusable_bootstrap_source_v9(
      approved.roots(),
      approved.fp8_model_source().unwrap(),
      @worker_wire.WorkerBootstrapKernelSource::new(
        manifest_locator="manifests/fp8-v5.json",
        manifest_digest_sha256=source_digest_c(),
      ),
      approved.fp8_execution().unwrap(),
      approved.ceilings(),
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("unapproved reusable FP8 source was admitted")
  }
}

///|
test "untrusted wire claim rejects zero digest" {
  let zero = "0".repeat(64)
  try
    @worker_wire.WorkerBootstrapLunaApproval::from_untrusted_wire_claim(
      manifest_digest_sha256=zero,
      key_identity_sha256="1".repeat(64),
      signature_digest_sha256="2".repeat(64),
      approved_source_sha256="3".repeat(64),
    )
  catch {
    InvalidBootstrapSource(Digest) => ()
    _ => fail("zero approval digest reported unrelated failure")
  } noraise {
    _ => fail("zero approval digest was admitted")
  }
}
