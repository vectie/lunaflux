///|
fn source_tp_fixture() -> @worker_wire.EncodedBootstrapSource raise {
  let ordinary = source_fixture()
  let ranks = [
    @worker_wire.TensorParallelRankRuntimeClaim::new(
      rank=0,
      device_ordinal=4,
      memory_ceiling_bytes=1073741824L,
      weight_reserved_bytes=268435456L,
      activation_workspace_reserved_bytes=134217728L,
      kv_reserved_bytes=268435456L,
      collective_reserved_bytes=0L,
    ),
    @worker_wire.TensorParallelRankRuntimeClaim::new(
      rank=1,
      device_ordinal=5,
      memory_ceiling_bytes=1073741824L,
      weight_reserved_bytes=268435456L,
      activation_workspace_reserved_bytes=134217728L,
      kv_reserved_bytes=268435456L,
      collective_reserved_bytes=67108864L,
    ),
  ]
  @worker_wire.admit_tensor_parallel_bootstrap_source(
    ordinary.roots(),
    ordinary.model(),
    ordinary.kernel(),
    ordinary.execution(),
    ordinary.ceilings(),
    @worker_wire.TensorParallelBootstrapPolicy::new(
      ranks,
      minimum_collective_runtime_version_code=21403,
      maximum_collective_runtime_version_code=21999,
      collective_exchange_timeout_millis=600000,
    ),
  )
}

///|
fn source_tp_approved_fixture() -> @worker_wire.EncodedBootstrapSource raise {
  let legacy = source_tp_fixture()
  let approval = @worker_wire.WorkerBootstrapLunaApproval::from_untrusted_wire_claim(
    manifest_digest_sha256=source_digest_b(),
    key_identity_sha256=source_digest_a(),
    signature_digest_sha256=source_digest_b(),
    approved_source_sha256=legacy.kernel().manifest_digest(),
  )
  @worker_wire.admit_tensor_parallel_approved_bootstrap_source(
    legacy.roots(),
    legacy.model(),
    @worker_wire.WorkerBootstrapKernelSource::new(
      manifest_locator=legacy.kernel().manifest_locator(),
      manifest_digest_sha256=legacy.kernel().manifest_digest(),
      luna_approval=Some(approval),
    ),
    legacy.execution(),
    legacy.ceilings(),
    legacy.tensor_parallel_policy().unwrap(),
  )
}

///|
test "tensor parallel source v4 roundtrips exact rank policy" {
  let source = source_tp_fixture()
  assert_eq(source.recipe(), DenseLlamaTensorParallelPagedAotV5)
  let bytes = FixedArray::make(source.capacity(), b'\x00')
  source.copy_to(bytes, destination_offset=0)
  let decoded = @worker_wire.decode_bootstrap_source(bytes, source.length())
  assert_eq(decoded.digest(), source.digest())
  guard decoded.tensor_parallel_policy() is Some(policy) else {
    fail("v4 policy disappeared")
  }
  assert_eq(policy.rank_count(), 2)
  assert_eq(policy.minimum_collective_runtime_version_code(), 21403)
  assert_eq(policy.maximum_collective_runtime_version_code(), 21999)
  assert_eq(policy.collective_exchange_timeout_millis(), 600000)
  assert_eq(policy.rank(1).unwrap().device_ordinal(), 5)
  assert_eq(policy.rank(1).unwrap().kv_reserved_bytes(), 268435456L)
  assert_eq(policy.rank(0).unwrap().collective_reserved_bytes(), 0L)
}

///|
test "tensor parallel source claim rejects Int64 reservation overflow" {
  try
    @worker_wire.TensorParallelRankRuntimeClaim::new(
      rank=0,
      device_ordinal=0,
      memory_ceiling_bytes=9223372036854775807L,
      weight_reserved_bytes=9223372036854775807L,
      activation_workspace_reserved_bytes=1L,
      kv_reserved_bytes=0L,
      collective_reserved_bytes=0L,
    )
  catch {
    InvalidBootstrapSource(Ceiling) => ()
    _ => fail("unexpected claim overflow rejection")
  } noraise {
    _ => fail("overflowed claim was admitted")
  }
}

///|
test "approved tensor parallel v8 roundtrips exact approval and rank policy" {
  let source = source_tp_approved_fixture()
  assert_eq(source.recipe(), DenseLlamaTensorParallelPagedAotApprovedV8)
  let bytes = FixedArray::make(source.capacity(), b'\x00')
  source.copy_to(bytes, destination_offset=0)
  let decoded = @worker_wire.decode_bootstrap_source(bytes, source.length())
  assert_eq(decoded.digest(), source.digest())
  assert_eq(decoded.tensor_parallel_policy().unwrap().rank_count(), 2)
  let approval = decoded.kernel().luna_approval().unwrap()
  assert_eq(approval.approved_source(), decoded.kernel().manifest_digest())
  assert_eq(approval.manifest_digest(), source_digest_b())
}

///|
test "approved tensor parallel v8 rejects approval source substitution" {
  let legacy = source_tp_fixture()
  for
    pair in [
      (source_digest_b(), source_digest_a()),
      (source_digest_a(), source_digest_b()),
    ] {
    let (manifest, source) = pair
    let approval = @worker_wire.WorkerBootstrapLunaApproval::from_untrusted_wire_claim(
      manifest_digest_sha256=manifest,
      key_identity_sha256=source_digest_a(),
      signature_digest_sha256=source_digest_b(),
      approved_source_sha256=source,
    )
    try
      @worker_wire.admit_tensor_parallel_approved_bootstrap_source(
        legacy.roots(),
        legacy.model(),
        @worker_wire.WorkerBootstrapKernelSource::new(
          manifest_locator=legacy.kernel().manifest_locator(),
          manifest_digest_sha256=legacy.kernel().manifest_digest(),
          luna_approval=Some(approval),
        ),
        legacy.execution(),
        legacy.ceilings(),
        legacy.tensor_parallel_policy().unwrap(),
      )
    catch {
      InvalidBootstrapSource(Digest) => ()
      _ => fail("wrong v8 source-substitution rejection")
    } noraise {
      _ => fail("v8 accepted approval for a different immutable source")
    }
  }
}
