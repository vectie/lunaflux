///|
fn fused_compiled_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn fused_compiled_expected_receipt_digest(
  receipt : Bytes,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest raise {
  @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(
    fused_compiled_sha256(receipt),
  )
}

///|
fn fused_compiled_receipt(
  candidate : @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate,
  cubin : Bytes,
) -> Bytes {
  let module_sha256 = fused_compiled_sha256(cubin)
  let target = candidate.target()
  @utf8.encode(
    "schema=lunaflux-luna-cuda-aot-evidence-v1\n" +
    "source_sha256=" +
    candidate.source_digest().as_hex() +
    "\n" +
    "recipe_sha256=" +
    candidate.recipe_digest().as_hex() +
    "\n" +
    "toolchain_sha256=" +
    candidate.compiler_policy().identity().toolchain_digest_sha256() +
    "\n" +
    "driver_identity_sha256=" +
    "d".repeat(64) +
    "\n" +
    "target=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\n" +
    "function_symbol=" +
    candidate.function_symbol() +
    "\n" +
    "artifact_sha256=" +
    module_sha256 +
    "\n" +
    "first_build_sha256=" +
    module_sha256 +
    "\n" +
    "second_build_sha256=" +
    module_sha256 +
    "\n" +
    "deterministic=1\n",
  )
}

///|
fn fused_compiled_residual_receipt(
  candidate : @luna_cuda_fused_parallel_aot.FusedResidualRmsNormCudaCandidate,
  cubin : Bytes,
) -> Bytes {
  let module_sha256 = fused_compiled_sha256(cubin)
  let target = candidate.target()
  @utf8.encode(
    "schema=lunaflux-luna-cuda-aot-evidence-v1\n" +
    "source_sha256=" +
    candidate.source_digest().as_hex() +
    "\n" +
    "recipe_sha256=" +
    candidate.recipe_digest().as_hex() +
    "\n" +
    "toolchain_sha256=" +
    candidate.compiler_policy().identity().toolchain_digest_sha256() +
    "\n" +
    "driver_identity_sha256=" +
    "d".repeat(64) +
    "\n" +
    "target=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\n" +
    "function_symbol=" +
    candidate.function_symbol() +
    "\n" +
    "artifact_sha256=" +
    module_sha256 +
    "\n" +
    "first_build_sha256=" +
    module_sha256 +
    "\n" +
    "second_build_sha256=" +
    module_sha256 +
    "\n" +
    "deterministic=1\n",
  )
}

///|
fn fused_compiled_production_residual_receipt(
  candidate : @luna_cuda_fused_parallel_aot.FusedResidualRmsNormProductionCudaCandidate,
  cubin : Bytes,
) -> Bytes {
  let module_sha256 = fused_compiled_sha256(cubin)
  let target = candidate.target()
  @utf8.encode(
    "schema=lunaflux-luna-cuda-aot-evidence-v1\n" +
    "source_sha256=" +
    candidate.source_digest().as_hex() +
    "\nrecipe_sha256=" +
    candidate.recipe_digest().as_hex() +
    "\ntoolchain_sha256=" +
    candidate.compiler_policy().identity().toolchain_digest_sha256() +
    "\ndriver_identity_sha256=" +
    "d".repeat(64) +
    "\ntarget=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\nfunction_symbol=" +
    candidate.function_symbol() +
    "\nartifact_sha256=" +
    module_sha256 +
    "\nfirst_build_sha256=" +
    module_sha256 +
    "\nsecond_build_sha256=" +
    module_sha256 +
    "\ndeterministic=1\n",
  )
}

///|
fn fused_compiled_production_qkv_receipt(
  candidate : @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteProductionCudaCandidate,
  cubin : Bytes,
) -> Bytes {
  let module_sha256 = fused_compiled_sha256(cubin)
  let target = candidate.target()
  @utf8.encode(
    "schema=lunaflux-luna-cuda-aot-evidence-v1\n" +
    "source_sha256=" +
    candidate.source_digest().as_hex() +
    "\nrecipe_sha256=" +
    candidate.recipe_digest().as_hex() +
    "\ntoolchain_sha256=" +
    candidate.compiler_policy().identity().toolchain_digest_sha256() +
    "\ndriver_identity_sha256=" +
    "d".repeat(64) +
    "\ntarget=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\nfunction_symbol=" +
    candidate.function_symbol() +
    "\nartifact_sha256=" +
    module_sha256 +
    "\nfirst_build_sha256=" +
    module_sha256 +
    "\nsecond_build_sha256=" +
    module_sha256 +
    "\ndeterministic=1\n",
  )
}

///|
test "compiled fused candidates bind exact receipt CUBIN symbol and ABI without authority" {
  let fixture = fused_compiled_black_box_fixture()
  let cubin = b"ELF-CUBIN-fused-candidate-black-box-v1"
  let qkv = fused_compiled_qkv_candidate(fixture)
  let qkv_receipt = fused_compiled_receipt(qkv, cubin)
  let first = @luna_cuda_fused_parallel_aot.bind_fused_qkv_rope_kv_compiled_candidate(
    qkv,
    cubin,
    cubin,
    qkv_receipt,
    fused_compiled_expected_receipt_digest(qkv_receipt),
    fused_compiled_black_box_limits(),
  )
  let second = @luna_cuda_fused_parallel_aot.bind_fused_qkv_rope_kv_compiled_candidate(
    qkv,
    cubin,
    cubin,
    qkv_receipt,
    fused_compiled_expected_receipt_digest(qkv_receipt),
    fused_compiled_black_box_limits(),
  )
  assert_eq(first.binding_digest(), second.binding_digest())
  assert_eq(first.candidate_digest(), qkv.candidate_digest())
  assert_eq(first.source_digest(), qkv.source_digest())
  assert_eq(first.recipe_digest(), qkv.recipe_digest())
  assert_eq(first.module_bytes(), cubin)
  assert_eq(first.compile_receipt_bytes(), qkv_receipt)
  assert_eq(first.function_symbol(), qkv.function_symbol())
  assert_eq(first.kernel_abi(), QkvPositionedRopePagedKvWriteAbiV1)
  assert_false(first.manifest_bindable())
  let record = @utf8.decode(first.binding_record_bytes())
  assert_true(record.contains("candidate_only=true\n"))
  assert_true(record.contains("compiler_authority=absent\n"))
  assert_true(record.contains("deployment_approval=absent\n"))
  assert_true(record.contains("promotion_authority=absent\n"))
  assert_true(record.contains("manifest_bindable=false\n"))
  assert_true(record.contains("block=128,1,1\n"))
  assert_true(
    record.contains(
      "abi=step_counts,query_positions,query_row_offsets,sequence_lengths,",
    ),
  )

  let residual = fused_compiled_residual_candidate(fixture)
  let residual_receipt = fused_compiled_residual_receipt(residual, cubin)
  let residual_binding = @luna_cuda_fused_parallel_aot.bind_fused_residual_rmsnorm_compiled_candidate(
    residual,
    cubin,
    cubin,
    residual_receipt,
    fused_compiled_expected_receipt_digest(residual_receipt),
    fused_compiled_black_box_limits(),
  )
  assert_eq(residual_binding.candidate_digest(), residual.candidate_digest())
  assert_eq(residual_binding.kernel_abi(), ResidualRmsNormAbiV1)
  assert_eq(residual_binding.launch_dimensions().shared_memory_bytes(), 512)
  assert_true(
    @utf8.decode(residual_binding.binding_record_bytes()).contains(
      "abi=step_counts,residual,branch,norm_weight,residual_output,",
    ),
  )

  let production = @luna_cuda_fused_parallel_aot.lower_fused_residual_rmsnorm_production_candidate(
    residual,
    @luna_cuda_fused_parallel_aot.FusedParallelAotLimits::new(
      maximum_source_bytes=1048576,
      maximum_recipe_bytes=65536,
      maximum_promotion_bytes=65536,
    ),
  )
  let production_receipt = fused_compiled_production_residual_receipt(
    production, cubin,
  )
  let production_binding = @luna_cuda_fused_parallel_aot.bind_fused_residual_rmsnorm_production_compiled_candidate(
    production,
    cubin,
    cubin,
    production_receipt,
    fused_compiled_expected_receipt_digest(production_receipt),
    fused_compiled_black_box_limits(),
  )
  assert_eq(production_binding.kernel_abi(), ResidualRmsNormProductionAbiV2)
  assert_eq(production_binding.dispatch_canary_per_token(), 0U)
  assert_eq(
    production_binding.qualification_candidate_digest(),
    Some(residual.candidate_digest()),
  )
  let production_record = @utf8.decode(
    production_binding.binding_record_bytes(),
  )
  assert_true(production_record.contains("dispatch_canary_per_token=0\n"))
  assert_true(
    production_record.contains(
      "abi=step_counts,residual,branch,norm_weight,residual_output,norm_output\n",
    ),
  )

  let qkv_production = @luna_cuda_fused_parallel_aot.lower_fused_qkv_rope_paged_kv_write_production_candidate(
    qkv,
    @luna_cuda_fused_parallel_aot.FusedParallelAotLimits::new(
      maximum_source_bytes=1048576,
      maximum_recipe_bytes=65536,
      maximum_promotion_bytes=65536,
    ),
  )
  let qkv_production_receipt = fused_compiled_production_qkv_receipt(
    qkv_production, cubin,
  )
  let qkv_production_binding = @luna_cuda_fused_parallel_aot.bind_fused_qkv_rope_paged_kv_write_production_compiled_candidate(
    qkv_production,
    cubin,
    cubin,
    qkv_production_receipt,
    fused_compiled_expected_receipt_digest(qkv_production_receipt),
    fused_compiled_black_box_limits(),
  )
  assert_eq(
    qkv_production_binding.kernel_abi(),
    QkvPositionedRopePagedKvWriteProductionAbiV2,
  )
  assert_eq(qkv_production_binding.dispatch_canary_per_token(), 0U)
  assert_eq(
    qkv_production_binding.qualification_candidate_digest(),
    Some(qkv.candidate_digest()),
  )
}

///|
fn assert_fused_compiled_receipt_rejected(
  candidate : @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate,
  cubin : Bytes,
  receipt : Bytes,
  expected_receipt_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  label : String,
) -> Unit raise {
  try
    @luna_cuda_fused_parallel_aot.bind_fused_qkv_rope_kv_compiled_candidate(
      candidate,
      cubin,
      cubin,
      receipt,
      expected_receipt_digest,
      fused_compiled_black_box_limits(),
    )
  catch {
    @luna_cuda_fused_parallel_aot.FusedParallelMismatch(CompileReceipt)
    | @luna_cuda_fused_parallel_aot.FusedParallelInvalid(CompileReceipt) => ()
    _ => fail("unexpected compile-receipt rejection: " + label)
  } noraise {
    _ => fail("substituted compile receipt was admitted: " + label)
  }
}

///|
test "compiled fused binding rejects every receipt identity substitution" {
  let fixture = fused_compiled_black_box_fixture()
  let candidate = fused_compiled_qkv_candidate(fixture)
  let cubin = b"ELF-CUBIN-fused-hostile-v1"
  let receipt = @utf8.decode(fused_compiled_receipt(candidate, cubin))
  let expected_receipt_digest = fused_compiled_expected_receipt_digest(
    @utf8.encode(receipt),
  )
  for
    index, substitution in [
      (
        "schema=lunaflux-luna-cuda-aot-evidence-v1", "schema=lunaflux-luna-cuda-aot-evidence-v2",
      ),
      (candidate.source_digest().as_hex(), "1".repeat(64)),
      (candidate.recipe_digest().as_hex(), "2".repeat(64)),
      (
        candidate.compiler_policy().identity().toolchain_digest_sha256(),
        "3".repeat(64),
      ),
      ("d".repeat(64), "4".repeat(64)),
      ("target=sm_90", "target=sm_89"),
      (
        "function_symbol=" + candidate.function_symbol(),
        "function_symbol=lunaflux_substituted_fused_symbol",
      ),
      (
        "artifact_sha256=" + fused_compiled_sha256(cubin),
        "artifact_sha256=" + "5".repeat(64),
      ),
      (
        "first_build_sha256=" + fused_compiled_sha256(cubin),
        "first_build_sha256=" + "6".repeat(64),
      ),
      (
        "second_build_sha256=" + fused_compiled_sha256(cubin),
        "second_build_sha256=" + "7".repeat(64),
      ),
      ("deterministic=1", "deterministic=0"),
    ] {
    let (original, replacement) = substitution
    assert_fused_compiled_receipt_rejected(
      candidate,
      cubin,
      @utf8.encode(receipt.replace(old=original, new=replacement)),
      expected_receipt_digest,
      index.to_string(),
    )
  }
  assert_fused_compiled_receipt_rejected(
    candidate,
    cubin,
    @utf8.encode(receipt + "unknown=forbidden\n"),
    expected_receipt_digest,
    "unknown-field",
  )
}

///|
test "compiled fused binding rejects coherently re-pinned semantic substitutions" {
  let fixture = fused_compiled_black_box_fixture()
  let candidate = fused_compiled_qkv_candidate(fixture)
  let cubin = b"ELF-CUBIN-fused-semantic-hostile-v1"
  let receipt = @utf8.decode(fused_compiled_receipt(candidate, cubin))
  for
    index, substitution in [
      (
        "schema=lunaflux-luna-cuda-aot-evidence-v1", "schema=lunaflux-luna-cuda-aot-evidence-v2",
      ),
      (candidate.source_digest().as_hex(), "1".repeat(64)),
      (candidate.recipe_digest().as_hex(), "2".repeat(64)),
      (
        candidate.compiler_policy().identity().toolchain_digest_sha256(),
        "3".repeat(64),
      ),
      ("target=sm_90", "target=sm_89"),
      (
        "function_symbol=" + candidate.function_symbol(),
        "function_symbol=lunaflux_substituted_fused_symbol",
      ),
      (
        "artifact_sha256=" + fused_compiled_sha256(cubin),
        "artifact_sha256=" + "5".repeat(64),
      ),
      (
        "first_build_sha256=" + fused_compiled_sha256(cubin),
        "first_build_sha256=" + "6".repeat(64),
      ),
      (
        "second_build_sha256=" + fused_compiled_sha256(cubin),
        "second_build_sha256=" + "7".repeat(64),
      ),
      ("deterministic=1", "deterministic=0"),
    ] {
    let (original, replacement) = substitution
    let substituted = @utf8.encode(
      receipt.replace(old=original, new=replacement),
    )
    assert_fused_compiled_receipt_rejected(
      candidate,
      cubin,
      substituted,
      fused_compiled_expected_receipt_digest(substituted),
      "semantic-" + index.to_string(),
    )
  }
}

///|
test "compiled fused binding rejects candidate artifact and bound substitutions" {
  let fixture = fused_compiled_black_box_fixture()
  let candidate = fused_compiled_qkv_candidate(fixture)
  let cubin = b"ELF-CUBIN-fused-deterministic-v1"
  let receipt = fused_compiled_receipt(candidate, cubin)
  try
    @luna_cuda_fused_parallel_aot.bind_fused_qkv_rope_kv_compiled_candidate(
      candidate,
      cubin,
      b"ELF-CUBIN-fused-different-v1",
      receipt,
      fused_compiled_expected_receipt_digest(receipt),
      fused_compiled_black_box_limits(),
    )
  catch {
    @luna_cuda_fused_parallel_aot.FusedParallelMismatch(Determinism) => ()
    _ => fail("unexpected nondeterministic-CUBIN rejection")
  } noraise {
    _ => fail("different CUBIN snapshots were admitted")
  }
  let residual = fused_compiled_residual_candidate(fixture)
  try
    @luna_cuda_fused_parallel_aot.bind_fused_residual_rmsnorm_compiled_candidate(
      residual,
      cubin,
      cubin,
      receipt,
      fused_compiled_expected_receipt_digest(receipt),
      fused_compiled_black_box_limits(),
    )
  catch {
    @luna_cuda_fused_parallel_aot.FusedParallelMismatch(CompileReceipt) => ()
    _ => fail("unexpected cross-candidate receipt rejection")
  } noraise {
    _ => fail("QKV receipt was replayed against residual/RMSNorm")
  }
  try
    @luna_cuda_fused_parallel_aot.bind_fused_qkv_rope_kv_compiled_candidate(
      candidate,
      cubin,
      cubin,
      receipt,
      fused_compiled_expected_receipt_digest(receipt),
      fused_compiled_black_box_limits(maximum_module_bytes=4L),
    )
  catch {
    @luna_cuda_fused_parallel_aot.FusedParallelLimitExceeded(CompiledArtifact) =>
      ()
    _ => fail("unexpected compiled-module limit rejection")
  } noraise {
    _ => fail("oversized fused CUBIN was admitted")
  }
}
