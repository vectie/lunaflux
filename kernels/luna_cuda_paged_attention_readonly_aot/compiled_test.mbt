///|
fn readonly_fixture_receipt(
  candidate : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCudaCandidate,
  cubin : Bytes,
) -> Bytes {
  let fallback = candidate.reference_fallback()
  let target = candidate.target()
  let module_sha = @crypto.bytes_to_hex_string(@crypto.sha256(cubin))
  @utf8.encode(
    "schema=lunaflux-paged-attention-readonly-aot-evidence.v1\n" +
    "candidate_sha256=" +
    candidate.candidate_digest().as_hex() +
    "\nsource_sha256=" +
    candidate.source_digest().as_hex() +
    "\nrecipe_sha256=" +
    candidate.recipe_digest().as_hex() +
    "\nfallback_source_sha256=" +
    fallback.source_digest().as_hex() +
    "\nfallback_recipe_sha256=" +
    fallback.recipe_digest().as_hex() +
    "\ntoolchain_sha256=" +
    fallback.compiler_policy().identity().toolchain_digest_sha256() +
    "\ndriver_identity_sha256=" +
    "d".repeat(64) +
    "\ntarget=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\nfunction_symbol=" +
    candidate.function_symbol() +
    "\nraw_pointer_abi=" +
    candidate.kernel_abi().canonical_name() +
    "\ninput_row_width=" +
    candidate.input_row_width().to_string() +
    "\ninput_activation_ref=" +
    candidate.input_activation().as_int().to_string() +
    "\ndispatch_canary_per_token=" +
    candidate.dispatch_canary_per_token().to_string() +
    "\ndispatch_canary_cell_count=" +
    candidate.profile().max_query_tokens().to_string() +
    "\nartifact_sha256=" +
    module_sha +
    "\nfirst_build_sha256=" +
    module_sha +
    "\nsecond_build_sha256=" +
    module_sha +
    "\ndeterministic=1\nkv_cache_mutation=none\n",
  )
}

///|
fn readonly_fixture_compiled_limits() -> @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCompiledLimits raise {
  @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCompiledLimits::new(
    maximum_module_bytes=4096L,
    maximum_receipt_bytes=8192,
    maximum_binding_bytes=8192,
  )
}

///|
test "compiled read-only candidate binds deterministic CUBIN and remains inert" {
  let candidate = readonly_fixture_candidate()
  let cubin = b"ELF-CUBIN-paged-attention-readonly-v1"
  let receipt = readonly_fixture_receipt(candidate, cubin)
  let receipt_digest = @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(receipt)),
  )
  let first = @luna_cuda_paged_attention_readonly_aot.bind_paged_attention_readonly_compiled_candidate(
    candidate,
    cubin,
    cubin,
    receipt,
    receipt_digest,
    readonly_fixture_compiled_limits(),
  )
  let second = @luna_cuda_paged_attention_readonly_aot.bind_paged_attention_readonly_compiled_candidate(
    candidate,
    cubin,
    cubin,
    receipt,
    receipt_digest,
    readonly_fixture_compiled_limits(),
  )
  assert_eq(first.binding_digest(), second.binding_digest())
  assert_eq(first.candidate_digest(), candidate.candidate_digest())
  assert_eq(first.source_digest(), candidate.source_digest())
  assert_eq(first.recipe_digest(), candidate.recipe_digest())
  assert_eq(first.module_bytes(), cubin)
  assert_false(first.manifest_bindable())
  assert_false(first.promotion_authorized())
  let binding = @utf8.decode(first.binding_record_bytes())
  assert_true(binding.contains("kv_cache_mutation=none\n"))
  assert_true(binding.contains("candidate_only=true\n"))
  assert_true(binding.contains("deployment_approval=absent\n"))
  assert_true(binding.contains("promotion_authority=absent\n"))
  assert_true(binding.contains("manifest_bindable=false\n"))
}

///|
test "compiled read-only binding rejects receipt and build substitution" {
  let candidate = readonly_fixture_candidate()
  let cubin = b"ELF-CUBIN-paged-attention-readonly-hostile-v1"
  let original = @utf8.decode(readonly_fixture_receipt(candidate, cubin))
  for
    substitution in [
      (
        "candidate_sha256=" + candidate.candidate_digest().as_hex(),
        "candidate_sha256=" + "a".repeat(64),
      ),
      (
        "source_sha256=" + candidate.source_digest().as_hex(),
        "source_sha256=" + "b".repeat(64),
      ),
      (
        "recipe_sha256=" + candidate.recipe_digest().as_hex(),
        "recipe_sha256=" + "c".repeat(64),
      ),
      ("target=sm_120", "target=sm_89"),
      (
        "function_symbol=" + candidate.function_symbol(),
        "function_symbol=foreign",
      ),
      (
        "input_row_width=" + candidate.input_row_width().to_string(),
        "input_row_width=4",
      ),
      ("input_activation_ref=1", "input_activation_ref=9"),
      (
        "dispatch_canary_per_token=" +
        candidate.dispatch_canary_per_token().to_string(),
        "dispatch_canary_per_token=0",
      ),
      ("dispatch_canary_cell_count=3", "dispatch_canary_cell_count=2"),
      ("kv_cache_mutation=none", "kv_cache_mutation=allowed"),
    ] {
    let hostile = @utf8.encode(
      original.replace(old=substitution.0, new=substitution.1),
    )
    let hostile_digest = @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyDigest::from_sha256(
      @crypto.bytes_to_hex_string(@crypto.sha256(hostile)),
    )
    try
      @luna_cuda_paged_attention_readonly_aot.bind_paged_attention_readonly_compiled_candidate(
        candidate,
        cubin,
        cubin,
        hostile,
        hostile_digest,
        readonly_fixture_compiled_limits(),
      )
    catch {
      @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyMismatch(
        CompileReceipt
      )
      | @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyInvalid(
        CompileReceipt
      ) => ()
      _ => fail("unexpected compile receipt rejection")
    } noraise {
      _ => fail("substituted compile receipt was accepted")
    }
  }
  let receipt = readonly_fixture_receipt(candidate, cubin)
  let receipt_digest = @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(receipt)),
  )
  try
    @luna_cuda_paged_attention_readonly_aot.bind_paged_attention_readonly_compiled_candidate(
      candidate,
      cubin,
      b"different-cubin",
      receipt,
      receipt_digest,
      readonly_fixture_compiled_limits(),
    )
  catch {
    @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyMismatch(
      Determinism
    ) => ()
    _ => fail("unexpected deterministic build rejection")
  } noraise {
    _ => fail("non-deterministic CUBIN pair was accepted")
  }
}
