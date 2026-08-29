///|
fn campaign_test_receipt(
  source : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  recipe : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  compiler : @luna_specialization.LunaCompilerPolicy,
  target : @catalog.DeviceTarget,
  symbol : String,
  cubin : Bytes,
) -> Bytes {
  let module_digest = campaign_sha256(cubin)
  @utf8.encode(
    "schema=lunaflux-luna-cuda-aot-evidence-v1\n" +
    "source_sha256=" +
    source.as_hex() +
    "\nrecipe_sha256=" +
    recipe.as_hex() +
    "\ntoolchain_sha256=" +
    compiler.identity().toolchain_digest_sha256() +
    "\ndriver_identity_sha256=" +
    "d".repeat(64) +
    "\ntarget=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\nfunction_symbol=" +
    symbol +
    "\nartifact_sha256=" +
    module_digest +
    "\nfirst_build_sha256=" +
    module_digest +
    "\nsecond_build_sha256=" +
    module_digest +
    "\ndeterministic=1\n",
  )
}

///|
fn campaign_test_bindings(
  set : @luna_fused_candidate_export.LunaFusedCandidateQualificationSet,
) -> FusedProbeBindings raise {
  let qkv = set.qkv_candidate()
  let residual = set.residual_norm_candidate()
  let qkv_cubin = b"ELF-CUBIN-fused-campaign-qkv-v1"
  let residual_cubin = b"ELF-CUBIN-fused-campaign-residual-v1"
  bind_fused_probe_artifacts(
    set,
    qkv_cubin,
    qkv_cubin,
    campaign_test_receipt(
      qkv.source_digest(),
      qkv.recipe_digest(),
      qkv.compiler_policy(),
      qkv.target(),
      qkv.function_symbol(),
      qkv_cubin,
    ),
    residual_cubin,
    residual_cubin,
    campaign_test_receipt(
      residual.source_digest(),
      residual.recipe_digest(),
      residual.compiler_policy(),
      residual.target(),
      residual.function_symbol(),
      residual_cubin,
    ),
  )
}

///|
fn campaign_test_observation(
  set : @luna_fused_candidate_export.LunaFusedCandidateQualificationSet,
  cycles : String,
) -> Bytes {
  let qkv = set.qkv_candidate()
  let residual = set.residual_norm_candidate()
  @utf8.encode(
    "outcome=fused-parallel-sm120-qualification-pass cycles=" +
    cycles +
    " shapes=4 launches=20 qkv_max_abs_error_ppb=0 qkv_max_relative_error_ppb=0 qkv_observed_dispatch_canary=" +
    (qkv.dispatch_canary() * 12U).to_string() +
    " residual_max_abs_error_ppb=0 residual_max_relative_error_ppb=0 residual_observed_dispatch_canary=" +
    (residual.dispatch_canary() * 12U).to_string() +
    " standalone_qkv_oracle=pass deterministic_cubins=2x5 target=sm_120 device_uuid=GPU-01234567-89ab-cdef-0123-456789abcdef device_pci=00000000:01:00.0 device_name_sha256=" +
    "6".repeat(64) +
    " device_total_memory_bytes=17179869184 cuda_driver_version=13010 resources=context0,stream0,event0,graph0,graph_exec0,allocation0,module0,function0,device_bytes0,pending0,cleanup0 authority=qualification-only\n",
  )
}

///|
fn campaign_test_audit(
  set : @luna_fused_candidate_export.LunaFusedCandidateQualificationSet,
) -> Bytes {
  let output = StringBuilder()
  for
    pair in [
      ("schema", "lunaflux-fused-approved-device-observation.v1"),
      ("device_uuid", "GPU-test-sm120"),
      ("device_pci", "00000000:01:00.0"),
      ("device_name_sha256", "6".repeat(64)),
      ("device_total_memory_bytes", "17179869184"),
      ("device_compute", "12.0"),
      ("host_driver_version", "590.48.01"),
      ("device_identity_record_sha256", "7".repeat(64)),
      ("approved_policy_sha256", "9".repeat(64)),
      ("nvcc_sha256", "8".repeat(64)),
      ("compute_sanitizer_sha256", "a".repeat(64)),
    ] {
    campaign_line(output, pair.0, pair.1)
  }
  for
    pair in [
      ("referee_source_sha256", "1".repeat(64)),
      ("probe_executable_sha256", "2".repeat(64)),
      (
        "standalone_qkv_source_sha256",
        set.qkv_fallback().source_digest().as_hex(),
      ),
      (
        "standalone_qkv_recipe_sha256",
        set.qkv_fallback().recipe_digest().as_hex(),
      ),
      ("standalone_qkv_cubin_sha256", "3".repeat(64)),
      ("standalone_qkv_receipt_sha256", "3".repeat(64)),
      (
        "standalone_rope_source_sha256",
        set.rope_fallback().source_digest().as_hex(),
      ),
      (
        "standalone_rope_recipe_sha256",
        set.rope_fallback().recipe_digest().as_hex(),
      ),
      ("standalone_rope_cubin_sha256", "3".repeat(64)),
      ("standalone_rope_receipt_sha256", "3".repeat(64)),
      (
        "standalone_kv_write_source_sha256",
        set.kv_write_fallback().source_digest().as_hex(),
      ),
      (
        "standalone_kv_write_recipe_sha256",
        set.kv_write_fallback().recipe_digest().as_hex(),
      ),
      ("standalone_kv_write_cubin_sha256", "3".repeat(64)),
      ("standalone_kv_write_receipt_sha256", "3".repeat(64)),
      ("standalone_oracle_output_sha256", "3".repeat(64)),
    ] {
    campaign_line(output, pair.0, pair.1)
  }
  campaign_line(output, "standalone_oracle_executed", "true")
  for
    key in [
      "ptxas_sha256", "cuobjdump_sha256", "nvdisasm_sha256", "ptxas_version_sha256",
      "cuobjdump_version_sha256", "nvdisasm_version_sha256",
    ] {
    campaign_line(output, key, "3".repeat(64))
  }
  for pair in [("qkv_", "0"), ("residual_", "512")] {
    campaign_line(output, pair.0 + "cuobjdump_sass_sha256", "4".repeat(64))
    campaign_line(output, pair.0 + "cuobjdump_sass_instruction_count", "17")
    campaign_line(output, pair.0 + "nvdisasm_sass_sha256", "5".repeat(64))
    campaign_line(output, pair.0 + "nvdisasm_sass_instruction_count", "17")
    campaign_line(output, pair.0 + "global_load_instruction_count", "4")
    campaign_line(output, pair.0 + "global_store_instruction_count", "3")
    campaign_line(output, pair.0 + "registers_per_thread", "64")
    campaign_line(output, pair.0 + "static_shared_bytes", pair.1)
    for
      suffix in [
        "local_bytes", "stack_bytes", "spill_store_bytes", "spill_load_bytes",
      ] {
      campaign_line(output, pair.0 + suffix, "0")
    }
  }
  @utf8.encode(output.to_string())
}

///|
test "campaign renderer composes exact record admitted by release boundary" {
  let set = fused_probe_candidate_set("b".repeat(64))
  let bindings = campaign_test_bindings(set)
  let observation = parse_fused_campaign_observation(
    campaign_test_observation(set, "1"),
    set,
  )
  let artifact_seal = campaign_digest("e".repeat(64))
  let evidence_seal = campaign_digest("f".repeat(64))
  let canonical = render_fused_campaign_canonical(
    set,
    bindings,
    observation,
    campaign_test_audit(set),
    artifact_seal,
    evidence_seal,
  )
  assert_eq(@utf8.decode(canonical).split("\n").count(), 172)
  let expected = campaign_digest(campaign_sha256(canonical))
  let admitted = @luna_fused_physical_evidence.admit_fused_physical_evidence(
    set.qkv_candidate(),
    bindings.qkv,
    set.residual_norm_candidate(),
    bindings.residual,
    canonical,
    expected_digest=expected,
    expected_approved_policy_digest=campaign_digest("9".repeat(64)),
  )
  assert_eq(admitted.digest(), expected)
  assert_eq(admitted.artifact_directory_seal_digest(), artifact_seal)
  assert_eq(admitted.evidence_directory_seal_digest(), evidence_seal)
  assert_false(admitted.manifest_bindable())
  assert_false(admitted.promotion_authorized())
}

///|
test "campaign parser rejects ambiguous numbers and dirty sanitizer summaries" {
  let set = fused_probe_candidate_set("b".repeat(64))
  for cycles in ["01", "+1", "-0"] {
    try
      parse_fused_campaign_observation(
        campaign_test_observation(set, cycles),
        set,
      )
    catch {
      FusedCudaProbeRejected(_) => ()
      _ => fail("unexpected ambiguous observation rejection")
    } noraise {
      _ => fail("ambiguous observation number was normalized")
    }
  }
  require_clean_sanitizer_log(
    b"========= ERROR SUMMARY: 0 errors\n", "memcheck",
  )
  for
    log in [
      b"========= ERROR SUMMARY: 1 error\n", b"========= ERROR SUMMARY: 0 errors trailing-garbage\n",
      b"ERROR SUMMARY: 0 errors\nERROR SUMMARY: 0 errors\n",
    ] {
    try require_clean_sanitizer_log(log, "memcheck") catch {
      FusedCudaProbeRejected(_) => ()
      _ => fail("unexpected sanitizer rejection")
    } noraise {
      _ => fail("dirty sanitizer summary passed")
    }
  }
}
