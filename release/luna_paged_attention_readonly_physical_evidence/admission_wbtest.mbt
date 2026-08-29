///|
priv struct ReadonlyEvidenceBundle {
  result : Bytes
  manifest : Bytes
  outer : Bytes
  outer_digest : PagedAttentionReadOnlyPhysicalEvidenceDigest
  policy : Bytes
  policy_digest : PagedAttentionReadOnlyPhysicalEvidenceDigest
}

///|
fn readonly_evidence_label(
  label : String,
) -> PagedAttentionReadOnlyPhysicalEvidenceDigest {
  readonly_physical_digest(@utf8.encode(label))
}

///|
fn readonly_evidence_manifest(
  candidate : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCudaCandidate,
  compiled : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCompiledBinding,
  driver_record : PagedAttentionReadOnlyPhysicalEvidenceDigest,
  device_record : PagedAttentionReadOnlyPhysicalEvidenceDigest,
  versions : Array[PagedAttentionReadOnlyPhysicalEvidenceDigest],
  probe_source : PagedAttentionReadOnlyPhysicalEvidenceDigest,
  probe_binary : PagedAttentionReadOnlyPhysicalEvidenceDigest,
  cu_sass : PagedAttentionReadOnlyPhysicalEvidenceDigest,
  nv_sass : PagedAttentionReadOnlyPhysicalEvidenceDigest,
  sanitizer : PagedAttentionReadOnlyPhysicalEvidenceDigest,
  policy_digest : PagedAttentionReadOnlyPhysicalEvidenceDigest,
) -> Bytes raise {
  let fallback = candidate.reference_fallback()
  let source = readonly_physical_read_digest(candidate.source_digest().as_hex())
  let cubin = readonly_physical_read_digest(compiled.module_digest().as_hex())
  let entries : Array[(String, PagedAttentionReadOnlyPhysicalEvidenceDigest)] = [
    (
      "export/readonly.canonical",
      readonly_physical_read_digest(candidate.candidate_digest().as_hex()),
    ),
    ("export/readonly.cu", source),
    (
      "export/readonly.recipe",
      readonly_physical_read_digest(candidate.recipe_digest().as_hex()),
    ),
    (
      "export/fallback.cu",
      readonly_physical_read_digest(fallback.source_digest().as_hex()),
    ),
    (
      "export/fallback.recipe",
      readonly_physical_read_digest(fallback.recipe_digest().as_hex()),
    ),
    ("builds/first/kernel.cubin", cubin),
    ("builds/second/kernel.cubin", cubin),
    ("builds/resource/kernel.cubin", cubin),
    ("builds/first/kernel.cu", source),
    ("builds/second/kernel.cu", source),
    ("builds/resource/kernel.cu", source),
    (
      "measurements/compile-receipt.v1",
      readonly_physical_read_digest(compiled.compile_receipt_digest().as_hex()),
    ),
    ("measurements/device-identity.v1", device_record),
    ("measurements/driver-before.v1", driver_record),
    ("measurements/driver-after.v1", driver_record),
    ("measurements/nvcc-version-before.stdout", versions[0]),
    ("measurements/nvcc-version-after.stdout", versions[0]),
    ("measurements/ptxas-version-before.stdout", versions[1]),
    ("measurements/ptxas-version-after.stdout", versions[1]),
    ("measurements/cuobjdump-version-before.stdout", versions[2]),
    ("measurements/cuobjdump-version-after.stdout", versions[2]),
    ("measurements/nvdisasm-version-before.stdout", versions[3]),
    ("measurements/nvdisasm-version-after.stdout", versions[3]),
    ("measurements/compute-sanitizer-version-before.stdout", versions[4]),
    ("measurements/compute-sanitizer-version-after.stdout", versions[4]),
    ("measurements/cuobjdump-sass.stdout", cu_sass),
    ("measurements/nvdisasm-sass.stdout", nv_sass),
    ("measurements/memcheck.log", sanitizer),
    ("measurements/racecheck.log", sanitizer),
    ("measurements/initcheck.log", sanitizer),
    ("probe/probe.cu", probe_source),
    ("probe/physical_probe", probe_binary),
    ("policy/approved.v1", policy_digest),
  ]
  entries.sort_by((left, right) => String::compare(left.0, right.0))
  let output = StringBuilder()
  for entry in entries {
    output.write_string(entry.1.as_hex())
    output.write_string("  ")
    output.write_string(entry.0)
    output.write_string("\n")
  }
  @utf8.encode(output.to_string())
}

///|
fn readonly_evidence_bundle(
  candidate : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCudaCandidate,
  compiled : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCompiledBinding,
) -> ReadonlyEvidenceBundle raise {
  let fallback = candidate.reference_fallback()
  let driver = readonly_physical_read_digest(
    fallback.compiler_policy().identity().toolchain_digest_sha256(),
  )
  let driver_record = readonly_evidence_label("driver record")
  let device_record = readonly_evidence_label("device record")
  let versions = [
    readonly_evidence_label("nvcc version"),
    readonly_evidence_label("ptxas version"),
    readonly_evidence_label("cuobjdump version"),
    readonly_evidence_label("nvdisasm version"),
    readonly_evidence_label("sanitizer version"),
  ]
  let probe_source = readonly_evidence_label("probe source")
  let probe_binary = readonly_evidence_label("probe binary")
  let cu_sass = readonly_evidence_label("cuobjdump sass")
  let nv_sass = readonly_evidence_label("nvdisasm sass")
  let sanitizer = readonly_evidence_label("sanitizer clean")
  let arbitrary = readonly_evidence_label("tool binary")
  let policy = @utf8.encode(
    "schema=lunaflux-paged-attention-readonly-approved-physical-policy.v1\n" +
    "target=sm_120\ncompiler_version=13.1.115\n" +
    "nvcc_sha256=" +
    arbitrary.as_hex() +
    "\nptxas_sha256=" +
    arbitrary.as_hex() +
    "\ncuobjdump_sha256=" +
    arbitrary.as_hex() +
    "\nnvdisasm_sha256=" +
    arbitrary.as_hex() +
    "\ncompute_sanitizer_sha256=" +
    arbitrary.as_hex() +
    "\ndriver_identity_sha256=" +
    driver.as_hex() +
    "\ndriver_record_sha256=" +
    driver_record.as_hex() +
    "\ndevice_uuid=GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6\n" +
    "device_pci_bus_id=00000000:17:00.0\n" +
    "device_name_hex=746573742d736d313230\n" +
    "device_total_memory_bytes=25769803776\ndevice_compute=12.0\n" +
    "policy_authority=deployment-approved\n",
  )
  let policy_digest = readonly_physical_digest(policy)
  let manifest = readonly_evidence_manifest(
    candidate, compiled, driver_record, device_record, versions, probe_source, probe_binary,
    cu_sass, nv_sass, sanitizer, policy_digest,
  )
  let manifest_digest = readonly_physical_digest(manifest)
  let fields = [
    ("schema", "lunaflux-paged-attention-readonly-physical-campaign.v1"),
    ("outcome", "paged-attention-readonly-physical-campaign-pass"),
    ("approved_policy_sha256", policy_digest.as_hex()),
    ("candidate_sha256", candidate.candidate_digest().as_hex()),
    ("source_sha256", candidate.source_digest().as_hex()),
    ("recipe_sha256", candidate.recipe_digest().as_hex()),
    ("fallback_source_sha256", fallback.source_digest().as_hex()),
    ("fallback_recipe_sha256", fallback.recipe_digest().as_hex()),
    ("compiled_binding_sha256", compiled.binding_digest().as_hex()),
    ("cubin_sha256", compiled.module_digest().as_hex()),
    ("compile_receipt_sha256", compiled.compile_receipt_digest().as_hex()),
    ("operation_id", "2"),
    ("target", "sm_120"),
    ("profile", "4,260,132"),
    ("kv_layout", "layer_major_split_key_value_v1"),
    ("kv_layout_stable_version", "1"),
    ("tokens_per_page", "2"),
    ("page_stride_bytes", "256"),
    ("physical_page_capacity", "160"),
    ("input_row_width", "8"),
    ("input_activation_ref", "1"),
    ("function_symbol", candidate.function_symbol()),
    ("raw_pointer_abi", candidate.kernel_abi().canonical_name()),
    (
      "selection_precondition", "standalone-positioned-rope-paged-kvwrite-complete-v1",
    ),
    ("dispatch_canary_per_token", "65624"),
    ("dispatch_canary_cell_count", "260"),
    ("dispatch_canary_publication", "exactly-once-after-output"),
    ("dispatch_canary_exact", "true"),
    ("dispatch_canary_tail_zero", "true"),
    ("dispatch_canary_sum_checked", "true"),
    ("kv_cache_mutation", "none"),
    ("driver_identity_sha256", driver.as_hex()),
    ("driver_record_sha256", driver_record.as_hex()),
    ("device_record_sha256", device_record.as_hex()),
    ("compiler_version", "13.1.115"),
    ("device_uuid", "GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6"),
    ("device_pci_bus_id", "00000000:17:00.0"),
    ("device_name_hex", "746573742d736d313230"),
    ("device_total_memory_bytes", "25769803776"),
    ("nvcc_sha256", arbitrary.as_hex()),
    ("ptxas_sha256", arbitrary.as_hex()),
    ("cuobjdump_sha256", arbitrary.as_hex()),
    ("nvdisasm_sha256", arbitrary.as_hex()),
    ("compute_sanitizer_sha256", arbitrary.as_hex()),
    ("nvcc_version_sha256", versions[0].as_hex()),
    ("ptxas_version_sha256", versions[1].as_hex()),
    ("cuobjdump_version_sha256", versions[2].as_hex()),
    ("nvdisasm_version_sha256", versions[3].as_hex()),
    ("compute_sanitizer_version_sha256", versions[4].as_hex()),
    ("probe_source_sha256", probe_source.as_hex()),
    ("probe_binary_sha256", probe_binary.as_hex()),
    ("cuobjdump_sass_sha256", cu_sass.as_hex()),
    ("cuobjdump_sass_instruction_count", "97"),
    ("nvdisasm_sass_sha256", nv_sass.as_hex()),
    ("nvdisasm_sass_instruction_count", "97"),
    ("sass_global_load_observed", "true"),
    ("sass_global_store_observed", "true"),
    ("registers_per_thread", "64"),
    ("register_bound", "128"),
    ("static_shared_bytes", "0"),
    ("static_shared_bound", "0"),
    ("stack_bytes", "0"),
    ("local_bytes", "0"),
    ("spill_store_bytes", "0"),
    ("spill_load_bytes", "0"),
    ("resource_bounds_passed", "true"),
    ("case_families", "origin,page-tail,cross-page,multirow,long-context"),
    ("scheduler_modes", "prefill-only,decode-only,mixed-prefill-decode"),
    ("dispatch_grid_x", "260"),
    ("numeric_case_count", "144"),
    ("output_dtype", "bf16"),
    ("absolute_tolerance", "0.0078125"),
    ("relative_tolerance", "0.01"),
    ("cpu_vs_candidate_max_abs_error", "0"),
    ("serial_vs_candidate_max_abs_error", "0"),
    ("cpu_oracle", "independent-ordered-f32-v1"),
    ("serial_cuda_oracle", "independent-ordered-f32-kernel-v1"),
    ("cache_snapshot_bytes", "81920"),
    ("cache_snapshot_unchanged", "true"),
    ("input_guards_unchanged", "true"),
    ("output_guards_unchanged", "true"),
    ("dispatch_symbol_resolved", "true"),
    ("memcheck_log_sha256", sanitizer.as_hex()),
    ("memcheck_error_summary_count", "1"),
    ("memcheck_errors", "0"),
    ("racecheck_log_sha256", sanitizer.as_hex()),
    ("racecheck_error_summary_count", "1"),
    ("racecheck_errors", "0"),
    ("initcheck_log_sha256", sanitizer.as_hex()),
    ("initcheck_error_summary_count", "1"),
    ("initcheck_errors", "0"),
    ("cleanup_balance", "module0,allocation0,device_bytes0,pending0,cleanup0"),
    ("physical_cuda_observed", "true"),
    ("synthetic_test_only", "false"),
    ("qualification_only", "true"),
    ("manifest_bindable", "false"),
    ("promotion_authority", "absent"),
    ("evidence_files_manifest_sha256", manifest_digest.as_hex()),
  ]
  let output = StringBuilder()
  for field in fields {
    output.write_string(field.0)
    output.write_string("=")
    output.write_string(field.1)
    output.write_string("\n")
  }
  let result = @utf8.encode(output.to_string())
  let outer = @utf8.encode(
    manifest_digest.as_hex() +
    "  FILES.sha256\n" +
    readonly_physical_digest(result).as_hex() +
    "  RESULT.txt\n",
  )
  {
    result,
    manifest,
    outer,
    outer_digest: readonly_physical_digest(outer),
    policy,
    policy_digest,
  }
}

///|
fn readonly_evidence_admit(
  candidate : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCudaCandidate,
  compiled : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCompiledBinding,
  bundle : ReadonlyEvidenceBundle,
) -> AdmittedPagedAttentionReadOnlyPhysicalEvidence raise {
  admit_paged_attention_readonly_physical_evidence(
    candidate,
    compiled,
    admit_approved_paged_attention_readonly_physical_policy(
      bundle.policy,
      bundle.policy_digest,
    ),
    bundle.result,
    bundle.manifest,
    bundle.outer,
    bundle.outer_digest,
    PagedAttentionReadOnlyPhysicalEvidenceLimits::new(
      maximum_result_bytes=131072,
      maximum_manifest_bytes=524288,
    ),
  )
}

///|
fn readonly_evidence_reseal(
  baseline : ReadonlyEvidenceBundle,
  result : Bytes,
  manifest : Bytes,
) -> ReadonlyEvidenceBundle {
  let outer = @utf8.encode(
    readonly_physical_digest(manifest).as_hex() +
    "  FILES.sha256\n" +
    readonly_physical_digest(result).as_hex() +
    "  RESULT.txt\n",
  )
  {
    ..baseline,
    result,
    manifest,
    outer,
    outer_digest: readonly_physical_digest(outer),
  }
}

///|
test "exact sealed read-only physical evidence admits and remains inert" {
  let candidate = readonly_evidence_candidate()
  let compiled = readonly_evidence_compiled(candidate)
  let bundle = readonly_evidence_bundle(candidate, compiled)
  assert_eq(@utf8.decode(bundle.result).split("\n").length(), 99)
  let evidence = readonly_evidence_admit(candidate, compiled, bundle)
  assert_eq(evidence.candidate_digest(), candidate.candidate_digest().as_hex())
  assert_eq(
    evidence.compiled_binding_digest(),
    compiled.binding_digest().as_hex(),
  )
  assert_eq(evidence.input_activation(), @plan.ActivationRef::new(1))
  assert_eq(evidence.input_row_width(), 8)
  assert_eq(evidence.dispatch_canary_per_token(), 65624)
  assert_eq(evidence.dispatch_canary_cell_count(), 260)
  assert_eq(evidence.numeric_case_count(), 144)
  assert_false(evidence.manifest_bindable())
  assert_false(evidence.promotion_authorized())
  let qualified = bind_qualified_paged_attention_readonly_artifact(
    candidate, compiled, evidence,
  )
  assert_true(qualified.physically_qualified())
  assert_false(qualified.manifest_bindable())
  assert_false(qualified.promotion_authorized())
}

///|
test "read-only admission rejects cache canary identity numeric sanitizer and authority substitution" {
  let candidate = readonly_evidence_candidate()
  let compiled = readonly_evidence_compiled(candidate)
  let baseline = readonly_evidence_bundle(candidate, compiled)
  for
    substitution in [
      ("input_row_width=8", "input_row_width=4"),
      ("input_activation_ref=1", "input_activation_ref=9"),
      (
        "device_uuid=GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6", "device_uuid=GPU-11111111-1111-1111-1111-111111111111",
      ),
      (
        "device_pci_bus_id=00000000:17:00.0", "device_pci_bus_id=00000000:18:00.0",
      ),
      ("dispatch_canary_per_token=65624", "dispatch_canary_per_token=0"),
      ("dispatch_canary_tail_zero=true", "dispatch_canary_tail_zero=false"),
      ("cache_snapshot_unchanged=true", "cache_snapshot_unchanged=false"),
      ("input_guards_unchanged=true", "input_guards_unchanged=false"),
      ("cpu_vs_candidate_max_abs_error=0", "cpu_vs_candidate_max_abs_error=1"),
      ("memcheck_errors=0", "memcheck_errors=1"),
      ("manifest_bindable=false", "manifest_bindable=true"),
      ("physical_cuda_observed=true", "physical_cuda_observed=false"),
      ("synthetic_test_only=false", "synthetic_test_only=true"),
    ] {
    let result = @utf8.encode(
      @utf8.decode(baseline.result).replace(
        old=substitution.0,
        new=substitution.1,
      ),
    )
    let resealed = readonly_evidence_reseal(baseline, result, baseline.manifest)
    try readonly_evidence_admit(candidate, compiled, resealed) catch {
      PagedAttentionReadOnlyPhysicalEvidenceInvalid(_)
      | PagedAttentionReadOnlyPhysicalEvidenceMismatch(_) => ()
      _ => fail("unexpected hostile evidence rejection")
    } noraise {
      _ => fail("hostile read-only physical evidence was admitted")
    }
  }
}

///|
test "read-only admission requires independently pinned approved policy digest" {
  let candidate = readonly_evidence_candidate()
  let compiled = readonly_evidence_compiled(candidate)
  let baseline = readonly_evidence_bundle(candidate, compiled)
  let changed_policy = @utf8.encode(
    @utf8.decode(baseline.policy).replace(
      old="device_total_memory_bytes=25769803776",
      new="device_total_memory_bytes=17179869184",
    ),
  )
  try
    admit_approved_paged_attention_readonly_physical_policy(
      changed_policy,
      baseline.policy_digest,
    )
  catch {
    PagedAttentionReadOnlyPhysicalEvidenceMismatch(Toolchain) => ()
    _ => fail("unexpected out-of-band policy digest rejection")
  } noraise {
    _ => fail("caller-written policy bypassed independently pinned digest")
  }
  let changed_digest = readonly_physical_digest(changed_policy)
  let changed = admit_approved_paged_attention_readonly_physical_policy(
    changed_policy, changed_digest,
  )
  try
    admit_paged_attention_readonly_physical_evidence(
      candidate,
      compiled,
      changed,
      baseline.result,
      baseline.manifest,
      baseline.outer,
      baseline.outer_digest,
      PagedAttentionReadOnlyPhysicalEvidenceLimits::new(
        maximum_result_bytes=131072,
        maximum_manifest_bytes=524288,
      ),
    )
  catch {
    PagedAttentionReadOnlyPhysicalEvidenceMismatch(Toolchain) => ()
    _ => fail("unexpected foreign approved-policy rejection")
  } noraise {
    _ => fail("sealed result replayed under a foreign approved policy")
  }
}

///|
test "read-only admission rejects stale outer seal and manifest replay" {
  let candidate = readonly_evidence_candidate()
  let compiled = readonly_evidence_compiled(candidate)
  let baseline = readonly_evidence_bundle(candidate, compiled)
  let changed = @utf8.encode(
    @utf8.decode(baseline.result).replace(
      old="sass_global_load_observed=true",
      new="sass_global_load_observed=false",
    ),
  )
  try
    readonly_evidence_admit(candidate, compiled, { ..baseline, result: changed })
  catch {
    PagedAttentionReadOnlyPhysicalEvidenceMismatch(Seal) => ()
    _ => fail("unexpected stale-seal rejection")
  } noraise {
    _ => fail("mutated result passed stale outer seal")
  }
  let manifest = @utf8.encode(
    @utf8.decode(baseline.manifest).replace_all(
      old="measurements/memcheck.log",
      new="measurements/memcheck-replay.log",
    ),
  )
  let resealed = readonly_evidence_reseal(baseline, baseline.result, manifest)
  try readonly_evidence_admit(candidate, compiled, resealed) catch {
    PagedAttentionReadOnlyPhysicalEvidenceMismatch(Seal) => ()
    _ => fail("unexpected manifest-replay rejection")
  } noraise {
    _ => fail("critical manifest path replay was admitted")
  }
}
