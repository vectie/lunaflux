///|
async fn export_fused_probe_inputs(args : Array[String]) -> Unit {
  require_fused_probe(
    args.length() == 4,
    "usage: fused_parallel_cuda_probe export TOOLCHAIN_SHA256 ABSOLUTE_NEW_OUTPUT",
  )
  require_fused_probe(
    args[2].length() == 64,
    "toolchain digest must be a SHA-256 hex string",
  )
  let set = fused_probe_candidate_set(args[2])
  require_fused_probe(
    !set.manifest_bindable(),
    "qualification set gained authority",
  )
  let evidence = @luna_fused_candidate_export.export_new(set, args[3])
  println(
    "outcome=fused-candidate-inputs-exported declaration_sha256=\{evidence.declaration_sha256()} inventory_sha256=\{evidence.inventory_sha256()} authority=qualification-only",
  )
}

///|
async fn read_probe_file(path : String) -> Bytes {
  @fs.read_file(path).binary()
}

///|
async fn campaign_bindings(
  args : Array[String],
) -> (
  @luna_fused_candidate_export.LunaFusedCandidateQualificationSet,
  FusedProbeBindings,
) {
  let set = fused_probe_candidate_set(args[2])
  let bindings = bind_fused_probe_artifacts(
    set,
    read_probe_file(args[3]),
    read_probe_file(args[4]),
    read_probe_file(args[5]),
    read_probe_file(args[6]),
    read_probe_file(args[7]),
    read_probe_file(args[8]),
  )
  (set, bindings)
}

///|
async fn render_fused_campaign(args : Array[String]) -> Unit {
  require_fused_probe(
    args.length() == 21,
    "usage: fused_parallel_cuda_probe render TOOLCHAIN_SHA256 QKV_FIRST QKV_SECOND QKV_RECEIPT RESIDUAL_FIRST RESIDUAL_SECOND RESIDUAL_RECEIPT RUNTIME_STDOUT RUNTIME_STDERR MEMCHECK_LOG RACECHECK_STDOUT RACECHECK_STDERR RACECHECK_LOG INITCHECK_STDOUT INITCHECK_STDERR INITCHECK_LOG AUDIT_RECORD ARTIFACT_SEAL_SHA256 EVIDENCE_SEAL_SHA256",
  )
  let (set, bindings) = campaign_bindings(args)
  let runtime_stdout = read_probe_file(args[9])
  let runtime_stderr = read_probe_file(args[10])
  let memcheck_log = read_probe_file(args[11])
  let race_stdout = read_probe_file(args[12])
  let race_stderr = read_probe_file(args[13])
  let race_log = read_probe_file(args[14])
  let init_stdout = read_probe_file(args[15])
  let init_stderr = read_probe_file(args[16])
  let init_log = read_probe_file(args[17])
  require_fused_probe(
    runtime_stderr.length() == 0 &&
    race_stderr.length() == 0 &&
    init_stderr.length() == 0,
    "physical runtime or racecheck emitted stderr",
  )
  require_fused_probe(
    runtime_stdout == race_stdout && runtime_stdout == init_stdout,
    "sanitizer physical observations differ",
  )
  require_clean_sanitizer_log(memcheck_log, "memcheck")
  require_clean_sanitizer_log(race_log, "racecheck")
  require_clean_sanitizer_log(init_log, "initcheck")
  let observation = parse_fused_campaign_observation(runtime_stdout, set)
  let canonical = render_fused_campaign_canonical(
    set,
    bindings,
    observation,
    read_probe_file(args[18]),
    campaign_digest(args[19]),
    campaign_digest(args[20]),
  )
  let document = @utf8.decode(canonical)
  println(document[:document.length() - 1])
}

///|
async fn admit_fused_campaign(args : Array[String]) -> Unit {
  require_fused_probe(
    args.length() == 12,
    "usage: fused_parallel_cuda_probe admit TOOLCHAIN_SHA256 QKV_FIRST QKV_SECOND QKV_RECEIPT RESIDUAL_FIRST RESIDUAL_SECOND RESIDUAL_RECEIPT CANONICAL EXPECTED_SHA256 EXPECTED_APPROVED_POLICY_SHA256",
  )
  let (set, bindings) = campaign_bindings(args)
  let canonical = read_probe_file(args[9])
  let expected = campaign_digest(args[10])
  let admitted = @luna_fused_physical_evidence.admit_fused_physical_evidence(
    set.qkv_candidate(),
    bindings.qkv,
    set.residual_norm_candidate(),
    bindings.residual,
    canonical,
    expected_digest=expected,
    expected_approved_policy_digest=campaign_digest(args[11]),
  )
  require_fused_probe(
    admitted.digest() == expected &&
    !admitted.manifest_bindable() &&
    !admitted.promotion_authorized(),
    "admitted physical evidence gained authority or changed digest",
  )
  println(
    "outcome=fused-physical-evidence-admitted evidence_sha256=\{admitted.digest().as_hex()} artifact_seal_sha256=\{admitted.artifact_directory_seal_digest().as_hex()} evidence_seal_sha256=\{admitted.evidence_directory_seal_digest().as_hex()} authority=observation-only",
  )
}

///|
async fn run_fused_probe(args : Array[String]) -> Unit {
  require_fused_probe(
    args.length() == 13,
    "usage: fused_parallel_cuda_probe run TOOLCHAIN_SHA256 QKV_FIRST QKV_SECOND QKV_RECEIPT RESIDUAL_FIRST RESIDUAL_SECOND RESIDUAL_RECEIPT ORACLE_QKV ORACLE_ROPE ORACLE_KV_WRITE CYCLES",
  )
  let cycles = @string.parse_int(args[12], base=10) catch {
    _ => raise FusedCudaProbeRejected("CYCLES must be an integer")
  }
  require_fused_probe(cycles > 0 && cycles <= 32, "CYCLES must be in 1..32")
  let set = fused_probe_candidate_set(args[2])
  let bindings = bind_fused_probe_artifacts(
    set,
    read_probe_file(args[3]),
    read_probe_file(args[4]),
    read_probe_file(args[5]),
    read_probe_file(args[6]),
    read_probe_file(args[7]),
    read_probe_file(args[8]),
  )
  let balance = fused_probe_resource_balance()
  let oracle_modules = fused_probe_oracle_modules(
    read_probe_file(args[9]),
    read_probe_file(args[10]),
    read_probe_file(args[11]),
  )
  let observation = run_fused_device_probe(
    bindings, oracle_modules, set, cycles, balance,
  )
  require_fused_probe(balance.is_zero(), "CUDA resource balance is nonzero")
  println(
    "outcome=fused-parallel-sm120-qualification-pass cycles=\{cycles} shapes=4 launches=\{observation.launches} qkv_max_abs_error_ppb=\{observation.maximum_qkv_absolute_error_ppb} qkv_max_relative_error_ppb=\{observation.maximum_qkv_relative_error_ppb} qkv_observed_dispatch_canary=\{observation.qkv_dispatch_canary} residual_max_abs_error_ppb=\{observation.maximum_residual_absolute_error_ppb} residual_max_relative_error_ppb=\{observation.maximum_residual_relative_error_ppb} residual_observed_dispatch_canary=\{observation.residual_dispatch_canary} standalone_qkv_oracle=pass deterministic_cubins=2x5 target=sm_120 device_uuid=\{observation.device_uuid} device_pci=\{observation.device_pci} device_name_sha256=\{observation.device_name_sha256} device_total_memory_bytes=\{observation.device_total_memory_bytes} cuda_driver_version=\{observation.cuda_driver_version} resources=\{balance.render()} authority=qualification-only",
  )
}

///|
async fn run_fused_production_probe(args : Array[String]) -> Unit {
  require_fused_probe(
    args.length() == 15,
    "usage: fused_parallel_cuda_probe run-production TOOLCHAIN_SHA256 QKV_FIRST QKV_SECOND QKV_RECEIPT READONLY_FIRST READONLY_SECOND RESIDUAL_FIRST RESIDUAL_SECOND RESIDUAL_RECEIPT ORACLE_QKV ORACLE_ROPE ORACLE_ATTENTION CYCLES",
  )
  let cycles = @string.parse_int(args[14], base=10) catch {
    _ => raise FusedCudaProbeRejected("CYCLES must be an integer")
  }
  require_fused_probe(cycles > 0 && cycles <= 32, "CYCLES must be in 1..32")
  let (context, candidates) = fused_production_probe_candidates(args[2])
  let bindings = bind_fused_production_probe_artifacts(
    candidates,
    read_probe_file(args[3]),
    read_probe_file(args[4]),
    read_probe_file(args[5]),
    read_probe_file(args[6]),
    read_probe_file(args[7]),
    read_probe_file(args[8]),
    read_probe_file(args[9]),
    read_probe_file(args[10]),
  )
  let balance = fused_probe_resource_balance()
  let observation = run_fused_production_device_probe(
    bindings,
    read_probe_file(args[11]),
    read_probe_file(args[12]),
    read_probe_file(args[13]),
    context,
    candidates,
    cycles,
    balance,
  )
  require_fused_probe(balance.is_zero(), "CUDA resource balance is nonzero")
  println(
    "outcome=fused-production-v2-sm120-qualification-pass cycles=\{cycles} shapes=4 launches=\{observation.launches} maximum_absolute_error_ppb=\{observation.maximum_absolute_error_ppb} maximum_relative_error_ppb=\{observation.maximum_relative_error_ppb} qkv_abi=QkvPositionedRopePagedKvWriteProductionAbiV2 readonly_abi=RotatedQueryPagedKvReadOnlyProductionRawPointerV2 residual_abi=ResidualRmsNormProductionAbiV2 executor=ordered-kernel-executor graph_policy=capture-required graph_mode=captured oracle=standalone-cuda device_uuid=\{observation.device_uuid} device_pci=\{observation.device_pci} device_name_sha256=\{observation.device_name_sha256} device_total_memory_bytes=\{observation.device_total_memory_bytes} cuda_driver_version=\{observation.cuda_driver_version} resources=\{balance.render()} authority=qualification-only manifest_bindable=0 promotion_authority=0",
  )
}

///|
async fn main {
  let args = @env.args()
  try {
    require_fused_probe(args.length() >= 2, "missing probe mode")
    match args[1] {
      "export" => export_fused_probe_inputs(args)
      "export-production" => export_fused_production_probe_inputs(args)
      "run" => run_fused_probe(args)
      "run-production" => run_fused_production_probe(args)
      "render" => render_fused_campaign(args)
      "admit" => admit_fused_campaign(args)
      _ =>
        raise FusedCudaProbeRejected(
          "probe mode must be export, export-production, run, run-production, render, or admit",
        )
    }
  } catch {
    FusedCudaProbeRejected(reason) => {
      println("outcome=fused-parallel-qualification-rejected reason=\{reason}")
      raise FusedCudaProbeRejected(reason)
    }
    error => raise error
  }
}
