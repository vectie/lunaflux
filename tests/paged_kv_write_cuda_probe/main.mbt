///|
async fn kv_read(path : String) -> Bytes {
  @fs.read_file(path).binary()
}

///|
async fn kv_run(args : Array[String]) -> Unit {
  require_kv_probe(
    args.length() == 9,
    "usage: run TOOLCHAIN_SHA256 FIRST_CUBIN SECOND_CUBIN RECEIPT ORACLE_FIRST ORACLE_SECOND CYCLES",
  )
  let cycles = @string.parse_int(args[8], base=10) catch {
    _ => raise KvWriteProbeRejected("cycles is not an integer")
  }
  require_kv_probe(cycles > 0 && cycles <= 32, "cycles outside 1..32")
  let candidate = kv_probe_candidate(args[2])
  let compiled = kv_probe_compiled(
    candidate,
    kv_read(args[3]),
    kv_read(args[4]),
    kv_read(args[5]),
  )
  let oracle_first = kv_read(args[6])
  let oracle_second = kv_read(args[7])
  require_kv_probe(
    !oracle_first.is_empty() && oracle_first == oracle_second,
    "serial oracle CUBINs are not deterministic",
  )
  let balance = kv_probe_balance()
  let observation = run_kv_write_device_probe(
    candidate, compiled, oracle_first, cycles, balance,
  )
  require_kv_probe(balance.is_zero(), "CUDA resource balance is nonzero")
  println(
    "outcome=paged-kv-write-sm120-qualification-pass cycles=\{cycles} cases=6 launches=\{observation.launches} qualified_token_count=\{observation.token_count} key_mutations=\{observation.key_mutations} value_mutations=\{observation.value_mutations} observed_dispatch_canary=\{observation.observed_canary} prefill=pass decode=pass mixed=pass origin=pass page_tail=pass cross_page=pass multirow=pass full_grid=pass guards=pass non_target=pass scalar_oracle=pass serial_cuda_oracle=pass device_uuid=\{observation.device_uuid} device_pci=\{observation.device_pci} device_name_sha256=\{observation.device_name_sha256} device_total_memory_bytes=\{observation.device_total_memory_bytes} cuda_driver_version=\{observation.cuda_driver_version} resources=\{balance.render()} authority=qualification-only",
  )
}

///|
async fn kv_admit(args : Array[String]) -> Unit {
  require_kv_probe(
    args.length() == 11,
    "usage: admit TOOLCHAIN_SHA256 FIRST_CUBIN SECOND_CUBIN RECEIPT CANONICAL EXPECTED_EVIDENCE_SHA256 EXPECTED_POLICY_SHA256 FILES_SHA256 OUTER_SEAL_SHA256",
  )
  let candidate = kv_probe_candidate(args[2])
  let compiled = kv_probe_compiled(
    candidate,
    kv_read(args[3]),
    kv_read(args[4]),
    kv_read(args[5]),
  )
  let qualified = @luna_paged_kv_write_physical_evidence.admit_qualified_paged_kv_write_artifact(
    candidate,
    compiled,
    kv_read(args[6]),
    expected_evidence_digest=@luna_cuda_paged_kv_write_aot.PagedKvWriteDigest::from_sha256(
      args[7],
    ),
    expected_approved_policy_digest=@luna_cuda_paged_kv_write_aot.PagedKvWriteDigest::from_sha256(
      args[8],
    ),
  )
  require_kv_probe(
    qualified.files_manifest_digest().as_hex() == args[9] &&
    qualified.outer_seal_digest().as_hex() == args[10] &&
    !qualified.manifest_bindable() &&
    !qualified.promotion_authority() &&
    !qualified.runtime_dispatch_authority(),
    "qualified artifact seal or authority drifted",
  )
  println(
    "outcome=paged-kv-write-physical-evidence-admitted evidence_sha256=\{qualified.evidence_digest().as_hex()} authority=qualification-only",
  )
}

///|
async fn kv_bind(args : Array[String]) -> Unit {
  require_kv_probe(
    args.length() == 6,
    "usage: bind TOOLCHAIN_SHA256 FIRST_CUBIN SECOND_CUBIN RECEIPT",
  )
  let candidate = kv_probe_candidate(args[2])
  let compiled = kv_probe_compiled(
    candidate,
    kv_read(args[3]),
    kv_read(args[4]),
    kv_read(args[5]),
  )
  println(
    "schema=lunaflux-paged-kv-write-probe-binding.v1 binding_sha256=\{compiled.binding_digest().as_hex()} module_sha256=\{compiled.module_digest().as_hex()} receipt_sha256=\{compiled.compile_receipt_digest().as_hex()} authority=qualification-only",
  )
}

///|
async fn main {
  let args = @env.args()
  try {
    require_kv_probe(args.length() >= 2, "missing mode")
    match args[1] {
      "export" => kv_export(args)
      "receipt" => kv_materialize_receipt(args)
      "run" => kv_run(args)
      "admit" => kv_admit(args)
      "bind" => kv_bind(args)
      _ =>
        raise KvWriteProbeRejected(
          "mode must be export, receipt, bind, run, or admit",
        )
    }
  } catch {
    KvWriteProbeRejected(reason) => {
      println("outcome=paged-kv-write-probe-rejected reason=\{reason}")
      raise KvWriteProbeRejected(reason)
    }
    error => raise error
  }
}
