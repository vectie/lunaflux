///|
fn kv_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn kv_builder_receipt(
  candidate : @luna_cuda_paged_kv_write_aot.PagedKvWriteCudaCandidate,
  module_bytes : Bytes,
  driver : String,
) -> Bytes {
  let target = candidate.target()
  let module_sha = kv_sha256(module_bytes)
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
    driver +
    "\n" +
    "target=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\n" +
    "function_symbol=" +
    candidate.function_symbol() +
    "\n" +
    "artifact_sha256=" +
    module_sha +
    "\n" +
    "first_build_sha256=" +
    module_sha +
    "\n" +
    "second_build_sha256=" +
    module_sha +
    "\n" +
    "deterministic=1\n",
  )
}

///|
async fn kv_materialize_receipt(args : Array[String]) -> Unit {
  require_kv_probe(
    args.length() == 6,
    "usage: receipt TOOLCHAIN_SHA256 CUBIN BUILDER_RECEIPT TYPED_RECEIPT",
  )
  let candidate = kv_probe_candidate(args[2])
  let module_bytes = kv_read(args[3])
  let builder = kv_read(args[4])
  let text = @utf8.decode(builder) catch {
    _ => raise KvWriteProbeRejected("builder receipt is not UTF-8")
  }
  let mut driver : String? = None
  for line in text.split("\n") {
    if line.has_prefix("driver_identity_sha256=") {
      driver = Some(line[23:].to_owned())
    }
  }
  guard driver is Some(driver) else {
    raise KvWriteProbeRejected("builder receipt lacks driver identity")
  }
  require_kv_probe(
    builder == kv_builder_receipt(candidate, module_bytes, driver),
    "builder receipt identity drifted",
  )
  kv_write_new(
    args[5],
    @luna_cuda_paged_kv_write_aot.positioned_paged_kv_write_compile_receipt(
      candidate, driver, module_bytes,
    ),
  )
  println(
    "outcome=paged-kv-write-typed-receipt-materialized authority=qualification-only",
  )
}
