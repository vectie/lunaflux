///|
fn tensor_core_probe_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn tensor_core_probe_record(
  serial : @luna_tile_ir.LunaTileCudaTranslationUnit,
  candidate : @luna_tensor_core_candidate.LunaTileBf16Mma16x16x16Candidate,
) -> Bytes {
  @utf8.encode(
    "schema=lunaflux-lunatile-tensor-core-sm120-fixture.v1\n" +
    "program_sha256=" +
    candidate.program_digest().as_hex() +
    "\nserial_translation_sha256=" +
    serial.digest().as_hex() +
    "\nserial_source_sha256=" +
    serial.source_digest().as_hex() +
    "\nserial_entry_point=" +
    serial.entry_point() +
    "\nserial_launch=1,1,1;1,1,1\nserial_shared_memory_bytes=" +
    serial.memory_plan().shared_memory_bytes().to_string() +
    "\nparallel_plan_sha256=" +
    candidate.parallel_plan_digest().as_hex() +
    "\ntensor_core_candidate_sha256=" +
    candidate.digest().as_hex() +
    "\ntensor_core_source_sha256=" +
    candidate.source_digest().as_hex() +
    "\ntensor_core_entry_point=" +
    candidate.entry_point() +
    "\ntensor_core_launch=64,1,1;32,1,1\ntensor_core_shared_memory_bytes=0\n" +
    "shape=16,16,16\nleft_dtype=bf16\nright_dtype=bf16\n" +
    "accumulator_dtype=f32\noutput_dtype=f32\nlayout=row_major\n" +
    "cpu_oracle=independent-ordered-f32-v1\nserial_cuda_oracle=required\n" +
    "absolute_tolerance=0.001\nrelative_tolerance=0.0001\n" +
    "hostile_fixture=bounded-signed-zero-small-large-v1\ntarget=sm_120\n" +
    "maximum_registers_per_thread=128\nmaximum_static_shared_bytes=0\n" +
    "sass_instruction_family=observed-by-cuobjdump-and-nvdisasm\n" +
    "qualification_only=true\nmanifest_bindable=false\npromotion_authority=absent\n",
  )
}

///|
async fn tensor_core_probe_write_new(path : String, bytes : Bytes) -> Unit {
  @fs.write_file(
    path,
    bytes,
    create_mode=CreateNew,
    permission=0o600,
    sync=Full,
  )
}

///|
async fn export_tensor_core_probe(args : Array[String]) -> Unit {
  require_tensor_core_probe(
    args.length() == 3,
    "usage: luna_tensor_core_cuda_probe export ABSOLUTE_EMPTY_OUTPUT",
  )
  let output = args[2]
  require_tensor_core_probe(
    output.has_prefix("/") && @fs.exists(output),
    "export output must be an existing absolute directory",
  )
  let (serial, candidate) = tensor_core_probe_candidates()
  require_tensor_core_probe(
    !candidate.manifest_bindable() && !candidate.promotion_authorized(),
    "candidate unexpectedly carries authority",
  )
  let record = tensor_core_probe_record(serial, candidate)
  tensor_core_probe_write_new(output + "/serial.cu", serial.source_bytes())
  tensor_core_probe_write_new(
    output + "/serial.canonical",
    serial.canonical_bytes(),
  )
  tensor_core_probe_write_new(
    output + "/tensor_core.cu",
    candidate.source_bytes(),
  )
  tensor_core_probe_write_new(
    output + "/tensor_core.canonical",
    candidate.canonical_bytes(),
  )
  tensor_core_probe_write_new(output + "/fixture.v1", record)
  println(
    "outcome=lunatile-tensor-core-sm120-fixture-exported fixture_sha256=\{tensor_core_probe_sha256(record)} serial_source_sha256=\{serial.source_digest().as_hex()} tensor_core_source_sha256=\{candidate.source_digest().as_hex()} authority=qualification-only",
  )
}
