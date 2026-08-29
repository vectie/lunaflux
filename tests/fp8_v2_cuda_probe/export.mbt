///|
fn fp8_probe_sha256(bytes : BytesView) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
async fn fp8_probe_write_new(path : String, bytes : Bytes) -> Unit {
  @fs.write_file(
    path,
    bytes,
    create_mode=CreateNew,
    permission=0o600,
    sync=Full,
  )
}

///|
async fn export_fp8_probe(args : Array[String]) -> Unit {
  require_fp8_probe(
    args.length() == 4,
    "usage: fp8_v2_cuda_probe export TOOLCHAIN_SHA256 ABSOLUTE_EXISTING_DIR",
  )
  require_fp8_probe(
    args[3].has_prefix("/") && @fs.exists(args[3]),
    "export directory must be absolute and existing",
  )
  let candidates = fp8_probe_candidates(args[2])
  let record = @utf8.encode(
    "schema=lunaflux-fp8-v2-sm120-probe-fixture.v1\n" +
    "model_content_sha256=" +
    candidates.plan.identity().content().as_hex() +
    "\nmodel_plan_sha256=" +
    candidates.plan.identity().plan().as_hex() +
    "\ntarget=sm_120\ncompiler_version=13.1.115\ntoolchain_sha256=" +
    args[2] +
    "\nsimple_operation_id=2\nsimple_symbol=" +
    candidates.simple.function_symbol() +
    "\nsimple_source_sha256=" +
    candidates.simple.source_digest().as_hex() +
    "\nsimple_recipe_sha256=" +
    candidates.simple.recipe_digest().as_hex() +
    "\ngated_operation_id=8\ngated_symbol=" +
    candidates.gated.function_symbol() +
    "\ngated_source_sha256=" +
    candidates.gated.source_digest().as_hex() +
    "\ngated_recipe_sha256=" +
    candidates.gated.recipe_digest().as_hex() +
    "\nabi=production-paged-v4-operands-v2\n" +
    "cpu_oracle=independent-ordered-f32-v1\n" +
    "absolute_tolerance=0.03125\nrelative_tolerance=0.015625\n" +
    "qualification_only=true\nmanifest_bindable=false\nreadiness=false\n",
  )
  fp8_probe_write_new(args[3] + "/simple.cu", candidates.simple.source_bytes())
  fp8_probe_write_new(
    args[3] + "/simple.recipe",
    candidates.simple.recipe_bytes(),
  )
  fp8_probe_write_new(args[3] + "/gated.cu", candidates.gated.source_bytes())
  fp8_probe_write_new(
    args[3] + "/gated.recipe",
    candidates.gated.recipe_bytes(),
  )
  fp8_probe_write_new(args[3] + "/fixture.v1", record)
  println(
    "outcome=fp8-v2-sm120-fixture-exported fixture_sha256=\{fp8_probe_sha256(record)} simple_source_sha256=\{candidates.simple.source_digest().as_hex()} gated_source_sha256=\{candidates.gated.source_digest().as_hex()} authority=qualification-only",
  )
}
