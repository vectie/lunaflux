///|
async fn write_compile_candidate(
  path : String,
  bytes : Bytes,
) -> Unit raise Fp8V2CompileCandidateError {
  @fs.write_file(
    path,
    bytes,
    create_mode=CreateNew,
    permission=0o600,
    sync=Full,
  ) catch {
    _ => raise Publication
  }
}

///|
async fn emit_compile_candidates(
  arguments : ArrayView[String],
) -> Unit raise Fp8V2CompileCandidateError {
  guard arguments is [_, target, toolchain_sha256, output] &&
    output.has_prefix("/") &&
    !output.contains("\n") else {
    raise InvalidArguments
  }
  let set = prepare_compile_candidates(target, toolchain_sha256)
  write_compile_candidate(output + "/simple.cu", set.simple.source_bytes())
  write_compile_candidate(output + "/simple.recipe", set.simple.recipe_bytes())
  write_compile_candidate(output + "/gated.cu", set.gated.source_bytes())
  write_compile_candidate(output + "/gated.recipe", set.gated.recipe_bytes())
  write_compile_candidate(output + "/candidates.v1", set.manifest)
  println(
    "outcome=fp8-v2-candidates-emitted target=" +
    set.target +
    " model_content_sha256=" +
    set.model_content_sha256 +
    " model_plan_sha256=" +
    set.model_plan_sha256 +
    " candidate_manifest_sha256=" +
    set.manifest_sha256 +
    " runtime_authority=0 numerical_execution=0 readiness=0",
  )
}

///|
async fn main {
  try! emit_compile_candidates(@env.args())
}
