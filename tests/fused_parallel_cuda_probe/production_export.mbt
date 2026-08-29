///|
async fn production_export_write_new(path : String, bytes : Bytes) -> Unit {
  @fs.write_file(
    path,
    bytes,
    create_mode=CreateNew,
    permission=0o600,
    sync=Full,
  )
}

///|
async fn production_export_cleanup(path : String) -> Unit noraise {
  @fs.rmdir(path, recursive=true) catch {
    _ => ()
  }
}

///|
fn production_export_identities(
  context : FusedProbeCandidateContext,
  production : FusedProductionProbeCandidates,
) -> Bytes {
  let output = StringBuilder()
  output.write_string("schema=lunaflux-fused-production-v2-probe-inputs.v1\n")
  output.write_string("target=sm_120\n")
  output.write_string("executor_identity=ordered-kernel-executor.v1\n")
  output.write_string(
    "bootstrap_join=production-compiled-candidate-source-and-symbol.v1\n",
  )
  output.write_string("graph_policy=capture-required\n")
  output.write_string("qkv_abi=QkvPositionedRopePagedKvWriteProductionAbiV2\n")
  output.write_string(
    "readonly_abi=RotatedQueryPagedKvReadOnlyProductionRawPointerV2\n",
  )
  output.write_string("residual_abi=ResidualRmsNormProductionAbiV2\n")
  output.write_string(
    "production_qkv_source_sha256=" +
    production.qkv.source_digest().as_hex() +
    "\n",
  )
  output.write_string(
    "production_qkv_candidate_sha256=" +
    production.qkv.candidate_digest().as_hex() +
    "\n",
  )
  output.write_string(
    "production_qkv_symbol=" + production.qkv.function_symbol() + "\n",
  )
  output.write_string(
    "production_qkv_recipe_sha256=" +
    production.qkv.recipe_digest().as_hex() +
    "\n",
  )
  output.write_string(
    "production_readonly_source_sha256=" +
    production.readonly_candidate.source_digest().as_hex() +
    "\n",
  )
  output.write_string(
    "production_readonly_candidate_sha256=" +
    production.readonly_candidate.candidate_digest().as_hex() +
    "\n",
  )
  output.write_string(
    "production_readonly_symbol=" +
    production.readonly_candidate.function_symbol() +
    "\n",
  )
  output.write_string(
    "production_readonly_recipe_sha256=" +
    production.readonly_candidate.recipe_digest().as_hex() +
    "\n",
  )
  output.write_string(
    "production_residual_source_sha256=" +
    production.residual.source_digest().as_hex() +
    "\n",
  )
  output.write_string(
    "production_residual_candidate_sha256=" +
    production.residual.candidate_digest().as_hex() +
    "\n",
  )
  output.write_string(
    "production_residual_symbol=" + production.residual.function_symbol() + "\n",
  )
  output.write_string(
    "production_residual_recipe_sha256=" +
    production.residual.recipe_digest().as_hex() +
    "\n",
  )
  output.write_string(
    "oracle_qkv_source_sha256=" +
    context.qualification.qkv_fallback().source_digest().as_hex() +
    "\n",
  )
  output.write_string(
    "oracle_rope_source_sha256=" +
    context.qualification.rope_fallback().source_digest().as_hex() +
    "\n",
  )
  output.write_string(
    "oracle_attention_source_sha256=" +
    production.oracle_attention.source_digest().as_hex() +
    "\n",
  )
  output.write_string("qualification_only=1\n")
  output.write_string("manifest_bindable=0\n")
  output.write_string("promotion_authority=0\n")
  @utf8.encode(output.to_string())
}

///|
async fn export_fused_production_probe_inputs(args : Array[String]) -> Unit {
  require_fused_probe(
    args.length() == 4,
    "usage: fused_parallel_cuda_probe export-production TOOLCHAIN_SHA256 ABSOLUTE_NEW_OUTPUT",
  )
  require_fused_probe(
    args[2].length() == 64 && args[3].has_prefix("/"),
    "production export requires a SHA-256 toolchain and absolute output",
  )
  let (context, production) = fused_production_probe_candidates(args[2])
  let staging = args[3] + ".partial"
  @fs.mkdir(staging, permission=0o700)
  try {
    @fs.mkdir(staging + "/sources", permission=0o700)
    @fs.mkdir(staging + "/recipes", permission=0o700)
    production_export_write_new(
      staging + "/sources/production_qkv_v2.cu",
      production.qkv.source_bytes(),
    )
    production_export_write_new(
      staging + "/recipes/production_qkv_v2.recipe",
      production.qkv.recipe_bytes(),
    )
    production_export_write_new(
      staging + "/sources/production_readonly_attention_v2.cu",
      production.readonly_candidate.source_bytes(),
    )
    production_export_write_new(
      staging + "/recipes/production_readonly_attention_v2.recipe",
      production.readonly_candidate.recipe_bytes(),
    )
    production_export_write_new(
      staging + "/sources/production_residual_rmsnorm_v2.cu",
      production.residual.source_bytes(),
    )
    production_export_write_new(
      staging + "/recipes/production_residual_rmsnorm_v2.recipe",
      production.residual.recipe_bytes(),
    )
    production_export_write_new(
      staging + "/sources/oracle_qkv.cu",
      context.qualification.qkv_fallback().source_bytes(),
    )
    production_export_write_new(
      staging + "/recipes/oracle_qkv.recipe",
      context.qualification.qkv_fallback().recipe_bytes(),
    )
    production_export_write_new(
      staging + "/sources/oracle_rope.cu",
      context.qualification.rope_fallback().source_bytes(),
    )
    production_export_write_new(
      staging + "/recipes/oracle_rope.recipe",
      context.qualification.rope_fallback().recipe_bytes(),
    )
    production_export_write_new(
      staging + "/sources/oracle_attention.cu",
      production.oracle_attention.source_bytes(),
    )
    production_export_write_new(
      staging + "/recipes/oracle_attention.recipe",
      production.oracle_attention.recipe_bytes(),
    )
    production_export_write_new(
      staging + "/IDENTITIES.v1",
      production_export_identities(context, production),
    )
    @fs.rename(staging, args[3], replace=false)
  } catch {
    error => {
      production_export_cleanup(staging)
      raise error
    }
  }
  println(
    "outcome=fused-production-v2-inputs-exported authority=qualification-only manifest_bindable=0 promotion_authority=0",
  )
}
