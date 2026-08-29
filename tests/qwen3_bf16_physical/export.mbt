///|
async fn qwen3_export(
  args : ArrayView[String],
) -> Unit raise Qwen3PhysicalError {
  qwen3_require(args.length() == 13, "export arguments are invalid")
  let model_root = @fs.realpath(args[3]) catch {
    _ => raise InvalidInput("model root is unavailable")
  }
  qwen3_require(model_root == args[3], "model root is not canonical")
  let metadata = qwen3_load_metadata(model_root, args[4], args[5])
  let corpus_cases = qwen3_verify_reference_corpus(
    model_root,
    args[6],
    args[7],
    args[8],
  )
  let inputs = qwen3_candidate_inputs(
    metadata,
    args[9],
    qwen3_parse_uint(args[10]),
    qwen3_parse_uint(args[11]),
    qwen3_parse_uint(args[12]),
  )
  let output = args[2]
  let evidence = @luna_bf16_candidate_export.export_new(
    inputs.candidates,
    output,
  ) catch {
    _ => raise CandidateMismatch("candidate publication failed")
  }
  println("schema=lunaflux-qwen3-bf16-candidate-export.v1")
  println("candidate_root=" + output + "/candidate-root")
  println(
    "candidate_inventory=" +
    output +
    "/candidate.files.sha256#sha256=" +
    evidence.inventory_sha256(),
  )
  println("candidate_set_sha256=" + evidence.candidate_set_sha256())
  println("model_plan_sha256=" + evidence.model_plan_sha256())
  println("operation_count=" + evidence.operation_count().to_string())
  println("layer_count=" + metadata.spec().layer_count().to_string())
  println("reference_cases=" + corpus_cases.to_string())
  println("qkv_independent_widths=passed")
  println("qk_rms_norm_capability=11")
  println("compiler_invoked=0")
  println("device_opened=0")
  println("promotion_authority=0")
}

///|
test "Qwen physical campaign rejects malformed compiler identity" {
  try qwen3_parse_uint("13.1") catch {
    InvalidInput(_) => ()
    _ => fail("unexpected compiler identity error")
  } noraise {
    _ => fail("malformed compiler identity was accepted")
  }
}
