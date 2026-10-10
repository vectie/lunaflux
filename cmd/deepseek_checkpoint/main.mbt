///|
async fn main {
  if @env.args()
    is [
      _,
      mode,
      model_root,
      rows,
      history,
      maximum,
      stop,
      frames,
      count,
      byte_budget,
    ] &&
    (mode == "preflight-generation" || mode == "preflight-generation-sequence") {
    preflight_generation(
      model_root,
      rows,
      history,
      maximum,
      stop,
      frames,
      count,
      byte_budget,
      sequence=mode == "preflight-generation-sequence",
    )
    return
  }
  if @env.args() is [_, "decode-tokens", model_root, label, ids, output] {
    let root = @approved_fs.ApprovedRoot::open_absolute(model_root)
    defer root.close()
    decode_text_tokens(root, model_root, label, ids, output)
    return
  }
  if @env.args()
    is [
      _,
      "text-frames",
      model_root,
      rows,
      history,
      tokenizer_label,
      input,
      output,
    ] {
    let root = @approved_fs.ApprovedRoot::open_absolute(model_root)
    defer root.close()
    prepare_text_frames(
      root,
      model_root,
      @string.parse_int(rows),
      @string.parse_int(history),
      tokenizer_label,
      input,
      output,
    )
    return
  }
  if @env.args()
    is [_, "token-frames", model_root, rows, history, token_ids, output] {
    let root = @approved_fs.ApprovedRoot::open_absolute(model_root)
    defer root.close()
    let spec = inspect_config(root)
    let rows = @string.parse_int(rows)
    let history = @string.parse_int(history)
    guard rows > 0 &&
      rows <= history &&
      history <= spec.max_position_embeddings() &&
      token_ids.length() <= 1048576 else {
      fail("invalid token-frame envelope")
    }
    let tokens : Array[Int] = []
    for id in token_ids.split(",") {
      guard tokens.length() < history - 1 else {
        fail("prompt exhausts reserved context")
      }
      tokens.push(@string.parse_int(id))
    }
    let (limits, inference) = prompt_limits(
      rows,
      history,
      spec.vocabulary_size(),
    )
    let frames = @prompt.encode_prompt(
      tokens,
      limits,
      inference,
      maximum_rows=rows,
      maximum_context=history,
      model_generation=1UL,
    )
    @fs.mkdir(output)
    for index, bytes in frames {
      @fs.write_file(
        output + "/plan-\{index}.bin",
        bytes,
        create_mode=CreateNew,
      )
    }
    println("prompt_frames=\{frames.length()} input_tokens=\{tokens.length()}")
    return
  }
  if @env.args() is [_, "config", model_root] {
    let root = @approved_fs.ApprovedRoot::open_absolute(model_root)
    defer root.close()
    let spec = inspect_config(root)
    println(
      "DeepSeek-V4 base config: layers=\{spec.layer_count()} hidden=\{spec.hidden_size()} vocabulary=\{spec.vocabulary_size()}",
    )
    return
  }
  guard @env.args()
    is [
      _,
      mode,
      model_root,
      inventory_name,
      rows,
      history,
      budget0,
      budget1,
      reserve,
      .. tail,
    ] else {
    fail(
      "MODE MODEL_ROOT SHARD_SHA256 ROWS HISTORY BUDGET0 BUDGET1 RESERVE [MODE_ARGS]",
    )
  }
  // Reject mode/arguments before opening or hashing checkpoint payloads.
  let mode_valid = match (mode, tail.length()) {
    ("export", 1)
    | ("export-dspark", 1)
    | ("egress", 4)
    | ("egress-dspark", 4)
    | ("ingress", 7)
    | ("generate", 9)
    | ("generate-dspark", 9)
    | ("generate-sequence-reference-dspark", 9)
    | ("generate-sequence-dspark", 9)
    | ("generate-reference-dspark", 9) => true
    _ => false
  }
  guard mode_valid else {
    fail(
      "export OUTDIR | egress AOT_ROOT AOT_FILE CONTROL_LISTEN ACTIVATION_LISTEN | ingress AOT_ROOT AOT_FILE CONTROL_PEER ACTIVATION_PEER FRAME_ROOT FRAME_COUNT OUTDIR | generate AOT_ROOT AOT_FILE CONTROL_PEER ACTIVATION_PEER FRAME_ROOT PREFILL_COUNT MAX_NEW_TOKENS STOP_TOKEN_OR_MINUS_ONE OUTDIR",
    )
  }
  let root = @approved_fs.ApprovedRoot::open_absolute(model_root)
  defer root.close()
  let startup = {
    errdefer println("checkpoint_startup_failed; see terminal error")
    inspect_startup(
      root,
      inventory_name,
      @string.parse_int(rows),
      @string.parse_int(history),
      [@string.parse_int64(budget0), @string.parse_int64(budget1)],
      @string.parse_int64(reserve),
      dspark=mode.has_suffix("-dspark"),
    )
  }
  for ordinal, range in startup.placement.stages() {
    let (first, end) = range.layer_range()
    println(
      "stage=\{ordinal} layers=[\{first},\{end}) planned_bytes=\{startup.placement.stage_bytes()[ordinal]} ceiling=\{startup.budgets[ordinal]}",
    )
  }
  match mode {
    "export" | "export-dspark" => {
      @fs.mkdir(tail[0])
      for ordinal in 0..<2 {
        let stage = stage_plan(startup, ordinal)
        defer stage.close()
        let source = if startup.dspark && ordinal == 1 {
          let prediction = prediction_plan(
            startup.spec,
            startup.rows,
            startup.history,
            startup.budgets[ordinal] - startup.reserve,
          )
          defer prediction.close()
          stage.source_bytes() + b"\n" + prediction.source_bytes()
        } else {
          stage.source_bytes()
        }
        @fs.write_file(
          tail[0] + "/stage-\{ordinal}.cu",
          source,
          create_mode=CreateNew,
        )
      }
      @fs.write_file(
        tail[0] + "/placement.txt",
        placement_receipt(startup),
        create_mode=CreateNew,
      )
    }
    "egress" | "egress-dspark" => run_egress(startup, root, tail)
    "ingress" => run_ingress(startup, root, tail, generate=false)
    "generate" | "generate-dspark" =>
      run_ingress(startup, root, tail, generate=true)
    "generate-reference-dspark" =>
      run_ingress(startup, root, tail, generate=true, speculate=false)
    "generate-sequence-reference-dspark" =>
      run_ingress(
        startup,
        root,
        tail,
        generate=true,
        speculate=false,
        sequence=true,
      )
    "generate-sequence-dspark" =>
      run_ingress(startup, root, tail, generate=true, sequence=true)
    _ => fail("invalid mode")
  }
}

///|
fn placement_receipt(startup : Startup) -> String {
  let builder = StringBuilder()
  builder <+
    "deepseek-base-checkpoint-placement-v1\nrows=\{startup.rows}\nhistory=\{startup.history}\nreserve=\{startup.reserve}\n"
  for ordinal, range in startup.placement.stages() {
    let (first, end) = range.layer_range()
    builder <+
      "stage=\{ordinal} first=\{first} end=\{end} planned=\{startup.placement.stage_bytes()[ordinal]} ceiling=\{startup.budgets[ordinal]}\n"
  }
  builder.to_string()
}
