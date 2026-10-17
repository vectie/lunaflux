///|
async fn main {
  if @env.args() is [_, "probe-tensor-context", rank, endpoint, policy] {
    guard policy is ("private" | "runtime") else { fail("private|runtime") }
    probe_tensor_context(@string.parse_int(rank), endpoint, policy == "runtime")
    return
  }
  if @env.args() is [_, "decode-tokens", model_root, label, ids, output] {
    decode_text_tokens(model_root, label, ids, output)
    return
  }
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
    mode is ("preflight-generation" | "preflight-generation-sequence") {
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
    prepare_text_frames(
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
      history <= spec.context_length() &&
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
      "GLM-5.3 Flash config: layers=\{spec.layer_count()} hidden=\{spec.hidden_size()} vocabulary=\{spec.vocabulary_size()}",
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
  // Explicit experimental arithmetic, shared by export and runtime planning.
  // Ordinary commands retain ordered recurrence; no request-path selection.
  let buffered = mode.has_suffix("-buffered-matrices")
  let mode = if buffered { mode[:mode.length() - 18].to_owned() } else { mode }
  let lookahead = mode.has_suffix("-lookahead-experts")
  let mode = if lookahead { mode[:mode.length() - 18].to_owned() } else { mode }
  let expert_transport : @matrix.ExpertTransport = if mode.has_suffix(
      "-expert-pipeline",
    ) {
    PackedPipeline
  } else if mode.has_suffix("-buffered-experts") {
    IndexedBuffered
  } else if mode.has_suffix("-packed-experts") {
    PackedRows
  } else {
    IndexedRows
  }
  let mode = match expert_transport {
    IndexedRows => mode
    IndexedBuffered => mode[:mode.length() - 17].to_owned()
    PackedRows => mode[:mode.length() - 15].to_owned()
    PackedPipeline => mode[:mode.length() - 16].to_owned()
  }
  let shape_aware = mode.has_suffix("-shape-aware")
  guard !buffered || shape_aware else {
    fail("buffered matrices require shape-aware arithmetic")
  }
  let mode = if shape_aware {
    mode[:mode.length() - 12].to_owned()
  } else {
    mode
  }
  let replay = mode.has_suffix("-graph-tp")
  let mode = if replay { mode[:mode.length() - 9].to_owned() } else { mode }
  let queued = mode.has_suffix("-queued-tp")
  guard !replay || queued else {
    fail("graph replay requires queued TP submission")
  }
  let mode = if queued { mode[:mode.length() - 10].to_owned() } else { mode }
  let low_bit = mode.has_suffix("-low-bit")
  guard !lookahead || low_bit else {
    fail("fragment lookahead requires native low-bit experts")
  }
  guard expert_transport == IndexedRows || low_bit else {
    fail("packed expert transport requires the explicit low-bit path")
  }
  let mode = if low_bit { mode[:mode.length() - 8].to_owned() } else { mode }
  let expert_tp = mode.has_suffix("-expert-tp")
  guard !queued || expert_tp else {
    fail("queued submission requires tensor parallelism")
  }
  let mode = if expert_tp { mode[:mode.length() - 10].to_owned() } else { mode }
  let tiled = mode.has_suffix("-tiled-grouped")
  let mode = if tiled { mode[:mode.length() - 14].to_owned() } else { mode }
  let chunk = mode.has_suffix("-chunk-kda")
  let mode = if chunk { mode[:mode.length() - 10].to_owned() } else { mode }
  // Reject mode/arguments before opening checkpoint payloads.
  let (tail, cancellation) = @prompt_file.split_request_cancellation_options(
    tail,
  )
  guard cancellation is None || mode.has_prefix("generate") else {
    fail("request cancellation is only a generation control")
  }
  let mode_valid = match (mode, tail.length()) {
    ("export", 1)
    | ("egress", 4)
    | ("egress-logits", 5)
    | ("ingress", 7)
    | ("generate", 9)
    | ("generate-sequence", 9) => true
    _ => false
  }
  guard mode_valid else {
    fail(
      "export OUTDIR | egress AOT_ROOT AOT_FILE CONTROL_LISTEN ACTIVATION_LISTEN | egress-logits AOT_ROOT AOT_FILE CONTROL_LISTEN ACTIVATION_LISTEN CAPTURE_DIR | ingress AOT_ROOT AOT_FILE CONTROL_PEER ACTIVATION_PEER FRAME_ROOT FRAME_COUNT OUTDIR | generate|generate-sequence AOT_ROOT AOT_FILE CONTROL_PEER ACTIVATION_PEER FRAME_ROOT PREFILL_COUNT_OR_REQUEST_COUNT MAX_NEW_TOKENS STOP_TOKEN_OR_MINUS_ONE OUTDIR",
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
      recurrent_evaluation=if chunk {
        ChunkTriangularFloat32
      } else {
        OrderedTokenFloat32
      },
      matrix_law=if shape_aware {
        ShapeAwareFloat32
      } else if tiled {
        MatrixFloat32
      } else {
        ScalarFloat32
      },
      expert_scaled_dot_law=if low_bit {
        BlockScaledMatrixFloat32(block_columns=16)
      } else {
        DecodeOperandsFloat32
      },
      expert_tensor_parallel=expert_tp,
      matrix_staging=if buffered {
        Buffered(inner_tile=32)
      } else {
        Synchronous
      },
      expert_transport~,
      expert_fragment_slots=if lookahead { 2 } else { 1 },
      retirement=if queued { WholeFrame } else { EachCollective },
      replay=if replay { CaptureIntervals } else { Eager },
    )
  }
  println(
    "expert_tensor_parallel=\{expert_tp} nonexpert_tensor_parallel=\{expert_tp} attention_replicated=false payload_hashing=false",
  )
  println(
    "expert_activation_quantization=\{low_bit} native_block_scaled_matrix=\{low_bit}",
  )
  println("tp_submission_queued=\{queued} tp_graph_replay=\{replay}")
  println(
    "recurrent_matrix_buffered=\{buffered} operand_slots=\{if buffered {2} else {1}}",
  )
  let transport_name = match expert_transport {
    IndexedRows => "indexed"
    IndexedBuffered => "buffered"
    PackedRows => "packed"
    PackedPipeline => "pipeline"
  }
  println("expert_fragment_slots=\{if lookahead { 2 } else { 1 }}")
  println("expert_operand_transport=\{transport_name}")
  for ordinal, range in startup.ranges {
    let (first, end) = range.layer_range()
    println(
      "rank=\{ordinal} layers=[\{first},\{end}) planned_bytes=\{startup.planned_bytes[ordinal]} ceiling=\{startup.budgets[ordinal]}",
    )
  }
  println(
    "recurrent_evaluation=" +
    (if chunk { "chunk-triangular-f32" } else { "ordered-token-f32" }),
  )
  println(
    "matrix_evaluation=" +
    (if shape_aware {
      "shape-aware-f32"
    } else if tiled {
      "bf16-matrix-f32-grouped"
    } else {
      "scalar-f32"
    }),
  )
  match mode {
    "export" => {
      @fs.mkdir(tail[0])
      for ordinal in 0..<2 {
        let stage = stage_plan(startup, ordinal)
        defer stage.close()
        @fs.write_file(
          tail[0] + "/stage-\{ordinal}.cu",
          stage.source_bytes(),
          create_mode=CreateNew,
        )
      }
      @fs.write_file(
        tail[0] + "/placement.txt",
        placement_receipt(startup),
        create_mode=CreateNew,
      )
    }
    "egress" | "egress-logits" => run_egress(startup, root, tail)
    "ingress" => run_ingress(startup, root, tail, generate=false)
    "generate" => run_ingress(startup, root, tail, generate=true, cancellation~)
    "generate-sequence" =>
      run_ingress(
        startup,
        root,
        tail,
        generate=true,
        sequence=true,
        cancellation~,
      )
    _ => fail("invalid mode")
  }
}

///|
fn placement_receipt(startup : Startup) -> String {
  let builder = StringBuilder()
  builder <+
    "glm-checkpoint-placement-v1\nrows=\{startup.rows}\nhistory=\{startup.history}\nreserve=\{startup.reserve}\n"
  builder <+ "expert_tensor_parallel=\{startup.expert_tensor_parallel}\n"
  for ordinal, range in startup.ranges {
    let (first, end) = range.layer_range()
    builder <+
      "rank=\{ordinal} first=\{first} end=\{end} planned=\{startup.planned_bytes[ordinal]} ceiling=\{startup.budgets[ordinal]}\n"
  }
  builder.to_string()
}
