///|
async fn main {
  if @env.args()
    is [
      _,
      "replay-layer"
      | "replay-layer-tiled"
      | "replay-layer-complex"
      | "replay-layer-striped"
      | "replay-layer-refined"
      | "replay-layer-nonlinear"
      | "replay-layer-vector-rms"
      | "replay-layer-hierarchical-controls"
      | "replay-layer-butterfly-controls"
      | "replay-layer-vector-query"
      | "replay-layer-router-arithmetic"
      | "replay-layer-shared-block-matrix"
      | "replay-layer-fp8-projections"
      | "replay-layer-query-block-matrix" as mode,
      .. arguments,
    ] &&
    (arguments.length() == 9 || arguments.length() == 10) {
    let output_matrix = mode == "replay-layer-fp8-projections"
    let query_matrix = output_matrix ||
      mode == "replay-layer-query-block-matrix"
    let shared_matrix = query_matrix ||
      mode == "replay-layer-shared-block-matrix"
    let router = shared_matrix || mode == "replay-layer-router-arithmetic"
    let grouped = router ||
      mode == "replay-layer-hierarchical-controls" ||
      mode == "replay-layer-butterfly-controls" ||
      mode == "replay-layer-vector-query"
    let vector_rms = grouped || mode == "replay-layer-vector-rms"
    let butterfly = router ||
      mode == "replay-layer-butterfly-controls" ||
      mode == "replay-layer-vector-query"
    replay_matched_layer(
      arguments,
      output_scaled_dot_law=if output_matrix {
        UnscaledBlockMatrixFloat32(
          block_columns=128,
          publication=SeparateMultiplyAdd,
        )
      } else {
        DecodeOperandsFloat32
      },
      projection_scaled_dot_law=if query_matrix {
        UnscaledBlockMatrixFloat32(
          block_columns=128,
          publication=SeparateMultiplyAdd,
        )
      } else {
        DecodeOperandsFloat32
      },
      routing_projection_law=if router {
        GroupedVectorFloat32(
          groups=1,
          partitions=32,
          vector_width=1,
          publication_partitions=1,
          multiply_add=FusedMultiplyAdd,
        )
      } else {
        OrderedFloat32
      },
      routing_softplus_law=if router {
        ThresholdedFloat32(threshold=20.0)
      } else {
        StableFloat32
      },
      routing_sum_law=if router { ButterflyFloat32 } else { OrderedFloat32 },
      shared_scaled_dot_law=if shared_matrix {
        UnscaledBlockMatrixFloat32(
          block_columns=128,
          publication=SeparateMultiplyAdd,
        )
      } else {
        DecodeOperandsFloat32
      },
      attention_accumulation=if mode != "replay-layer" {
        TiledFloat32
      } else {
        OrderedFloat32
      },
      rotary_product=if mode == "replay-layer-complex" ||
        mode == "replay-layer-striped" ||
        mode == "replay-layer-refined" ||
        mode == "replay-layer-nonlinear" ||
        vector_rms {
        ComplexFloat32Fma
      } else {
        SeparateFloat32Products
      },
      output_accumulation=if mode == "replay-layer-striped" ||
        mode == "replay-layer-refined" ||
        mode == "replay-layer-nonlinear" ||
        vector_rms {
        StripedTreeFloat32(partitions=32)
      } else {
        OrderedFloat32
      },
      control_square_law=if vector_rms {
        VectorizedTreeFloat32(partitions=512, vector_width=4)
      } else if mode == "replay-layer-refined" ||
        mode == "replay-layer-nonlinear" {
        CompensatedFloat32
      } else {
        OrderedFloat32
      },
      control_dot_law=matched_control_dot_law(mode),
      query_square_law=if router || mode == "replay-layer-vector-query" {
        VectorizedTreeFloat32(partitions=256, vector_width=4)
      } else {
        VectorizedTreeFloat32(partitions=256, vector_width=1)
      },
      control_vector_affine_law=if mode == "replay-layer-nonlinear" ||
        vector_rms {
        FusedMultiplyAdd
      } else {
        SeparateMultiplyAdd
      },
      control_sigmoid_law=if mode == "replay-layer-nonlinear" || vector_rms {
        DirectReciprocal
      } else {
        StableBranches
      },
      control_matrix_affine_law=if butterfly {
        FusedMultiplyAdd
      } else {
        SeparateMultiplyAdd
      },
      matrix_sum_law=if butterfly { ButterflyFloat32 } else { OrderedFloat32 },
      control_root_law=if vector_rms {
        ReciprocalSqrtFloat32
      } else {
        DivideRoundedSqrtFloat32
      },
      normalization_square_law=if vector_rms {
        VectorizedTreeFloat32(partitions=512, vector_width=4)
      } else {
        VectorizedTreeFloat32(partitions=256, vector_width=1)
      },
    )
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
  let (tail, cancellation) = @prompt_file.split_request_cancellation_options(
    tail,
  )
  guard cancellation is None || mode.has_prefix("generate") else {
    fail("request cancellation is only a generation control")
  }
  let mode_valid = match (mode, tail.length()) {
    ("export", 1)
    | ("export-dspark", 1)
    | ("egress", 4)
    | ("egress-dspark", 4)
    | ("egress-logits-dspark", 5)
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
    "egress" | "egress-dspark" | "egress-logits-dspark" =>
      run_egress(startup, root, tail)
    "ingress" => run_ingress(startup, root, tail, generate=false)
    "generate" | "generate-dspark" =>
      run_ingress(startup, root, tail, generate=true, cancellation~)
    "generate-reference-dspark" =>
      run_ingress(
        startup,
        root,
        tail,
        generate=true,
        speculate=false,
        cancellation~,
      )
    "generate-sequence-reference-dspark" =>
      run_ingress(
        startup,
        root,
        tail,
        generate=true,
        speculate=false,
        sequence=true,
        cancellation~,
      )
    "generate-sequence-dspark" =>
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
    "deepseek-base-checkpoint-placement-v1\nrows=\{startup.rows}\nhistory=\{startup.history}\nreserve=\{startup.reserve}\n"
  for ordinal, range in startup.placement.stages() {
    let (first, end) = range.layer_range()
    builder <+
      "stage=\{ordinal} first=\{first} end=\{end} planned=\{startup.placement.stage_bytes()[ordinal]} ceiling=\{startup.budgets[ordinal]}\n"
  }
  builder.to_string()
}
