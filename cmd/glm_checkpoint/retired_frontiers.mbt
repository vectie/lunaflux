///|
/// Explicit offline diagnostic, invoked only after the rank commits. Copies
/// literal numerical operands; no digest, reference, or comparison enters the
/// compiler or runtime. At most one retired frame is resident in host memory.
fn capture_retired_frontiers(
  startup : Startup,
  stage : @glm.GlmDecoderStage,
  selected_layer? : Int,
  include_experts? : Bool = false,
) -> Array[(String, Bytes)] raise {
  let (first, end) = stage.layer_range()
  let (_, example, _, _) = stage.borrow_retired_ffn(first)
  let upper_bound = example.byte_count() /
    startup.rows.to_int64() *
    startup.rows.to_int64() *
    (end - first).to_int64() *
    3L
  guard upper_bound <= startup.reserve / 4L else {
    fail("retired frontier capture exceeds host reserve")
  }
  let output = []
  let expert_layer = selected_layer
  for layer in first..<end {
    if expert_layer is Some(selected) && selected != layer {
      continue
    }
    let (rows, residual, input, branch) = stage.borrow_retired_ffn(layer)
    for
      (name, buffer) in [
        ("residual", residual),
        ("input", input),
        ("branch", branch),
      ] {
      let bytes = buffer.byte_count() /
        startup.rows.to_int64() *
        rows.to_int64()
      output.push(
        ("layer-\{layer}-\{name}.bf16", buffer.copy_to_host(0L, bytes)),
      )
    }
    if include_experts {
      let buffers = stage.borrow_retired_experts(layer)
      let bytes = for (_, buffer) in buffers; total = 0L {
        continue total + buffer.byte_count()
      } nobreak {
        total
      }
      guard bytes + upper_bound <= startup.reserve / 4L else {
        fail("retired expert capture exceeds host reserve")
      }
      for (name, buffer) in buffers {
        output.push(
          (
            "layer-\{layer}-\{name}.bin",
            buffer.copy_to_host(0L, buffer.byte_count()),
          ),
        )
      }
    }
  }
  output
}

///|
async fn write_retired_frontiers(
  root : String,
  index : Int,
  startup : Startup,
  stage : @glm.GlmDecoderStage,
) -> Unit {
  let directory = root + "/prefill-\{index}"
  @fs.mkdir(directory)
  let selected = @env.get_env_var("LUNA_GLM_RETIRED_EXPERT_LAYER")
  let (first, end) = stage.layer_range()
  let layers = match selected {
    Some("all") => Array::makei(end - first, i => first + i)
    Some(text) => {
      let layer = @string.parse_int(text)
      guard layer >= first && layer < end else { fail("invalid retired layer") }
      [layer]
    }
    None => []
  }
  // The control owner withholds its commit reply until this offline effect
  // completes. Only one layer is copied at a time; no whole-model leaf arena.
  let mut written = 0
  if selected is None {
    let buffers = capture_retired_frontiers(startup, stage)
    for (name, bytes) in buffers {
      @fs.write_file(directory + "/" + name, bytes, create_mode=CreateNew)
      written += 1
    }
  } else {
    for layer in layers {
      let buffers = capture_retired_frontiers(
        startup,
        stage,
        selected_layer=layer,
        include_experts=true,
      )
      for (name, bytes) in buffers {
        @fs.write_file(directory + "/" + name, bytes, create_mode=CreateNew)
        written += 1
      }
    }
  }
  println("retired_frontiers prefill=\{index} buffers=\{written}")
}
