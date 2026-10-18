///|
/// Explicit offline diagnostic, invoked only after the rank commits. Copies
/// literal numerical operands; no digest, reference, or comparison enters the
/// compiler or runtime. At most one retired frame is resident in host memory.
fn capture_retired_frontiers(
  startup : Startup,
  stage : @glm.GlmDecoderStage,
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
  let expert_layer = @env.get_env_var("LUNA_GLM_RETIRED_EXPERT_LAYER").map(value => {
    @string.parse_int(value)
  })
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
    if expert_layer is Some(_) {
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
  buffers : ArrayView[(String, Bytes)],
) -> Unit {
  let directory = root + "/prefill-\{index}"
  @fs.mkdir(directory)
  for (name, bytes) in buffers {
    @fs.write_file(directory + "/" + name, bytes, create_mode=CreateNew)
  }
  println("retired_frontiers prefill=\{index} buffers=\{buffers.length()}")
}
