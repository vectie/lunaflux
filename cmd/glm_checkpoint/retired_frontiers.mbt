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
  for layer in first..<end {
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
