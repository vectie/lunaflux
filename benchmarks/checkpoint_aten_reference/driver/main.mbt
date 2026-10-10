///|
fn self_test(session : @native.Session, output : String) -> Unit raise {
  let x = session.run(Constant, a=2, b=4, d=2.0)
  defer x.close()
  let w = session.run(Constant, a=1, b=4, d=1.0)
  defer w.close()
  let normalized = session.run(RmsNorm, x~, y=w, d=0.000001)
  defer normalized.close()
  normalized.write(output + "/normalized.bf16")
  let bytes = session.bytes(output + "/normalized.bf16")
  guard bytes.length() == 16 else { fail("wrong BF16 output extent") }
  for i in 0..<8 {
    guard bytes[i * 2] == 0x80 && bytes[i * 2 + 1] == 0x3f else {
      fail("wrong independent RMSNorm result")
    }
  }
  try session.load(output + "/normalized.bf16", 0L, 4, 4) catch {
    _ => ()
  } noraise {
    t => {
      t.close()
      fail("short weight read must fail")
    }
  }
  try session.run(Constant, a=65536, b=65536) catch {
    _ => ()
  } noraise {
    t => {
      t.close()
      fail("memory budget must reject before allocation")
    }
  }
  guard session.bytes(output + "/normalized.bf16") == bytes else {
    fail("metadata read must recover after a failed native operation")
  }
  let recovered = session.run(Add, x~, y=x)
  defer recovered.close()
  for
    (name, op) in [
      ("f32", @native.Op::LinearFloat32),
      ("f64", @native.Op::LinearFloat64),
      ("bf16-full", @native.Op::LinearBf16FullReduction),
    ] {
    let projected = session.run(op, x~, y=w)
    defer projected.close()
    let path = output + "/linear-" + name + ".bf16"
    projected.write(path)
    guard session.bytes(path) == b"\x00\x41\x00\x41" else {
      fail("precision oracle projection arithmetic")
    }
  }
  normalized.close()
  recovered.close()
  w.close()
  x.close()
  x.close()
  guard session.retained_bytes() == 0L else {
    fail("failure or explicit cleanup leaked tensors")
  }
  try session.run(Add, x~, y=x) catch {
    _ => ()
  } noraise {
    t => {
      t.close()
      fail("closed operand must fail")
    }
  }
  println(
    "reference_self_test_passed arithmetic=bf16_rmsnorm short_read=clean budget=preallocation double_close=clean closed_operand=rejected retained_bytes=0",
  )
}

///|
fn main raise {
  match @env.args() {
    [_, "replay-gate-input", library, model, layer, row, channel, first, second] => {
      let session = @native.Session::new(library, device=0, budget=2097152L)
      defer session.close()
      replay_gate_input(
        session,
        model,
        @string.parse_int(layer),
        @string.parse_int(row),
        @string.parse_int(channel),
        first,
        second,
      )
    }
    [_, "activation-factorization", library, rows, width, first, second, output] => {
      let rows = @string.parse_int(rows)
      let width = @string.parse_int(width)
      ignore(activation_geometry(rows, width))
      let session = @native.Session::new(library, device=1, budget=2147483648L)
      defer session.close()
      activation_factorization(session, first, second, rows, width, output)
    }
    [_, "layer-projection-oracle", library, model, rows, layer, input, output] => {
      let rows = @string.parse_int(rows)
      let layer = @string.parse_int(layer)
      ignore(replay_layer_index(rows, layer))
      let session = @native.Session::new(library, device=1, budget=2147483648L)
      defer session.close()
      projection_oracle(session, model, input, output, rows~, layer~)
    }
    [_, "replay-layer", library, model, rows, layer, input, output] => {
      let rows = @string.parse_int(rows)
      let layer = @string.parse_int(layer)
      ignore(replay_layer_index(rows, layer))
      let session = @native.Session::new(library, device=1, budget=2147483648L)
      defer session.close()
      replay_layer(session, model, rows, layer, input, output)
    }
    [_, "replay-down-reduction", library, model, input, oracle] => {
      let session = @native.Session::new(library, device=0, budget=2097152L)
      defer session.close()
      replay_down_reduction(session, model, input, oracle)
    }
    [_, "projection-oracle", library, model, input, output] => {
      let session = @native.Session::new(library, device=1, budget=2147483648L)
      defer session.close()
      projection_oracle(session, model, input, output)
    }
    [_, mode, library, model, text_ids, output, input_root] if mode ==
      "replay-operators" ||
      mode == "replay-full-reduction" => {
      let ids : Array[Int] = []
      for id in text_ids.split(",") {
        ids.push(@string.parse_int(id))
      }
      let session = @native.Session::new(library, device=1, budget=2147483648L)
      defer session.close()
      encode(
        session,
        model,
        ids,
        output,
        1,
        operators=true,
        replay_root=input_root,
        arithmetic=if mode == "replay-full-reduction" {
          FullReductionBf16
        } else {
          UpstreamBf16
        },
      )
    }
    [_, "self-test", library, output] => {
      let session = @native.Session::new(library, device=0, budget=2097152L)
      defer session.close()
      self_test(session, output)
    }
    [_, mode, library, model, text_ids, output, depth] if mode == "encode" ||
      mode == "encode-operators" ||
      mode == "encode-full-reduction" => {
      let ids : Array[Int] = []
      for id in text_ids.split(",") {
        ids.push(@string.parse_int(id))
      }
      let layers = @string.parse_int(depth)
      let session = @native.Session::new(library, device=1, budget=2147483648L)
      defer session.close()
      guard mode != "encode-operators" || layers == 1 else {
        fail("operator diagnostic observes exactly the first layer")
      }
      encode(
        session,
        model,
        ids,
        output,
        layers,
        operators=mode == "encode-operators",
        arithmetic=if mode == "encode-full-reduction" {
          FullReductionBf16
        } else {
          UpstreamBf16
        },
      )
    }
    _ =>
      fail(
        "self-test LIB OUTPUT_DIRECTORY | encode|encode-operators|encode-full-reduction LIB ORIGINAL_ENCODER_ROOT COMMA_IDS OUTPUT_DIRECTORY LAYERS | replay-layer LIB MODEL ROWS NEXT_HIDDEN_INDEX INPUT_BF16 OUTPUT | layer-projection-oracle LIB MODEL ROWS NEXT_HIDDEN_INDEX INPUT_OPERATORS OUTPUT | activation-factorization LIB ROWS WIDTH FIRST_PROJECTIONS SECOND_PROJECTIONS OUTPUT | replay-gate-input LIB MODEL LAYER ROW CHANNEL FIRST_OPERATORS SECOND_OPERATORS | replay-operators|replay-full-reduction LIB MODEL IDS OUTPUT NATIVE_INPUTS | projection-oracle LIB MODEL NATIVE_INPUTS OUTPUT | replay-down-reduction LIB MODEL NATIVE_INPUTS PRECISION_ORACLE",
      )
  }
}
