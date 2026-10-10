///|
fn replay_layer_index(rows : Int, layer : Int) -> Int raise {
  guard rows > 0 && rows <= 1024 && layer > 0 && layer <= 50 else {
    fail("layer replay needs 1..1024 rows and next hidden-state index 1..50")
  }
  layer - 1
}

///|
test "one-layer replay uses the previous hidden-state frontier" {
  for layer in [1, 7, 50] {
    assert_eq(replay_layer_index(140, layer), layer - 1)
  }
  for (rows, layer) in [(0, 7), (1025, 7), (140, 0), (140, 51)] {
    try replay_layer_index(rows, layer) catch {
      _ => ()
    } noraise {
      _ => fail("invalid replay frontier accepted")
    }
  }
}

///|
/// Independent complete-layer arithmetic on an identical native input. This
/// diagnostic is neither a propagated reference prefix nor fifty-layer output.
fn replay_layer(
  session : @native.Session,
  root : String,
  rows : Int,
  layer : Int,
  input : String,
  output : String,
) -> Unit raise {
  let index = replay_layer_index(rows, layer)
  let model = Model::new(session, root)
  let initial = session.load(input, 0L, rows, model.hidden)
  defer initial.close()
  let cosine = session.run(Cosine, a=rows, b=model.dim, d=model.theta)
  defer cosine.close()
  let sine = session.run(Sine, a=rows, b=model.dim, d=model.theta)
  defer sine.close()
  initial.write(output + "/input-hidden.bf16")
  let result = encode_layer(
    model,
    session,
    initial,
    index,
    cosine,
    sine,
    UpstreamBf16,
    observe=(name, tensor) => tensor.write(output + "/" + name + ".bf16"),
  )
  defer result.close()
  result.write(output + "/hidden-\{layer}.bf16")
  result.close()
  sine.close()
  cosine.close()
  initial.close()
  guard session.retained_bytes() == 0L else {
    fail("layer replay owner leaked")
  }
  println(
    "reference_layer_replay_complete input_hidden_state=\{index} output_hidden_state=\{layer} rows=\{rows} retained_bytes=0 scope=one_complete_layer_only",
  )
}
