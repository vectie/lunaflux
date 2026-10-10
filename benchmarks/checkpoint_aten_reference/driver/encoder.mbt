///|
priv enum ProjectionArithmetic {
  UpstreamBf16
  FullReductionBf16
}

///|
fn projection_operation(arithmetic : ProjectionArithmetic) -> @native.Op {
  match arithmetic {
    UpstreamBf16 => Linear
    FullReductionBf16 => LinearBf16FullReduction
  }
}

///|
test "ordinary reference retains upstream BF16 arithmetic" {
  assert_true(projection_operation(UpstreamBf16) is Linear)
  assert_true(
    projection_operation(FullReductionBf16) is LinearBf16FullReduction,
  )
}

///|
fn embedding(
  model : Model,
  session : @native.Session,
  ids : ArrayView[Int],
) -> @native.Tensor raise {
  let (path, offset) = model.region(
    session,
    "model.language_model.embed_tokens.weight",
    model.vocabulary,
    model.hidden,
  )
  for i = 0, state = (None : @native.Tensor?); i < ids.length(); {
    guard ids[i] >= 0 && ids[i] < model.vocabulary else {
      fail("invalid input token")
    }
    let row = session.load(
      path,
      offset + ids[i].to_int64() * model.hidden.to_int64() * 2L,
      1,
      model.hidden,
    )
    let next = match state {
      None => row
      Some(previous) => {
        defer row.close()
        defer previous.close()
        session.run(ConcatRows, x=previous, y=row)
      }
    }
    continue i + 1, Some(next)
  } nobreak {
    state.unwrap()
  }
}

///|
fn project(
  model : Model,
  session : @native.Session,
  x : @native.Tensor,
  prefix : String,
  name : String,
  rows : Int,
  columns : Int,
  arithmetic : ProjectionArithmetic,
) -> @native.Tensor raise {
  let w = model.weight(session, prefix + name + ".weight", rows, columns)
  defer w.close()
  session.run(projection_operation(arithmetic), x~, y=w)
}

///|
fn norm(
  model : Model,
  session : @native.Session,
  x : @native.Tensor,
  prefix : String,
  name : String,
  width : Int,
) -> @native.Tensor raise {
  let w = model.weight(session, prefix + name + ".weight", 1, width)
  defer w.close()
  session.run(RmsNorm, x~, y=w, d=model.epsilon)
}

///|
fn encode_layer(
  model : Model,
  session : @native.Session,
  x : @native.Tensor,
  layer : Int,
  cosine : @native.Tensor,
  sine : @native.Tensor,
  arithmetic : ProjectionArithmetic,
  observe? : (String, @native.Tensor) -> Unit raise,
  replay? : OperatorReplayInputs,
) -> @native.Tensor raise {
  let prefix = "model.language_model.layers.\{layer}."
  let normalized = norm(
    model,
    session,
    replay_operand(replay, "embedding", x),
    prefix,
    "input_layernorm",
    model.hidden,
  )
  defer normalized.close()
  if observe is Some(f) {
    f("norm-input", normalized)
  }
  let q = project(
    model,
    session,
    replay_operand(replay, "norm-input", normalized),
    prefix,
    "self_attn.q_proj",
    model.heads * model.dim,
    model.hidden,
    arithmetic,
  )
  defer q.close()
  if observe is Some(f) {
    f("q", q)
  }
  let k = project(
    model,
    session,
    replay_operand(replay, "norm-input", normalized),
    prefix,
    "self_attn.k_proj",
    model.kv_heads * model.dim,
    model.hidden,
    arithmetic,
  )
  defer k.close()
  if observe is Some(f) {
    f("k", k)
  }
  let v = project(
    model,
    session,
    replay_operand(replay, "norm-input", normalized),
    prefix,
    "self_attn.v_proj",
    model.kv_heads * model.dim,
    model.hidden,
    arithmetic,
  )
  defer v.close()
  if observe is Some(f) {
    f("v", v)
  }
  let qn = norm(
    model,
    session,
    replay_operand(replay, "q", q),
    prefix,
    "self_attn.q_norm",
    model.dim,
  )
  defer qn.close()
  let kn = norm(
    model,
    session,
    replay_operand(replay, "k", k),
    prefix,
    "self_attn.k_norm",
    model.dim,
  )
  defer kn.close()
  if observe is Some(f) {
    f("q-norm", qn)
    f("k-norm", kn)
  }
  let qr = session.run(
    Rotary,
    x=replay_operand(replay, "q-norm", qn),
    y=cosine,
    z=sine,
    a=model.heads,
    b=model.dim,
  )
  defer qr.close()
  let kr = session.run(
    Rotary,
    x=replay_operand(replay, "k-norm", kn),
    y=cosine,
    z=sine,
    a=model.kv_heads,
    b=model.dim,
  )
  defer kr.close()
  if observe is Some(f) {
    f("q-rotary", qr)
    f("k-rotary", kr)
  }
  let attention = session.run(
    Attention,
    x=replay_operand(replay, "q-rotary", qr),
    y=replay_operand(replay, "k-rotary", kr),
    z=replay_operand(replay, "v", v),
    a=model.heads,
    b=model.kv_heads,
    c=model.dim,
  )
  defer attention.close()
  if observe is Some(f) {
    f("attention", attention)
  }
  let output = project(
    model,
    session,
    replay_operand(replay, "attention", attention),
    prefix,
    "self_attn.o_proj",
    model.hidden,
    model.heads * model.dim,
    arithmetic,
  )
  defer output.close()
  if observe is Some(f) {
    f("output", output)
  }
  let residual = session.run(
    Add,
    x=replay_operand(replay, "embedding", x),
    y=replay_operand(replay, "output", output),
  )
  defer residual.close()
  if observe is Some(f) {
    f("residual-attn", residual)
  }
  let mlp_input = norm(
    model,
    session,
    replay_operand(replay, "residual-attn", residual),
    prefix,
    "post_attention_layernorm",
    model.hidden,
  )
  defer mlp_input.close()
  if observe is Some(f) {
    f("norm-mlp", mlp_input)
  }
  let gate = project(
    model,
    session,
    replay_operand(replay, "norm-mlp", mlp_input),
    prefix,
    "mlp.gate_proj",
    model.intermediate,
    model.hidden,
    arithmetic,
  )
  defer gate.close()
  let up = project(
    model,
    session,
    replay_operand(replay, "norm-mlp", mlp_input),
    prefix,
    "mlp.up_proj",
    model.intermediate,
    model.hidden,
    arithmetic,
  )
  defer up.close()
  let activation = session.run(SiluMultiply, x=gate, y=up)
  defer activation.close()
  if observe is Some(f) {
    f("activation", activation)
  }
  let down = project(
    model,
    session,
    replay_operand(replay, "activation", activation),
    prefix,
    "mlp.down_proj",
    model.hidden,
    model.intermediate,
    arithmetic,
  )
  defer down.close()
  if observe is Some(f) {
    f("down", down)
  }
  let result = session.run(
    Add,
    x=replay_operand(replay, "residual-attn", residual),
    y=replay_operand(replay, "down", down),
  )
  errdefer result.close()
  if observe is Some(f) {
    f("residual-mlp", result)
  }
  result
}

///|
fn encode(
  session : @native.Session,
  root : String,
  ids : ArrayView[Int],
  output : String,
  layers : Int,
  operators? : Bool = false,
  replay_root? : String,
  arithmetic? : ProjectionArithmetic = UpstreamBf16,
) -> Unit raise {
  guard ids.length() > 0 && ids.length() <= 1024 && layers > 0 && layers <= 50 else {
    fail("reference request must use 1..1024 tokens and 1..50 selected layers")
  }
  let model = Model::new(session, root)
  guard replay_root is None || (operators && layers == 1) else {
    fail("matched replay requires first-layer operator mode")
  }
  let replay = replay_root.map(path => {
    OperatorReplayInputs::load(session, path, ids.length())
  })
  defer (if replay is Some(owner) { owner.close() })
  let cosine = session.run(Cosine, a=ids.length(), b=model.dim, d=model.theta)
  defer cosine.close()
  let sine = session.run(Sine, a=ids.length(), b=model.dim, d=model.theta)
  defer sine.close()
  let initial = embedding(model, session, ids)
  initial.write(output + "/hidden-0.bf16")
  if operators {
    initial.write(output + "/embedding.bf16")
  }
  for layer = 0, hidden = initial; layer < layers; {
    let next = encode_layer(
      model,
      session,
      hidden,
      layer,
      cosine,
      sine,
      arithmetic,
      observe?=if operators && layer == 0 {
        Some((name, tensor) => tensor.write(output + "/" + name + ".bf16"))
      } else {
        None
      },
      replay?,
    )
    hidden.close()
    next.write(output + "/hidden-\{layer + 1}.bf16")
    println(
      "reference_layer_complete layer=\{layer + 1} retained_bytes=\{session.retained_bytes()}",
    )
    continue layer + 1, next
  } nobreak {
    hidden.close()
  }
  cosine.close()
  sine.close()
  if replay is Some(owner) {
    owner.close()
  }
  guard session.retained_bytes() == 0L else {
    fail("reference tensor owner leaked")
  }
  println(
    "reference_complete hidden_state=\{layers} rows=\{ids.length()} width=\{model.hidden} retained_bytes=0",
  )
  println(
    "reference_projection_arithmetic=" +
    (match arithmetic {
      UpstreamBf16 => "upstream_bf16"
      FullReductionBf16 => "bf16_full_reduction_attribution_only"
    }),
  )
}
