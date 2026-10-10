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
) -> @native.Tensor raise {
  let w = model.weight(session, prefix + name + ".weight", rows, columns)
  defer w.close()
  session.run(Linear, x~, y=w)
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
) -> @native.Tensor raise {
  let prefix = "model.language_model.layers.\{layer}."
  let normalized = norm(
    model,
    session,
    x,
    prefix,
    "input_layernorm",
    model.hidden,
  )
  defer normalized.close()
  let q = project(
    model,
    session,
    normalized,
    prefix,
    "self_attn.q_proj",
    model.heads * model.dim,
    model.hidden,
  )
  defer q.close()
  let k = project(
    model,
    session,
    normalized,
    prefix,
    "self_attn.k_proj",
    model.kv_heads * model.dim,
    model.hidden,
  )
  defer k.close()
  let v = project(
    model,
    session,
    normalized,
    prefix,
    "self_attn.v_proj",
    model.kv_heads * model.dim,
    model.hidden,
  )
  defer v.close()
  let qn = norm(model, session, q, prefix, "self_attn.q_norm", model.dim)
  defer qn.close()
  let kn = norm(model, session, k, prefix, "self_attn.k_norm", model.dim)
  defer kn.close()
  let qr = session.run(
    Rotary,
    x=qn,
    y=cosine,
    z=sine,
    a=model.heads,
    b=model.dim,
  )
  defer qr.close()
  let kr = session.run(
    Rotary,
    x=kn,
    y=cosine,
    z=sine,
    a=model.kv_heads,
    b=model.dim,
  )
  defer kr.close()
  let attention = session.run(
    Attention,
    x=qr,
    y=kr,
    z=v,
    a=model.heads,
    b=model.kv_heads,
    c=model.dim,
  )
  defer attention.close()
  let output = project(
    model,
    session,
    attention,
    prefix,
    "self_attn.o_proj",
    model.hidden,
    model.heads * model.dim,
  )
  defer output.close()
  let residual = session.run(Add, x~, y=output)
  defer residual.close()
  let mlp_input = norm(
    model,
    session,
    residual,
    prefix,
    "post_attention_layernorm",
    model.hidden,
  )
  defer mlp_input.close()
  let gate = project(
    model,
    session,
    mlp_input,
    prefix,
    "mlp.gate_proj",
    model.intermediate,
    model.hidden,
  )
  defer gate.close()
  let up = project(
    model,
    session,
    mlp_input,
    prefix,
    "mlp.up_proj",
    model.intermediate,
    model.hidden,
  )
  defer up.close()
  let activation = session.run(SiluMultiply, x=gate, y=up)
  defer activation.close()
  let down = project(
    model,
    session,
    activation,
    prefix,
    "mlp.down_proj",
    model.hidden,
    model.intermediate,
  )
  defer down.close()
  session.run(Add, x=residual, y=down)
}

///|
fn encode(
  session : @native.Session,
  root : String,
  ids : ArrayView[Int],
  output : String,
  layers : Int,
) -> Unit raise {
  guard ids.length() > 0 && ids.length() <= 1024 && layers > 0 && layers <= 50 else {
    fail("reference request must use 1..1024 tokens and 1..50 selected layers")
  }
  let model = Model::new(session, root)
  let cosine = session.run(Cosine, a=ids.length(), b=model.dim, d=model.theta)
  defer cosine.close()
  let sine = session.run(Sine, a=ids.length(), b=model.dim, d=model.theta)
  defer sine.close()
  let initial = embedding(model, session, ids)
  initial.write(output + "/hidden-0.bf16")
  for layer = 0, hidden = initial; layer < layers; {
    let next = encode_layer(model, session, hidden, layer, cosine, sine)
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
  guard session.retained_bytes() == 0L else {
    fail("reference tensor owner leaked")
  }
  println(
    "reference_complete hidden_state=\{layers} rows=\{ids.length()} width=\{model.hidden} retained_bytes=0",
  )
}
