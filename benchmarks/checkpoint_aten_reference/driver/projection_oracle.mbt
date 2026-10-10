///|
/// Compare independent BF16/full-reduction/F32/F64 projections on identical input and
/// original checkpoint weights. Outputs are still BF16 at the same boundary.
fn projection_oracle(
  session : @native.Session,
  root : String,
  input_root : String,
  output : String,
) -> Unit raise {
  let model = Model::new(session, root)
  for
    (name, input, weight, width, columns) in [
      (
        "q",
        "norm-input",
        "self_attn.q_proj",
        model.heads * model.dim,
        model.hidden,
      ),
      (
        "k",
        "norm-input",
        "self_attn.k_proj",
        model.kv_heads * model.dim,
        model.hidden,
      ),
      (
        "v",
        "norm-input",
        "self_attn.v_proj",
        model.kv_heads * model.dim,
        model.hidden,
      ),
      (
        "output",
        "attention",
        "self_attn.o_proj",
        model.hidden,
        model.heads * model.dim,
      ),
      ("down", "activation", "mlp.down_proj", model.hidden, model.intermediate),
    ] {
    let x = session.load(input_root + "/" + input + ".bf16", 0L, 3, columns)
    defer x.close()
    let w = model.weight(
      session,
      "model.language_model.layers.0." + weight + ".weight",
      width,
      columns,
    )
    defer w.close()
    for
      (precision, op) in [
        ("bf16", @native.Op::Linear),
        ("bf16-full", @native.Op::LinearBf16FullReduction),
        ("f32", @native.Op::LinearFloat32),
        ("f64", @native.Op::LinearFloat64),
      ] {
      let y = session.run(op, x~, y=w)
      defer y.close()
      y.write(output + "/" + name + "-" + precision + ".bf16")
      println(
        "projection_oracle_complete name=\{name} accumulation=\{precision} rows=3 width=\{width}",
      )
    }
  }
  guard session.retained_bytes() == 0L else {
    fail("projection oracle owner leaked")
  }
  println("projection_oracle_complete retained_bytes=0")
}
