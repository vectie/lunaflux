///|
fn activation_combinations() -> Array[(String, Int, Int)] {
  [
    ("first-input", 0, 0),
    ("second-input", 1, 1),
    ("gate-first-up-second", 0, 1),
    ("gate-second-up-first", 1, 0),
  ]
}

///|
fn activation_geometry(rows : Int, width : Int) -> Int64 raise {
  guard rows > 0 && rows <= 1024 && width > 0 && width <= 65536 else {
    fail("bounded activation diagnostic geometry required")
  }
  rows.to_int64() * width.to_int64() * 2L
}

///|
/// Recombine captured projection operands, keeping activation arithmetic fixed.
/// F64 names refer to the preceding projection, whose stored output is BF16.
fn activation_factorization(
  session : @native.Session,
  first : String,
  second : String,
  rows : Int,
  width : Int,
  output : String,
) -> Unit raise {
  ignore(activation_geometry(rows, width))
  for precision in ["bf16", "f64"] {
    let gate_first = session.load(
      first + "/gate-" + precision + ".bf16",
      0L,
      rows,
      width,
    )
    defer gate_first.close()
    let gate_second = session.load(
      second + "/gate-" + precision + ".bf16",
      0L,
      rows,
      width,
    )
    defer gate_second.close()
    let up_first = session.load(
      first + "/up-" + precision + ".bf16",
      0L,
      rows,
      width,
    )
    defer up_first.close()
    let up_second = session.load(
      second + "/up-" + precision + ".bf16",
      0L,
      rows,
      width,
    )
    defer up_second.close()
    for (name, gate, up) in activation_combinations() {
      let x = if gate == 0 { gate_first } else { gate_second }
      let y = if up == 0 { up_first } else { up_second }
      let product = session.run(SiluMultiply, x~, y~)
      defer product.close()
      product.write(output + "/" + name + "-" + precision + ".bf16")
      println(
        "activation_factorization_complete case=\{name} projection=\{precision} rows=\{rows} width=\{width}",
      )
    }
  }
  guard session.retained_bytes() == 0L else {
    fail("activation oracle owner leaked")
  }
  println("activation_factorization_complete retained_bytes=0")
}
