///|
fn bf16_value(bytes : Bytes, index : Int) -> Float {
  let bits = bytes[index * 2].to_uint() | (bytes[index * 2 + 1].to_uint() << 8)
  Float::reinterpret_from_uint(bits << 16)
}

///|
fn bf16_bits(value : Float) -> UInt {
  let bits = value.reinterpret_as_uint()
  (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16
}

///|
/// Explicit wider intermediates preserve separately rounded F32 operations
/// without relying on a host compiler's FMA contraction setting.
fn ordered_product_sum(sum : Float, x : Float, w : Float) -> Float {
  let product = Float::from_double(x.to_double() * w.to_double())
  Float::from_double(sum.to_double() + product.to_double())
}

///|
test "ordered replay preserves F32 rounding and BF16 tie-to-even" {
  assert_eq(ordered_product_sum(100000000.0F, 1.0F, 1.0F), 100000000.0F)
  assert_eq(bf16_bits(1.00390625F), 0x3f80U)
  assert_eq(bf16_bits(1.01171875F), 0x3f82U)
  assert_eq(bf16_value(b"\x80\x3f", 0), 1.0F)
}

///|
/// CPU replay of every first-layer native down output differing from F64.
/// Only corresponding original weight rows are read. This is arithmetic
/// attribution, not a production implementation or a model acceptance gate.
fn replay_down_reduction(
  session : @native.Session,
  root : String,
  input_root : String,
  oracle_root : String,
) -> Unit raise {
  let model = Model::new(session, root)
  let native = session.bytes(input_root + "/down.bf16")
  let oracle = session.bytes(oracle_root + "/down-f64.bf16")
  let input = session.bytes(input_root + "/activation.bf16")
  guard native.length() == 3 * model.hidden * 2 &&
    oracle.length() == native.length() &&
    input.length() == 3 * model.intermediate * 2 else {
    fail("down replay requires the complete matched three-row operands")
  }
  let (path, offset) = model.region(
    session,
    "model.language_model.layers.0.mlp.down_proj.weight",
    model.hidden,
    model.intermediate,
  )
  let mut selected = 0
  let mut ordered_differences = 0
  let mut wider_differences = 0
  for index in 0..<(3 * model.hidden) {
    let actual = bf16_value(native, index)
    let expected = bf16_value(oracle, index)
    guard !actual.is_nan() &&
      !actual.is_inf() &&
      !expected.is_nan() &&
      !expected.is_inf() else {
      fail("nonfinite down replay boundary")
    }
    if actual != expected {
      let row = index / model.hidden
      let column = index % model.hidden
      let weight = session.bytes(
        path,
        offset=offset + column.to_int64() * model.intermediate.to_int64() * 2L,
        length=model.intermediate * 2,
      )
      let (ordered, wider) = for
        inner in 0..<model.intermediate
        ordered = 0.0F, wider = 0.0 {
        let x = bf16_value(input, row * model.intermediate + inner)
        let w = bf16_value(weight, inner)
        continue ordered_product_sum(ordered, x, w),
          wider + x.to_double() * w.to_double()
      } nobreak {
        (ordered, wider)
      }
      let native_bits = actual.reinterpret_as_uint() >> 16
      let oracle_bits = expected.reinterpret_as_uint() >> 16
      if bf16_bits(ordered) != native_bits {
        ordered_differences += 1
      }
      if bf16_bits(Float::from_double(wider)) != oracle_bits {
        wider_differences += 1
      }
      selected += 1
      println(
        "down_reduction index=\{index} native=\{actual} oracle=\{expected} ordered_f32=\{ordered} wider_f64=\{wider} ordered_matches=\{bf16_bits(ordered) == native_bits} wider_matches=\{bf16_bits(Float::from_double(wider)) == oracle_bits}",
      )
    }
  }
  println(
    "down_reduction_replay selected=\{selected} ordered_differences=\{ordered_differences} wider_differences=\{wider_differences} retained_bytes=\{session.retained_bytes()} scope=first_layer_native_vs_f64_mismatches_only",
  )
  guard selected > 0 && ordered_differences == 0 && wider_differences == 0 else {
    fail(
      "independent down reduction replay does not explain the selected outputs",
    )
  }
}
