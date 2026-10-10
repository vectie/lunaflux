///|
struct DotSensitivity {
  first : Double
  second : Double
  absolute_first : Double
  absolute_second : Double
  signed_delta : Double
  absolute_delta : Double
  largest_delta : Double
  largest_delta_index : Int
  largest_input_delta : Double
  largest_input_delta_index : Int
  input_delta_squared : Double
  input_second_squared : Double
}

///|
/// Pure dot-product attribution on BF16 operands, without changing their
/// numerical boundaries. Explicit F64 terms avoid host FMA contraction.
fn dot_sensitivity(
  first : Bytes,
  second : Bytes,
  weight : Bytes,
) -> DotSensitivity raise {
  guard first.length() > 0 &&
    first.length() % 2 == 0 &&
    first.length() == second.length() &&
    first.length() == weight.length() else {
    fail("matched nonempty BF16 input and weight rows required")
  }
  for
    i in 0..<(first.length() / 2)
    left = 0.0,
    right = 0.0,
    absolute_left = 0.0,
    absolute_right = 0.0,
    delta_sum = 0.0,
    absolute_delta = 0.0,
    maximum = -1.0,
    index = 0,
    input_maximum = -1.0,
    input_index = 0,
    delta_squared = 0.0,
    input_squared = 0.0 {
    let x = bf16_value(first, i).to_double()
    let y = bf16_value(second, i).to_double()
    let w = bf16_value(weight, i).to_double()
    guard !x.is_nan() &&
      !x.is_inf() &&
      !y.is_nan() &&
      !y.is_inf() &&
      !w.is_nan() &&
      !w.is_inf() else {
      fail("nonfinite dot attribution operand")
    }
    let a = x * w
    let b = y * w
    let difference = (y - x) * w
    let magnitude = difference.abs()
    let input_magnitude = (y - x).abs()
    continue left + a,
      right + b,
      absolute_left + a.abs(),
      absolute_right + b.abs(),
      delta_sum + difference,
      absolute_delta + magnitude,
      if magnitude > maximum {
        magnitude
      } else {
        maximum
      },
      if magnitude > maximum {
        i
      } else {
        index
      },
      if input_magnitude > input_maximum {
        input_magnitude
      } else {
        input_maximum
      },
      if input_magnitude > input_maximum {
        i
      } else {
        input_index
      },
      delta_squared + (y - x) * (y - x),
      input_squared + y * y
  } nobreak {
    {
      first: left,
      second: right,
      absolute_first: absolute_left,
      absolute_second: absolute_right,
      signed_delta: delta_sum,
      absolute_delta,
      largest_delta: maximum,
      largest_delta_index: index,
      largest_input_delta: input_maximum,
      largest_input_delta_index: input_index,
      input_delta_squared: delta_squared,
      input_second_squared: input_squared,
    }
  }
}

///|
fn diagnostic_ratio(numerator : Double, denominator : Double) -> Double {
  if denominator > 0.0 {
    numerator / denominator
  } else if numerator == 0.0 {
    0.0
  } else {
    numerator / denominator
  }
}

///|
/// Read only one original weight row and two captured normalized-input rows.
/// CPU-only diagnostic; no GPU weights, hashing or serving-side validation.
fn replay_gate_input(
  session : @native.Session,
  root : String,
  layer : Int,
  row : Int,
  channel : Int,
  first_root : String,
  second_root : String,
) -> Unit raise {
  let index = replay_layer_index(1, layer)
  let model = Model::new(session, root)
  guard row >= 0 && row < 1024 && channel >= 0 && channel < model.intermediate else {
    fail("gate attribution row/channel outside model geometry")
  }
  let (path, offset) = model.region(
    session,
    "model.language_model.layers.\{index}.mlp.gate_proj.weight",
    model.intermediate,
    model.hidden,
  )
  let weight = session.bytes(
    path,
    offset=offset + channel.to_int64() * model.hidden.to_int64() * 2L,
    length=model.hidden * 2,
  )
  let input_offset = row.to_int64() * model.hidden.to_int64() * 2L
  let first = session.bytes(
    first_root + "/norm-mlp.bf16",
    offset=input_offset,
    length=model.hidden * 2,
  )
  let second = session.bytes(
    second_root + "/norm-mlp.bf16",
    offset=input_offset,
    length=model.hidden * 2,
  )
  let statistics = dot_sensitivity(first, second, weight)
  println(
    "gate_input_replay layer=\{layer} weight_layer=\{index} row=\{row} channel=\{channel} hidden=\{model.hidden}",
  )
  println(
    "first_dot_f64=\{statistics.first}\nsecond_dot_f64=\{statistics.second}\nsigned_input_delta_dot=\{statistics.signed_delta}\nabsolute_input_delta_dot=\{statistics.absolute_delta}",
  )
  println(
    "first_bf16=\{Float::reinterpret_from_uint(bf16_bits(Float::from_double(statistics.first)) << 16)}\nsecond_bf16=\{Float::reinterpret_from_uint(bf16_bits(Float::from_double(statistics.second)) << 16)}",
  )
  println(
    "first_absolute_products=\{statistics.absolute_first}\nsecond_absolute_products=\{statistics.absolute_second}\nfirst_cancellation_ratio=\{diagnostic_ratio(statistics.absolute_first, statistics.first.abs())}\nsecond_cancellation_ratio=\{diagnostic_ratio(statistics.absolute_second, statistics.second.abs())}",
  )
  println(
    "input_relative_l2=\{diagnostic_ratio(statistics.input_delta_squared, statistics.input_second_squared).sqrt()}\nlargest_weighted_delta_index=\{statistics.largest_delta_index}\nlargest_weighted_delta=\{statistics.largest_delta}",
  )
  for
    (label, selected) in [
      ("largest-weighted", statistics.largest_delta_index),
      ("largest-input", statistics.largest_input_delta_index),
    ] {
    println(
      "contributor=\{label} index=\{selected} first=\{bf16_value(first, selected)} second=\{bf16_value(second, selected)} weight=\{bf16_value(weight, selected)}",
    )
  }
  println(
    "largest_input_delta=\{statistics.largest_input_delta}\nretained_bytes=\{session.retained_bytes()}\nscope=one_original_gate_row_cpu_only_not_full_model_acceptance",
  )
  guard session.retained_bytes() == 0L else {
    fail("CPU gate replay retained a tensor")
  }
}
