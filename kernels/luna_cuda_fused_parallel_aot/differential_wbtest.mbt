///|
priv struct ScalarQkvResult {
  qkv : Array[Float]
  key_cache : Array[Float]
  value_cache : Array[Float]
  canary : Int
}

///|
fn scalar_dot(
  input : ArrayView[Float],
  weights : ArrayView[Float],
  row : Int,
  input_width : Int,
  column : Int,
) -> Float {
  let mut sum = 0.0F
  for inner in 0..<input_width {
    sum += input[row * input_width + inner] *
      weights[column * input_width + inner]
  }
  sum
}

///|
test "asymmetric projection fixture distinguishes standalone column-major weights" {
  let input = [2.0F, 3.0F, 5.0F]
  let weights = [1.0F, 2.0F, 3.0F, 10.0F, 20.0F, 30.0F]
  assert_eq(scalar_dot(input, weights, 0, 3, 0), 23.0F)
  assert_eq(scalar_dot(input, weights, 0, 3, 1), 230.0F)
  let old_row_major_first = input[0] * weights[0] +
    input[1] * weights[2] +
    input[2] * weights[4]
  assert_true(old_row_major_first != 23.0F)
}

///|
fn scalar_baseline_qkv_rope_kv(
  input : ArrayView[Float],
  q_weight : ArrayView[Float],
  k_weight : ArrayView[Float],
  v_weight : ArrayView[Float],
  positions : ArrayView[Int],
) -> ScalarQkvResult {
  let rows = positions.length()
  let input_width = 4
  let query_heads = 2
  let kv_heads = 1
  let head_dimension = 4
  let query_width = query_heads * head_dimension
  let kv_width = kv_heads * head_dimension
  let output_width = query_width + 2 * kv_width
  let projected = Array::make(rows * output_width, 0.0F)
  for row in 0..<rows {
    for column in 0..<query_width {
      projected[row * output_width + column] = scalar_dot(
        input, q_weight, row, input_width, column,
      )
    }
    for column in 0..<kv_width {
      projected[row * output_width + query_width + column] = scalar_dot(
        input, k_weight, row, input_width, column,
      )
      projected[row * output_width + query_width + kv_width + column] = scalar_dot(
        input, v_weight, row, input_width, column,
      )
    }
  }
  let rotated = projected.copy()
  for row in 0..<rows {
    for head in 0..<(query_heads + kv_heads) {
      let base = row * output_width + head * head_dimension
      for pair in 0..<(head_dimension / 2) {
        let angle = Float::from_int(positions[row]) /
          @math.powf(
            10000.0F,
            Float::from_int(pair * 2) / Float::from_int(head_dimension),
          )
        let first = projected[base + pair]
        let second = projected[base + pair + head_dimension / 2]
        rotated[base + pair] = first * @math.cosf(angle) -
          second * @math.sinf(angle)
        rotated[base + pair + head_dimension / 2] = second * @math.cosf(angle) +
          first * @math.sinf(angle)
      }
    }
  }
  let key_cache = Array::make(32, -99.0F)
  let value_cache = Array::make(32, -99.0F)
  for row in 0..<rows {
    let page = positions[row] / 4
    let token = positions[row] % 4
    let offset = page * 16 + token * kv_width
    for component in 0..<kv_width {
      key_cache[offset + component] = rotated[row * output_width +
        query_width +
        component]
      value_cache[offset + component] = rotated[row * output_width +
        query_width +
        kv_width +
        component]
    }
  }
  { qkv: rotated, key_cache, value_cache, canary: 0 }
}

///|
fn scalar_fused_qkv_rope_kv(
  input : ArrayView[Float],
  q_weight : ArrayView[Float],
  k_weight : ArrayView[Float],
  v_weight : ArrayView[Float],
  positions : ArrayView[Int],
) -> ScalarQkvResult {
  let rows = positions.length()
  let input_width = 4
  let query_width = 8
  let kv_width = 4
  let output_width = 16
  let output = Array::make(rows * output_width, 0.0F)
  let key_cache = Array::make(32, -99.0F)
  let value_cache = Array::make(32, -99.0F)
  let mut canary = 0
  for row in 0..<rows {
    for column in 0..<output_width {
      let (weights, local_column) = if column < query_width {
        (q_weight, column)
      } else if column < query_width + kv_width {
        (k_weight, column - query_width)
      } else {
        (v_weight, column - query_width - kv_width)
      }
      let projected = scalar_dot(input, weights, row, input_width, local_column)
      let mut value = projected
      if column < query_width + kv_width {
        let component = column % 4
        let pair_component = if component < 2 {
          component + 2
        } else {
          component - 2
        }
        let pair_column = column - component + pair_component
        let pair_local = if column < query_width {
          pair_column
        } else {
          pair_column - query_width
        }
        let pair_weights = if column < query_width {
          q_weight
        } else {
          k_weight
        }
        let paired = scalar_dot(
          input, pair_weights, row, input_width, pair_local,
        )
        let angle = Float::from_int(positions[row]) /
          @math.powf(10000.0F, Float::from_int(component % 2 * 2) / 4.0F)
        value = if component < 2 {
          projected * @math.cosf(angle) - paired * @math.sinf(angle)
        } else {
          paired * @math.sinf(angle) + projected * @math.cosf(angle)
        }
      }
      output[row * output_width + column] = value
      if column >= query_width {
        let is_value = column >= query_width + kv_width
        let component = column -
          (if is_value { query_width + kv_width } else { query_width })
        let offset = positions[row] / 4 * 16 +
          positions[row] % 4 * kv_width +
          component
        if is_value {
          value_cache[offset] = value
        } else {
          key_cache[offset] = value
        }
      }
    }
    canary += 1
  }
  { qkv: output, key_cache, value_cache, canary }
}

///|
fn scalar_assert_close(
  actual : ArrayView[Float],
  expected : ArrayView[Float],
  tolerance : Float,
) -> Unit raise {
  assert_eq(actual.length(), expected.length())
  for index in 0..<actual.length() {
    if (actual[index] - expected[index]).abs() > tolerance {
      fail("scalar differential mismatch at " + index.to_string())
    }
  }
}

///|
test "fused QKV positioned RoPE and page-edge KV scatter match scalar pipeline" {
  let input = [0.25F, -0.5F, 0.75F, -1.0F, -0.125F, 0.375F, -0.625F, 0.875F]
  let q_weight = Array::makei(32, index => {
    Float::from_int(index % 11 - 5) * 0.03125F
  })
  let k_weight = Array::makei(16, index => {
    Float::from_int(index % 7 - 3) * 0.0625F
  })
  let v_weight = Array::makei(16, index => {
    Float::from_int(index % 5 - 2) * -0.09375F
  })
  let positions = [3, 4]
  let baseline = scalar_baseline_qkv_rope_kv(
    input, q_weight, k_weight, v_weight, positions,
  )
  let fused = scalar_fused_qkv_rope_kv(
    input, q_weight, k_weight, v_weight, positions,
  )
  scalar_assert_close(fused.qkv, baseline.qkv, 0.000001F)
  scalar_assert_close(fused.key_cache, baseline.key_cache, 0.000001F)
  scalar_assert_close(fused.value_cache, baseline.value_cache, 0.000001F)
  assert_eq(baseline.canary, 0)
  assert_eq(fused.canary, 2)
  assert_eq(fused.key_cache[0], -99.0F)
  assert_true(fused.key_cache[12] != -99.0F)
  assert_true(fused.key_cache[16] != -99.0F)
  assert_eq(fused.key_cache[20], -99.0F)
}

///|
fn scalar_residual_then_rmsnorm(
  residual : ArrayView[Float],
  branch : ArrayView[Float],
  weight : ArrayView[Float],
  width : Int,
) -> (Array[Float], Array[Float], Int) {
  let rows = residual.length() / width
  let sum = Array::make(residual.length(), 0.0F)
  let norm = Array::make(residual.length(), 0.0F)
  for row in 0..<rows {
    let mut squares = 0.0F
    for column in 0..<width {
      let offset = row * width + column
      sum[offset] = residual[offset] + branch[offset]
      squares += sum[offset] * sum[offset]
    }
    let scale = 1.0F / (squares / Float::from_int(width) + 0.00001F).sqrt()
    for column in 0..<width {
      let offset = row * width + column
      norm[offset] = sum[offset] * scale * weight[column]
    }
  }
  (sum, norm, 0)
}

///|
fn scalar_fused_residual_rmsnorm(
  residual : ArrayView[Float],
  branch : ArrayView[Float],
  weight : ArrayView[Float],
  width : Int,
) -> (Array[Float], Array[Float], Int) {
  let rows = residual.length() / width
  let sum = Array::make(residual.length(), 0.0F)
  let norm = Array::make(residual.length(), 0.0F)
  let partial = FixedArray::make(128, 0.0F)
  let mut canary = 0
  for row in 0..<rows {
    for lane in 0..<128 {
      let mut lane_sum = 0.0F
      for column = lane; column < width; column = column + 128 {
        let offset = row * width + column
        sum[offset] = residual[offset] + branch[offset]
        lane_sum += sum[offset] * sum[offset]
      }
      partial[lane] = lane_sum
    }
    for stride = 64; stride > 0; stride = stride / 2 {
      for lane in 0..<stride {
        partial[lane] += partial[lane + stride]
      }
    }
    let scale = 1.0F / (partial[0] / Float::from_int(width) + 0.00001F).sqrt()
    for column in 0..<width {
      let offset = row * width + column
      norm[offset] = sum[offset] * scale * weight[column]
    }
    canary += 1
  }
  (sum, norm, canary)
}

///|
test "fused residual RMSNorm tree agrees with independent scalar fallback" {
  let width = 256
  let residual = Array::makei(width * 2, index => {
    Float::from_int(index % 17 - 8) * 0.03125F
  })
  let branch = Array::makei(width * 2, index => {
    Float::from_int(index % 13 - 6) * -0.015625F
  })
  let weight = Array::makei(width, index => {
    0.75F + Float::from_int(index % 5) * 0.0625F
  })
  let (baseline_sum, baseline_norm, baseline_canary) = scalar_residual_then_rmsnorm(
    residual, branch, weight, width,
  )
  let (fused_sum, fused_norm, fused_canary) = scalar_fused_residual_rmsnorm(
    residual, branch, weight, width,
  )
  scalar_assert_close(fused_sum, baseline_sum, 0.0F)
  scalar_assert_close(fused_norm, baseline_norm, 0.000002F)
  assert_eq(baseline_canary, 0)
  assert_eq(fused_canary, 2)
}

///|
test "fused candidate source recipe and evidence bounds fail closed" {
  let fixture = fused_test_fixture()
  let candidate = fused_test_qkv_candidate(fixture)
  try
    lower_fused_qkv_rope_paged_kv_write_candidate(
      fixture.plan,
      fixture.layout,
      fixture.qkv,
      fixture.rope,
      fixture.kv_write,
      fixture.target,
      fixture.compiler,
      FusedParallelAotLimits::new(
        maximum_source_bytes=candidate.source_bytes().length() - 1,
        maximum_recipe_bytes=65536,
        maximum_promotion_bytes=65536,
      ),
    )
  catch {
    FusedParallelLimitExceeded(Source) => ()
    _ => fail("unexpected source-bound rejection")
  } noraise {
    _ => fail("undersized source limit admitted candidate")
  }
  try
    FusedParallelAotLimits::new(
      maximum_source_bytes=0,
      maximum_recipe_bytes=1,
      maximum_promotion_bytes=1,
    )
  catch {
    FusedParallelInvalid(Limits) => ()
    _ => fail("unexpected limits rejection")
  } noraise {
    _ => fail("invalid limits were admitted")
  }
  try FusedParallelDigest::from_sha256("0".repeat(64)) catch {
    FusedParallelInvalid(Digest) => ()
    _ => fail("unexpected digest rejection")
  } noraise {
    _ => fail("zero digest was admitted")
  }
}
