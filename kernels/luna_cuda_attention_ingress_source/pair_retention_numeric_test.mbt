///|
fn round_bf16(value : Float) -> Float {
  let bits = value.reinterpret_as_uint()
  let rounded = bits + 0x7fffU + ((bits >> 16) & 1U)
  Float::reinterpret_from_uint(rounded & 0xffff0000U)
}

///|
/// Host differential isolates the exact rewrite, not CUDA trigonometry or
/// device rsqrt. The same externally supplied inverse-RMS and rotary operands
/// are used on each side, including signed zero and BF16 halfway inputs.
test "retained local pairs preserve staged BF16 normalization and rotation" {
  for dimension in [64, 128] {
    let half = dimension / 2
    let input = Array::makei(dimension, index => {
      if index % 11 == 0 {
        -0.0F
      } else {
        Float::from_int(index % 19 - 9) * 0.031373F
      }
    })
    let weight = Array::makei(dimension, index => {
      round_bf16(Float::from_int(index % 7 - 3) * 0.125F)
    })
    let inverse = 1.03125F
    let staged = Array::makei(dimension, index => {
      round_bf16(input[index] * inverse * weight[index])
    })
    let expected = Array::make(dimension, 0.0F)
    let actual = Array::make(dimension, 0.0F)
    let ownership = @elementwise_physical_ir.SplitPairRetention::new(
      width=dimension,
      owners=32,
    ).unwrap()
    for pair in 0..<half {
      let sine = Float::from_int(pair % 5 - 2) * 0.125F
      let cosine = 0.96875F
      expected[pair] = round_bf16(
        staged[pair] * cosine - staged[pair + half] * sine,
      )
      expected[pair + half] = round_bf16(
        staged[pair + half] * cosine + staged[pair] * sine,
      )
    }
    for lane in 0..<32 {
      let first = Array::makei(half / 32, item => input[lane + item * 32])
      let second = Array::makei(half / 32, item => {
        input[lane + item * 32 + half]
      })
      // Register retention must also retain the original ascending fold,
      // not accidentally change it to pair-interleaved square accumulation.
      let old_sum = for component in lane..<dimension; sum = 0.0F {
        if component % 32 == lane {
          continue sum + input[component] * input[component]
        } else {
          continue sum
        }
      } nobreak {
        sum
      }
      let first_sum = first.fold(init=0.0F, (sum, value) => sum + value * value)
      let retained_sum = second.fold(init=first_sum, (sum, value) => {
        sum + value * value
      })
      assert_eq(
        retained_sum.reinterpret_as_uint(),
        old_sum.reinterpret_as_uint(),
      )
      for item in 0..<(half / 32) {
        let (pair, second_index) = ownership.components(lane, item).unwrap()
        let normalized_first = round_bf16(first[item] * inverse * weight[pair])
        let normalized_second = round_bf16(
          second[item] * inverse * weight[second_index],
        )
        let sine = Float::from_int(pair % 5 - 2) * 0.125F
        let cosine = 0.96875F
        actual[pair] = round_bf16(
          normalized_first * cosine - normalized_second * sine,
        )
        actual[second_index] = round_bf16(
          normalized_second * cosine + normalized_first * sine,
        )
      }
    }
    for index in 0..<dimension {
      assert_eq(
        actual[index].reinterpret_as_uint(),
        expected[index].reinterpret_as_uint(),
      )
    }
  }
}
