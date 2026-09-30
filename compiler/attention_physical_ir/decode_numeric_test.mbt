///|
/// Paged/GQA oracle is intentionally independent of the ownership interpreter:
/// F64 ascending dots and one globally normalized softmax. The compiler law
/// uses owner-rotated F32 trees followed by the declared blockwise fold.
test "dual score blockwise law handles GQA paged masks and ragged tails" {
  let dimensions = 128
  let page_tokens = 8
  for group in [2, 4] {
    for length in [1, 9, 31, 33, 65] {
      let page_count = (length + page_tokens - 1) / page_tokens
      let pages = Array::makei(page_count, i => (i * 3 + 1) % 11)
      for valid_keys in [1, length] {
        for head in 0..<8 {
          let kv_head = head / group
          let q = Array::makei(dimensions, i => {
            Float::from_int((i * 17 + head * 3) % 59 - 29) / 31.0F
          })
          let reference_scores = Array::make(valid_keys, 0.0)
          let scores = Array::make(valid_keys, 0.0F)
          for key in 0..<valid_keys {
            let physical = pages[key / page_tokens] * page_tokens +
              key % page_tokens
            let k = Array::makei(dimensions, i => {
              Float::from_int((i * 31 + physical * 5 + kv_head * 7) % 67 - 33) /
              37.0F
            })
            reference_scores[key] = for i in 0..<dimensions; dot = 0.0 {
              continue dot + q[i].to_double() * k[i].to_double()
            } nobreak {
              dot / dimensions.to_double().sqrt()
            }
            scores[key] = dual_score_dot(q, k, key % 2) /
              dimensions.to_float().sqrt()
          }
          let maximum = reference_scores.fold(init=-1.0e30, Double::max)
          let weights = reference_scores.map(score => @math.exp(score - maximum))
          let denominator = weights.fold(init=0.0, (a, b) => a + b)
          for component in [0, 37, 127] {
            let values = Array::makei(valid_keys, key => {
              let physical = pages[key / page_tokens] * page_tokens +
                key % page_tokens
              Float::from_int(
                (physical * 19 + component * 11 + kv_head) % 71 - 35,
              ) /
              32.0F
            })
            let numerator = for key in 0..<valid_keys; sum = 0.0 {
              continue sum + weights[key] * values[key].to_double()
            } nobreak {
              sum
            }
            for tile in [32, 64] {
              let actual = blockwise_value(scores, values, tile)
              assert_true(
                (actual.to_double() - numerator / denominator).abs() < 0.00001,
              )
              assert_eq(actual, blockwise_value(scores, values, tile))
            }
          }
        }
      }
    }
  }
}
