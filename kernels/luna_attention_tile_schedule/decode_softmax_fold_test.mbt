///|
test "decode finite score map covers ragged active prefixes once" {
  for tile in [1, 17, 32, 64, 128] {
    for partitions in [1, 2, 3, 4, 8, 32] {
      let offsets = @luna_attention_tile_schedule.scalarize_attention_decode_softmax_tile(
        tile, partitions,
      )
      for active in 0..<=tile {
        let visits = Array::make(tile, 0)
        for part in 0..<partitions {
          for offset in offsets {
            let key = part + offset
            if key < active {
              visits[key] += 1
            }
          }
        }
        for key in 0..<tile {
          assert_eq(visits[key], if key < active { 1 } else { 0 })
        }
      }
    }
  }
}

///|
test "decode finite score map rejects invalid geometry" {
  for geometry in [(0, 4), (129, 4), (32, 0), (32, 33)] {
    let (tile, partitions) = geometry
    try
      @luna_attention_tile_schedule.scalarize_attention_decode_softmax_tile(
        tile, partitions,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid tile fold accepted")
    }
  }
}

///|
// The device performs F32 arithmetic; this independent Double implementation
// checks the identity, masks, and max-delta algebra, not device bit equality.
fn chunked_softmax_partition(
  scores : ArrayView[Double],
  values : ArrayView[Double],
  active : Int,
  tile : Int,
  partitions : Int,
  part : Int,
) -> (Double, Double, Double) raise {
  let offsets = @luna_attention_tile_schedule.scalarize_attention_decode_softmax_tile(
    tile, partitions,
  )
  for base = 0, maximum = -1.0e300, denominator = 0.0, numerator = 0.0; base <
     active; {
    let mut tile_maximum = -1.0e300
    let mut has_key = false
    for offset in offsets {
      let key = base + part + offset
      if part + offset < tile && key < active {
        tile_maximum = Double::max(tile_maximum, scores[key])
        has_key = true
      }
    }
    if !has_key {
      continue base + tile, maximum, denominator, numerator
    }
    let next_maximum = Double::max(maximum, tile_maximum)
    let scale = if denominator == 0.0 {
      0.0
    } else {
      @math.exp(maximum - next_maximum)
    }
    let mut next_denominator = denominator * scale
    let mut next_numerator = numerator * scale
    for offset in offsets {
      let key = base + part + offset
      if part + offset < tile && key < active {
        let weight = @math.exp(scores[key] - next_maximum)
        next_denominator += weight
        next_numerator += weight * values[key]
      }
    }
    continue base + tile, next_maximum, next_denominator, next_numerator
  } nobreak {
    (maximum, denominator, numerator)
  }
}

///|
test "tile softmax is stable across causal tails empty partitions and large logits" {
  for shift in [-10000.0, 0.0, 10000.0] {
    let scores = Array::makei(1540, i => {
      shift + (i * 37 % 127 - 63).to_double() * 0.25
    })
    let values = Array::makei(1540, i => (i * 11 % 31 - 15).to_double() / 7.0)
    for active in [0, 1, 3, 31, 32, 33, 64, 65, 129, 1537] {
      // Future/masked values must not enter the maximum or weighted sum.
      let masked_scores = scores.copy()
      for key in active..<scores.length() {
        masked_scores[key] = 1.0e100
      }
      let expected_maximum = scores[:active].fold(init=-1.0e300, Double::max)
      let expected_denominator = scores[:active].fold(init=0.0, (sum, score) => {
        sum + @math.exp(score - expected_maximum)
      })
      let expected_numerator = for key in 0..<active; sum = 0.0 {
        continue sum + @math.exp(scores[key] - expected_maximum) * values[key]
      } nobreak {
        sum
      }
      for tile in [17, 32, 64] {
        for partitions in [1, 2, 4, 8] {
          let states = Array::makei(partitions, part => {
            chunked_softmax_partition(
              masked_scores, values, active, tile, partitions, part,
            )
          })
          let maximum = states.fold(init=-1.0e300, (m, state) => {
            if state.1 > 0.0 {
              Double::max(m, state.0)
            } else {
              m
            }
          })
          let (denominator, numerator) = states.fold(init=(0.0, 0.0), (
            totals,
            state,
          ) => {
            if state.1 == 0.0 {
              totals
            } else {
              let scale = @math.exp(state.0 - maximum)
              (totals.0 + state.1 * scale, totals.1 + state.2 * scale)
            }
          })
          if active == 0 {
            assert_eq(denominator, 0.0)
            assert_eq(numerator, 0.0)
          } else {
            assert_true((maximum - expected_maximum).abs() < 1.0e-12)
            assert_true(
              (denominator / expected_denominator - 1.0).abs() < 1.0e-10,
            )
            assert_true(
              (numerator / denominator -
              expected_numerator / expected_denominator).abs() <
              1.0e-10,
            )
          }
        }
      }
    }
  }
}
