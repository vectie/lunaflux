///|
test "blockwise law separates F32 probabilities from BF16 and ordered-key folds" {
  let fold = @attention_physical_ir.BlockwiseFold::new(dimensions=128, keys=32).unwrap()
  assert_eq(fold.numeric_law(), "blockwise-f32-probability-v1")
  assert_eq(fold.dimensions(), (128, 32))
  assert_eq(fold.steps(), [
    ComputeTileScores,
    PublishTileScores,
    ReduceTileMaximum,
    FormFloat32Probabilities,
    ReduceTileDenominator,
    PublishTileProbabilities,
    AccumulateTileValues,
    MergeTileState,
  ])
  assert_true(
    @attention_physical_ir.BlockwiseFold::new(dimensions=0, keys=32) is None,
  )
  assert_true(
    @attention_physical_ir.BlockwiseFold::new(dimensions=128, keys=0) is None,
  )
}

///|
/// Test interpreter for the numerical law (no CUDA and no runtime fallback).
/// Matches subgroup 32-lane probability sum and ordered tile/PV traversal.
fn blockwise_value(
  scores : ArrayView[Float],
  values : ArrayView[Float],
  tile : Int,
) -> Float {
  for begin = 0, maximum = -3.4028235e38F, denominator = 0.0F, numerator = 0.0F; begin <
     scores.length(); {
    let end = Int::min(begin + tile, scores.length())
    let tile_max = for i in begin..<end; m = -3.4028235e38F {
      continue Float::max(m, scores[i])
    } nobreak {
      m
    }
    let weights = Array::make(end - begin, 0.0F)
    let lanes = Array::make(32, 0.0F)
    for i in begin..<end {
      let p = @math.expf(scores[i] - tile_max)
      weights[i - begin] = p
      lanes[(i - begin) % 32] += p
    }
    for stride = 16; stride > 0; stride = stride / 2 {
      for lane in 0..<stride {
        lanes[lane] += lanes[lane + stride]
      }
    }
    let tile_numerator = for i in begin..<end; n = 0.0F {
      continue n + weights[i - begin] * values[i]
    } nobreak {
      n
    }
    let next = Float::max(maximum, tile_max)
    let old_scale = if denominator == 0.0F {
      0.0F
    } else {
      @math.expf(maximum - next)
    }
    let tile_scale = @math.expf(tile_max - next)
    continue end,
      next,
      denominator * old_scale + lanes[0] * tile_scale,
      numerator * old_scale + tile_numerator * tile_scale
  } nobreak {
    numerator / denominator
  }
}

///|
test "blockwise numerical law handles tails and changing tile maxima" {
  for count in [1, 17, 31, 32, 33, 63, 64, 65, 129, 257, 4096] {
    let scores = Array::makei(count, i => {
      Float::from_int(i * 17 % 97 - 48) / 8.0F
    })
    let values = Array::makei(count, i => {
      Float::from_int(i * 31 % 67 - 33) / 32.0F
    })
    let max_score = for s in scores; m = -1.0e30 {
      continue Double::max(m, s.to_double())
    } nobreak {
      m
    }
    let (denominator, numerator) = for i in 0..<count; d = 0.0, n = 0.0 {
      let p = @math.exp(scores[i].to_double() - max_score)
      continue d + p, n + p * values[i].to_double()
    } nobreak {
      (d, n)
    }
    for tile in [32, 64] {
      let got = blockwise_value(scores, values, tile).to_double()
      assert_true((got - numerator / denominator).abs() < 0.00001)
      assert_eq(
        blockwise_value(scores, values, tile),
        blockwise_value(scores, values, tile),
      )
    }
  }
}
