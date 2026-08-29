///|
fn phase4_checked_add(
  left : Int64,
  right : Int64,
) -> Int64 raise Phase4PrefixGateError {
  if left < 0L || right < 0L || left > 9223372036854775807L - right {
    raise Phase4IntegerOverflow
  }
  left + right
}

///|
fn phase4_fraction_compare(
  left_numerator : Int64,
  left_denominator : Int64,
  right_numerator : Int64,
  right_denominator : Int64,
) -> Int raise Phase4PrefixGateError {
  if left_numerator < 0L ||
    left_denominator <= 0L ||
    right_numerator < 0L ||
    right_denominator <= 0L {
    raise Phase4InvalidMeasurement
  }
  for left_n = left_numerator, left_d = left_denominator, right_n = right_numerator, right_d = right_denominator, reversed = false {
    let left_whole = left_n / left_d
    let right_whole = right_n / right_d
    if left_whole != right_whole {
      let ordinary = if left_whole > right_whole { 1 } else { -1 }
      break if reversed { -ordinary } else { ordinary }
    }
    let left_remainder = left_n % left_d
    let right_remainder = right_n % right_d
    if left_remainder == 0L || right_remainder == 0L {
      let ordinary = if left_remainder == right_remainder {
        0
      } else if left_remainder > 0L {
        1
      } else {
        -1
      }
      break if reversed { -ordinary } else { ordinary }
    }
    continue left_d, left_remainder, right_d, right_remainder, !reversed
  }
}

///|
fn phase4_rate_at_least(
  candidate : Phase4ProfileAggregate,
  baseline : Phase4ProfileAggregate,
) -> Bool raise Phase4PrefixGateError {
  phase4_fraction_compare(
    candidate.completed,
    candidate.elapsed_ns,
    baseline.completed,
    baseline.elapsed_ns,
  ) >=
  0
}

///|
fn phase4_rate_within_budget(
  candidate : Phase4ProfileAggregate,
  baseline : Phase4ProfileAggregate,
  budget_bps : Int,
) -> Bool raise Phase4PrefixGateError {
  if budget_bps < 0 || budget_bps > PHASE4_MAX_COLD_REGRESSION_BPS {
    raise Phase4InvalidPolicy
  }
  let retained = PHASE4_BASIS_POINTS - Int64::from_int(budget_bps)
  let required_numerator = baseline.completed * retained
  let required_denominator = baseline.elapsed_ns * PHASE4_BASIS_POINTS
  if required_numerator < 0L || required_denominator <= 0L {
    raise Phase4IntegerOverflow
  }
  phase4_fraction_compare(
    candidate.completed,
    candidate.elapsed_ns,
    required_numerator,
    required_denominator,
  ) >=
  0
}

///|
fn phase4_best_baseline(
  matrix : Phase4ProfileMatrix,
) -> (String, Phase4ProfileAggregate) raise Phase4PrefixGateError {
  if phase4_rate_at_least(matrix.vllm, matrix.sglang) {
    ("vllm", matrix.vllm)
  } else {
    ("sglang", matrix.sglang)
  }
}
