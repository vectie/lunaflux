///|
fn oracle_float_finite(value : Float) -> Bool {
  !value.is_nan() && value != @float.infinity && value != @float.neg_infinity
}

///|
fn oracle_moe_swiglu(
  input : ArrayView[Double],
  router : ArrayView[Float],
  correction : ArrayView[Float],
  gate : ArrayView[Double],
  up : ArrayView[Double],
  down : ArrayView[Double],
  rows : Int,
  width : Int,
  intermediate : Int,
  expert_count : Int,
  experts_per_token : Int,
  score_function : @advanced_decoder_execution_plan.ExpertScoreFunction,
  normalize_selected : Bool,
  route_scale : Float,
  limits : GlmTinyOracleLimits,
) -> (Array[Double], Array[Int]) raise GlmHybridReferenceError {
  if router.length() != expert_count * width ||
    correction.length() != expert_count ||
    gate.length() != expert_count * intermediate * width ||
    up.length() != gate.length() ||
    down.length() != expert_count * width * intermediate {
    raise InvalidShape
  }
  oracle_check_values(router.length(), limits)
  oracle_check_values(correction.length(), limits)
  for value in router {
    if !oracle_float_finite(value) {
      raise NonFiniteInput
    }
  }
  for value in correction {
    if !oracle_float_finite(value) {
      raise NonFiniteInput
    }
  }
  oracle_require_finite(gate, limits)
  oracle_require_finite(up, limits)
  oracle_require_finite(down, limits)
  let output = Array::make(rows * width, 0.0)
  let selected : Array[Int] = []
  let reference_limits = @advanced_decoder_reference.ReferenceLimits::new(
    max_values=limits.maximum_values,
    max_width=limits.maximum_width,
    max_iterations=32,
  ) catch {
    _ => raise SharedPrimitiveFailure
  }
  for row in 0..<rows {
    let logits = Array::make(expert_count, 0.0F)
    for expert in 0..<expert_count {
      let mut sum = 0.0F
      for component in 0..<width {
        sum += Float::from_double(input[row * width + component]) *
          router[expert * width + component]
      }
      if !oracle_float_finite(sum) {
        raise NonFiniteInput
      }
      logits[expert] = sum
    }
    let routing = @advanced_decoder_reference.route_top_k(
      logits,
      correction,
      top_k=experts_per_token,
      score_function~,
      normalize=normalize_selected,
      scaling=route_scale,
      limits=reference_limits,
    ) catch {
      _ => raise SharedPrimitiveFailure
    }
    for rank in 0..<routing.selected_count() {
      let expert = routing.expert_indices()[rank]
      selected.push(expert)
      let gate_start = expert * intermediate * width
      let down_start = expert * width * intermediate
      let expert_output = oracle_swiglu(
        input[row * width:(row + 1) * width],
        gate[gate_start:gate_start + intermediate * width],
        up[gate_start:gate_start + intermediate * width],
        down[down_start:down_start + width * intermediate],
        1,
        width,
        intermediate,
        limits,
      )
      let route_weight = routing.weights()[rank].to_double()
      for component in 0..<width {
        output[row * width + component] += route_weight *
          expert_output[component]
      }
    }
  }
  oracle_require_finite(output, limits)
  (output, selected)
}

///|
fn oracle_feed_forward(
  input : ArrayView[Double],
  plan : GlmTinyHybridBlockPlan,
  weights : GlmTinyFeedForwardWeights,
  width : Int,
  limits : GlmTinyOracleLimits,
) -> (Array[Double], Array[Int]) raise GlmHybridReferenceError {
  match (plan.feed_forward, weights.storage) {
    (Dense, DenseWeights(gate, up, down)) =>
      (
        oracle_swiglu(
          input,
          gate,
          up,
          down,
          plan.token_count,
          width,
          plan.intermediate_size,
          limits,
        ),
        [],
      )
    (
      Moe(
        expert_count~,
        experts_per_token~,
        score_function~,
        normalize_selected~,
        route_scale~
      ),
      MoeWeights(router, correction, gate, up, down),
    ) =>
      oracle_moe_swiglu(
        input,
        router,
        correction,
        gate,
        up,
        down,
        plan.token_count,
        width,
        plan.intermediate_size,
        expert_count,
        experts_per_token,
        score_function,
        normalize_selected,
        route_scale,
        limits,
      )
    _ => raise InvalidShape
  }
}
