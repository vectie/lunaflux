///|
fn better_route_choice(
  value : Float,
  index : Int,
  best_value : Float,
  best_index : Int,
) -> Bool {
  best_index < 0 ||
  value > best_value ||
  (value == best_value && index < best_index)
}

///|
/// Tiny serial grouped-routing oracle. Lower ids deterministically break ties;
/// this is not a PyTorch `topk(sorted=False)` tie-order parity claim.
pub fn moe_grouped_route(
  uncorrected_scores : ArrayView[Float],
  choice_scores : ArrayView[Float],
  row_count~ : Int,
  routed_expert_count~ : Int,
  group_count~ : Int,
  group_score_topk~ : Int,
  selected_group_count~ : Int,
  experts_per_token~ : Int,
  normalize~ : Bool,
  scaling~ : Float,
  limits~ : GlmHybridReferenceLimits,
) -> GlmMoeGroupedRoutingResult raise GlmHybridReferenceError {
  limits.check_sequence(row_count)
  limits.check_channels(routed_expert_count)
  if routed_expert_count <= 0 ||
    group_count <= 0 ||
    routed_expert_count % group_count != 0 ||
    group_score_topk <= 0 ||
    group_score_topk > routed_expert_count / group_count ||
    selected_group_count <= 0 ||
    selected_group_count > group_count ||
    selected_group_count * (routed_expert_count / group_count) <
    experts_per_token ||
    experts_per_token <= 0 ||
    experts_per_token > routed_expert_count ||
    uncorrected_scores.length() != row_count * routed_expert_count ||
    choice_scores.length() != uncorrected_scores.length() ||
    !oracle_float_finite(scaling) ||
    scaling <= 0.0F {
    raise InvalidShape
  }
  for value in uncorrected_scores {
    if !oracle_float_finite(value) {
      raise NonFiniteInput
    }
  }
  for value in choice_scores {
    if !oracle_float_finite(value) {
      raise NonFiniteInput
    }
  }
  let experts_per_group = routed_expert_count / group_count
  let indices = Array::make(row_count * experts_per_token, -1)
  let weights = Array::make(row_count * experts_per_token, 0.0F)
  for row in 0..<row_count {
    let score_base = row * routed_expert_count
    let output_base = row * experts_per_token
    let group_scores = Array::make(group_count, 0.0F)
    for group in 0..<group_count {
      let selected = Array::make(group_score_topk, -1)
      for slot in 0..<group_score_topk {
        let mut best = -1
        let mut best_value = @float.neg_infinity
        for offset in 0..<experts_per_group {
          let expert = group * experts_per_group + offset
          if !selected[:slot].contains(expert) &&
            better_route_choice(
              choice_scores[score_base + expert],
              expert,
              best_value,
              best,
            ) {
            best = expert
            best_value = choice_scores[score_base + expert]
          }
        }
        selected[slot] = best
        group_scores[group] += best_value
      }
    }
    let selected_groups = Array::make(selected_group_count, -1)
    for slot in 0..<selected_group_count {
      let mut best = -1
      let mut best_value = @float.neg_infinity
      for group in 0..<group_count {
        if !selected_groups[:slot].contains(group) &&
          better_route_choice(group_scores[group], group, best_value, best) {
          best = group
          best_value = group_scores[group]
        }
      }
      selected_groups[slot] = best
    }
    for slot in 0..<experts_per_token {
      let mut best = -1
      let mut best_value = @float.neg_infinity
      for expert in 0..<routed_expert_count {
        if selected_groups.contains(expert / experts_per_group) &&
          !indices[output_base:output_base + slot].contains(expert) &&
          better_route_choice(
            choice_scores[score_base + expert],
            expert,
            best_value,
            best,
          ) {
          best = expert
          best_value = choice_scores[score_base + expert]
        }
      }
      indices[output_base + slot] = best
    }
    let mut denominator = 0.0F
    for slot in 0..<experts_per_token {
      denominator += uncorrected_scores[score_base + indices[output_base + slot]]
    }
    if normalize {
      denominator += 1.0e-20F
    }
    for slot in 0..<experts_per_token {
      let value = uncorrected_scores[score_base + indices[output_base + slot]]
      weights[output_base + slot] = (if normalize {
          value / denominator
        } else {
          value
        }) *
        scaling
    }
  }
  {
    row_count,
    experts_per_token,
    selected_indices: ReadOnlyArray::from_array(indices),
    selected_weights: ReadOnlyArray::from_array(weights),
  }
}

///|
pub fn GlmMoeGroupedRoutingResult::selected_indices(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.selected_indices
}

///|
pub fn GlmMoeGroupedRoutingResult::selected_weights(
  self : Self,
) -> ReadOnlyArray[Float] {
  self.selected_weights
}

///|
pub fn GlmMoeGroupedRoutingResult::row_count(self : Self) -> Int {
  self.row_count
}

///|
pub fn GlmMoeGroupedRoutingResult::experts_per_token(self : Self) -> Int {
  self.experts_per_token
}
