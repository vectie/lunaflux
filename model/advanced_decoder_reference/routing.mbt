///|
fn score_one(
  value : Float,
  function : @advanced_decoder_execution_plan.ExpertScoreFunction,
) -> Float {
  match function {
    Sigmoid =>
      if value >= 0.0F {
        1.0F / (1.0F + @math.expf(-value))
      } else {
        let exponential = @math.expf(value)
        exponential / (1.0F + exponential)
      }
    SqrtSoftplus => {
      let softplus = if value > 0.0F {
        value + @math.ln_1pf(@math.expf(-value))
      } else {
        @math.ln_1pf(@math.expf(value))
      }
      softplus.sqrt()
    }
  }
}

///|
/// Applies the plan-selected expert score transform with stable formulas.
pub fn transform_expert_scores(
  logits : ArrayView[Float],
  function : @advanced_decoder_execution_plan.ExpertScoreFunction,
  limits~ : ReferenceLimits,
) -> ReadOnlyArray[Float] raise AdvancedDecoderReferenceError {
  validate_values(logits, limits)
  let transformed : Array[Float] = []
  for value in logits {
    let score = score_one(value, function)
    if !finite(score) {
      raise NonFiniteParameter
    }
    transformed.push(score)
  }
  ReadOnlyArray::from_array(transformed)
}

///|
fn stable_top_k_indices(scores : ArrayView[Float], top_k : Int) -> Array[Int] {
  let selected : Array[Int] = []
  for candidate in 0..<scores.length() {
    let mut insertion = selected.length()
    for position, previous in selected {
      if scores[candidate] > scores[previous] ||
        (scores[candidate] == scores[previous] && candidate < previous) {
        insertion = position
        break
      }
    }
    if insertion < top_k {
      selected.insert(insertion, candidate)
      if selected.length() > top_k {
        ignore(selected.pop())
      }
    } else if selected.length() < top_k {
      selected.push(candidate)
    }
  }
  selected
}

///|
fn selected_weights(
  transformed : ArrayView[Float],
  indices : ArrayView[Int],
  normalize : Bool,
  scaling : Float,
) -> Array[Float] raise AdvancedDecoderReferenceError {
  if !finite(scaling) || scaling <= 0.0F {
    raise NonFiniteParameter
  }
  let mut sum = 0.0F
  if normalize {
    for index in indices {
      sum += transformed[index]
    }
    if !finite(sum) || sum <= 0.0F {
      raise ZeroNormalization
    }
  } else {
    sum = 1.0F
  }
  let weights : Array[Float] = []
  for index in indices {
    let weight = transformed[index] / sum * scaling
    if !finite(weight) {
      raise NonFiniteParameter
    }
    weights.push(weight)
  }
  weights
}

///|
/// Performs stable lower-index-wins expert selection. `selection_bias` is used
/// only for top-k choice; returned weights always derive from unbiased scores.
pub fn route_top_k(
  logits : ArrayView[Float],
  selection_bias : ArrayView[Float],
  top_k~ : Int,
  score_function~ : @advanced_decoder_execution_plan.ExpertScoreFunction,
  normalize~ : Bool,
  scaling~ : Float,
  limits~ : ReferenceLimits,
) -> ExpertRouting raise AdvancedDecoderReferenceError {
  validate_values(logits, limits)
  if top_k <= 0 || top_k > logits.length() {
    raise InvalidTopK
  }
  if !selection_bias.is_empty() {
    validate_values(selection_bias, limits)
    if selection_bias.length() != logits.length() {
      raise InvalidDimension
    }
  }
  let transformed = transform_expert_scores(logits, score_function, limits~)
  let choice_scores : Array[Float] = []
  for index, score in transformed {
    let bias = if selection_bias.is_empty() {
      0.0F
    } else {
      selection_bias[index]
    }
    let biased = score + bias
    if !finite(biased) {
      raise NonFiniteParameter
    }
    choice_scores.push(biased)
  }
  let indices = stable_top_k_indices(choice_scores, top_k)
  let weights = selected_weights(transformed, indices, normalize, scaling)
  {
    expert_indices: ReadOnlyArray::from_array(indices),
    weights: ReadOnlyArray::from_array(weights),
  }
}

///|
/// Validates a complete token-to-expert table, then applies its selected row.
/// Every row must contain distinct in-range expert indices.
pub fn route_token_hash(
  logits : ArrayView[Float],
  table : ArrayView[Int],
  vocabulary_size~ : Int,
  experts_per_token~ : Int,
  token_id~ : Int,
  score_function~ : @advanced_decoder_execution_plan.ExpertScoreFunction,
  normalize~ : Bool,
  scaling~ : Float,
  limits~ : ReferenceLimits,
) -> ExpertRouting raise AdvancedDecoderReferenceError {
  validate_values(logits, limits)
  if vocabulary_size <= 0 ||
    experts_per_token <= 0 ||
    experts_per_token > logits.length() ||
    vocabulary_size > limits.max_values / experts_per_token ||
    table.length() != vocabulary_size * experts_per_token {
    raise InvalidHashTable
  }
  if token_id < 0 || token_id >= vocabulary_size {
    raise TokenOutOfRange
  }
  for token in 0..<vocabulary_size {
    for position in 0..<experts_per_token {
      let index = table[token * experts_per_token + position]
      if index < 0 || index >= logits.length() {
        raise ExpertIndexOutOfRange(token * experts_per_token + position)
      }
      for previous in 0..<position {
        if table[token * experts_per_token + previous] == index {
          raise DuplicateExpertIndex(token * experts_per_token + position)
        }
      }
    }
  }
  let transformed = transform_expert_scores(logits, score_function, limits~)
  let indices : Array[Int] = []
  for position in 0..<experts_per_token {
    indices.push(table[token_id * experts_per_token + position])
  }
  let weights = selected_weights(transformed, indices, normalize, scaling)
  {
    expert_indices: ReadOnlyArray::from_array(indices),
    weights: ReadOnlyArray::from_array(weights),
  }
}

///|
/// Combines precomputed routed-expert outputs in routing order, then adds every
/// precomputed shared-expert output. Inputs are flat row-major Float tensors.
pub fn combine_expert_outputs(
  routing : ExpertRouting,
  routed_outputs : ArrayView[Float],
  shared_outputs : ArrayView[Float],
  width~ : Int,
  limits~ : ReferenceLimits,
) -> ReadOnlyArray[Float] raise AdvancedDecoderReferenceError {
  validate_shape(routing.selected_count(), width, limits)
  if routed_outputs.length() != routing.selected_count() * width ||
    shared_outputs.length() % width != 0 ||
    !combined_values_within_limit(
      routed_outputs.length(),
      shared_outputs.length(),
      0,
      limits,
    ) {
    raise InvalidDimension
  }
  validate_values(routed_outputs, limits)
  if !shared_outputs.is_empty() {
    validate_values(shared_outputs, limits)
  }
  let output = Array::make(width, 0.0F)
  for route in 0..<routing.selected_count() {
    for column in 0..<width {
      output[column] += routing.weights[route] *
        routed_outputs[route * width + column]
    }
  }
  let shared_count = shared_outputs.length() / width
  for shared in 0..<shared_count {
    for column in 0..<width {
      output[column] += shared_outputs[shared * width + column]
    }
  }
  for index, value in output {
    if !finite(value) {
      raise NonFiniteInput(index)
    }
  }
  ReadOnlyArray::from_array(output)
}
