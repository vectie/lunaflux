///|
fn copy_floats(values : ArrayView[Float]) -> ReadOnlyArray[Float] {
  let owned : Array[Float] = []
  for value in values {
    owned.push(value)
  }
  ReadOnlyArray::from_array(owned)
}

///|
fn block_value_count(inputs : ArrayView[ArrayView[Float]]) -> Int64 {
  let mut total = 0L
  for input in inputs {
    total += input.length().to_int64()
  }
  total
}

///|
pub fn TinyAdvancedDecoderBlockInput::new(
  stream_activations~ : ArrayView[Float],
  query_weight~ : ArrayView[Float],
  keys~ : ArrayView[Float],
  values~ : ArrayView[Float],
  output_weight~ : ArrayView[Float],
  mixing_logits~ : ArrayView[Float],
  router_weight~ : ArrayView[Float],
  selection_bias~ : ArrayView[Float],
  expert_up_weight~ : ArrayView[Float],
  expert_gate_weight~ : ArrayView[Float],
  expert_down_weight~ : ArrayView[Float],
  stream_count~ : Int,
  width~ : Int,
  key_count~ : Int,
  index_top_k~ : Int,
  expert_count~ : Int,
  experts_per_token~ : Int,
  expert_hidden_width~ : Int,
  sinkhorn_iterations~ : Int,
  epsilon~ : Float,
  limits~ : ReferenceLimits,
) -> Self raise AdvancedDecoderReferenceError {
  validate_shape(stream_count, width, limits)
  if key_count <= 0 ||
    key_count > limits.max_values / width ||
    index_top_k <= 0 ||
    index_top_k > key_count ||
    expert_count <= 0 ||
    expert_count > limits.max_width ||
    experts_per_token <= 0 ||
    experts_per_token > expert_count ||
    expert_hidden_width <= 0 ||
    expert_hidden_width > limits.max_width ||
    sinkhorn_iterations <= 0 ||
    sinkhorn_iterations > limits.max_iterations ||
    !finite(epsilon) ||
    epsilon <= 0.0F {
    raise InvalidBlock
  }
  let expert_up_count = expert_count * expert_hidden_width * width
  let expert_down_count = expert_count * width * expert_hidden_width
  if stream_activations.length() != stream_count * width ||
    query_weight.length() != width * width ||
    keys.length() != key_count * width ||
    values.length() != key_count * width ||
    output_weight.length() != width * width ||
    mixing_logits.length() != stream_count * stream_count ||
    router_weight.length() != expert_count * width ||
    (!selection_bias.is_empty() && selection_bias.length() != expert_count) ||
    expert_up_weight.length() != expert_up_count ||
    expert_gate_weight.length() != expert_up_count ||
    expert_down_weight.length() != expert_down_count ||
    block_value_count([
      stream_activations, query_weight, keys, values, output_weight, mixing_logits,
      router_weight, selection_bias, expert_up_weight, expert_gate_weight, expert_down_weight,
    ]) >
    limits.max_values.to_int64() {
    raise InvalidBlock
  }
  for
    input in [
      stream_activations, query_weight, keys, values, output_weight, mixing_logits,
      router_weight, expert_up_weight, expert_gate_weight, expert_down_weight,
    ] {
    validate_values(input, limits)
  }
  if !selection_bias.is_empty() {
    validate_values(selection_bias, limits)
  }
  {
    stream_activations: copy_floats(stream_activations),
    query_weight: copy_floats(query_weight),
    keys: copy_floats(keys),
    values: copy_floats(values),
    output_weight: copy_floats(output_weight),
    mixing_logits: copy_floats(mixing_logits),
    router_weight: copy_floats(router_weight),
    selection_bias: copy_floats(selection_bias),
    expert_up_weight: copy_floats(expert_up_weight),
    expert_gate_weight: copy_floats(expert_gate_weight),
    expert_down_weight: copy_floats(expert_down_weight),
    stream_count,
    width,
    index_top_k,
    expert_count,
    experts_per_token,
    expert_hidden_width,
    sinkhorn_iterations,
    epsilon,
  }
}

///|
fn matrix_vector(
  input : ArrayView[Float],
  weight : ArrayView[Float],
  rows : Int,
  columns : Int,
) -> Array[Float] {
  let output = Array::make(rows, 0.0F)
  for row in 0..<rows {
    for column in 0..<columns {
      output[row] += weight[row * columns + column] * input[column]
    }
  }
  output
}

///|
fn normalize_rms(
  input : ArrayView[Float],
  epsilon : Float,
) -> Array[Float] raise AdvancedDecoderReferenceError {
  let mut squares = 0.0F
  for value in input {
    squares += value * value
  }
  let denominator = (squares / Float::from_int(input.length()) + epsilon).sqrt()
  if !finite(denominator) || denominator <= 0.0F {
    raise ZeroNormalization
  }
  let output : Array[Float] = []
  for value in input {
    output.push(value / denominator)
  }
  output
}

///|
fn selected_attention(
  query : ArrayView[Float],
  keys : ArrayView[Float],
  values : ArrayView[Float],
  selected : ArrayView[Int],
  width : Int,
) -> (Array[Float], Array[Float]) raise AdvancedDecoderReferenceError {
  let logits : Array[Float] = []
  let scale = Float::from_int(width).sqrt()
  for key in selected {
    let mut score = 0.0F
    for column in 0..<width {
      score += query[column] * keys[key * width + column]
    }
    logits.push(score / scale)
  }
  let mut maximum = logits[0]
  for logit in logits {
    if logit > maximum {
      maximum = logit
    }
  }
  let weights : Array[Float] = []
  let mut total = 0.0F
  for logit in logits {
    let value = @math.expf(logit - maximum)
    weights.push(value)
    total += value
  }
  if !finite(total) || total <= 0.0F {
    raise ZeroNormalization
  }
  let context = Array::make(width, 0.0F)
  for position, key in selected {
    weights[position] /= total
    for column in 0..<width {
      context[column] += weights[position] * values[key * width + column]
    }
  }
  (weights, context)
}

///|
fn expert_swiglu(
  hidden : ArrayView[Float],
  expert : Int,
  input : TinyAdvancedDecoderBlockInput,
) -> Array[Float] {
  let up_base = expert * input.expert_hidden_width * input.width
  let down_base = expert * input.width * input.expert_hidden_width
  let activated = Array::make(input.expert_hidden_width, 0.0F)
  for row in 0..<input.expert_hidden_width {
    let mut up = 0.0F
    let mut gate = 0.0F
    for column in 0..<input.width {
      let index = up_base + row * input.width + column
      up += input.expert_up_weight[index] * hidden[column]
      gate += input.expert_gate_weight[index] * hidden[column]
    }
    let sigmoid = if gate >= 0.0F {
      1.0F / (1.0F + @math.expf(-gate))
    } else {
      let exponential = @math.expf(gate)
      exponential / (1.0F + exponential)
    }
    activated[row] = gate * sigmoid * up
  }
  let output = Array::make(input.width, 0.0F)
  for row in 0..<input.width {
    for column in 0..<input.expert_hidden_width {
      output[row] += input.expert_down_weight[down_base +
        row * input.expert_hidden_width +
        column] *
        activated[column]
    }
  }
  output
}

///|
/// Execute one deterministic correctness-only advanced decoder block. The
/// stage order is RMS norm, learned sparse index, attention, output projection,
/// hyper-connection mixing, stable routing, selected SwiGLU experts, residual.
pub fn run_tiny_advanced_decoder_block(
  input : TinyAdvancedDecoderBlockInput,
  limits~ : ReferenceLimits,
) -> TinyAdvancedDecoderBlockOutput raise AdvancedDecoderReferenceError {
  let main : Array[Float] = []
  for column in 0..<input.width {
    main.push(input.stream_activations[column])
  }
  let normalized = normalize_rms(main, input.epsilon)
  let query = matrix_vector(
    normalized,
    input.query_weight,
    input.width,
    input.width,
  )
  let selected = learned_index_top_k(
    query,
    input.keys,
    [1.0F],
    head_count=1,
    head_dimension=input.width,
    top_k=input.index_top_k,
    limits~,
  )
  let (attention_weights, context) = selected_attention(
    query,
    input.keys,
    input.values,
    selected.indices,
    input.width,
  )
  let attention = matrix_vector(
    context,
    input.output_weight,
    input.width,
    input.width,
  )
  let residual_streams : Array[Float] = []
  for stream in 0..<input.stream_count {
    for column in 0..<input.width {
      let residual = if stream == 0 { attention[column] } else { 0.0F }
      residual_streams.push(
        input.stream_activations[stream * input.width + column] + residual,
      )
    }
  }
  let mixing = sinkhorn_normalize(
    input.mixing_logits,
    width=input.stream_count,
    iterations=input.sinkhorn_iterations,
    epsilon=input.epsilon,
    limits~,
  )
  let mixed = mix_hyper_connections(
    residual_streams,
    mixing,
    stream_count=input.stream_count,
    feature_width=input.width,
    limits~,
  )
  let mixed_main : Array[Float] = []
  for column in 0..<input.width {
    mixed_main.push(mixed[column])
  }
  let router_logits = matrix_vector(
    mixed_main,
    input.router_weight,
    input.expert_count,
    input.width,
  )
  let routing = route_top_k(
    router_logits,
    input.selection_bias,
    top_k=input.experts_per_token,
    score_function=Sigmoid,
    normalize=true,
    scaling=1.0F,
    limits~,
  )
  let routed : Array[Float] = []
  for expert in routing.expert_indices {
    let output = expert_swiglu(mixed_main, expert, input)
    for value in output {
      routed.push(value)
    }
  }
  let expert_output = combine_expert_outputs(
    routing,
    routed,
    [],
    width=input.width,
    limits~,
  )
  let hidden : Array[Float] = []
  for column in 0..<input.width {
    let value = mixed_main[column] + expert_output[column]
    if !finite(value) {
      raise NonFiniteParameter
    }
    hidden.push(value)
  }
  {
    selected_key_indices: selected.indices,
    attention_weights: ReadOnlyArray::from_array(attention_weights),
    routing,
    mixed_streams: mixed,
    hidden: ReadOnlyArray::from_array(hidden),
  }
}
