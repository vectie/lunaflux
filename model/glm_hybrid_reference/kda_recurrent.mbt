///|
fn normalized_component(
  values : ArrayView[Double],
  base : Int,
  component : Int,
  dimension : Int,
) -> Double {
  let mut squares = 0.0
  for offset in 0..<dimension {
    let value = values[base + offset]
    squares += value * value
  }
  values[base + component] / (squares + 0.000001).sqrt()
}

///|
/// Exact scalar recurrence of GLM-5.3-Flash Kimi Delta Attention. Q/K are L2
/// normalized in FP64 reference arithmetic, Q is scaled by `1/sqrt(D)`, state
/// decays by per-key-component `exp(g[token, head, key])`, and the delta update
/// removes the current memory read.
pub fn kda_recurrent_delta_update(
  query : ArrayView[Double],
  key : ArrayView[Double],
  value : ArrayView[Double],
  log_decay : ArrayView[Double],
  beta : ArrayView[Double],
  initial_state : ArrayView[Double],
  sequence_length~ : Int,
  head_count~ : Int,
  head_dimension~ : Int,
  limits~ : GlmHybridReferenceLimits,
) -> GlmKdaRecurrentResult raise GlmHybridReferenceError {
  limits.check_sequence(sequence_length)
  limits.check_heads(head_count, head_dimension)
  let vectors = sequence_length * head_count * head_dimension
  let gates = sequence_length * head_count
  let state_elements = head_count * head_dimension * head_dimension
  if query.length() != vectors ||
    key.length() != vectors ||
    value.length() != vectors ||
    log_decay.length() != vectors ||
    beta.length() != gates ||
    initial_state.length() != state_elements {
    raise InvalidShape
  }
  require_finite(query)
  require_finite(key)
  require_finite(value)
  require_finite(log_decay)
  require_finite(beta)
  require_finite(initial_state)
  for decay in log_decay {
    if decay < -5.0 || decay > 0.0 {
      raise InvalidProbability
    }
  }
  for gate in beta {
    if gate < 0.0 || gate > 1.0 {
      raise InvalidProbability
    }
  }
  let state = [ for value in initial_state => value ]
  let output = Array::make(vectors, 0.0)
  let query_scale = 1.0 / head_dimension.to_double().sqrt()
  for token in 0..<sequence_length {
    for head in 0..<head_count {
      let vector_base = (token * head_count + head) * head_dimension
      let gate_index = token * head_count + head
      let state_base = head * head_dimension * head_dimension
      for key_component in 0..<head_dimension {
        let decay = @math.exp(log_decay[vector_base + key_component])
        for output_component in 0..<head_dimension {
          let index = state_base +
            key_component * head_dimension +
            output_component
          state[index] = state[index] * decay
        }
      }
      let memory = Array::make(head_dimension, 0.0)
      for output_component in 0..<head_dimension {
        for key_component in 0..<head_dimension {
          memory[output_component] += state[state_base +
            key_component * head_dimension +
            output_component] *
            normalized_component(
              key, vector_base, key_component, head_dimension,
            )
        }
      }
      for output_component in 0..<head_dimension {
        let delta = (
            value[vector_base + output_component] - memory[output_component]
          ) *
          beta[gate_index]
        for key_component in 0..<head_dimension {
          state[state_base + key_component * head_dimension + output_component] = state[state_base +
            key_component * head_dimension +
            output_component] +
            normalized_component(
              key, vector_base, key_component, head_dimension,
            ) *
            delta
        }
      }
      for output_component in 0..<head_dimension {
        let mut projected = 0.0
        for query_component in 0..<head_dimension {
          projected += state[state_base +
            query_component * head_dimension +
            output_component] *
            normalized_component(
              query, vector_base, query_component, head_dimension,
            ) *
            query_scale
        }
        output[vector_base + output_component] = projected
      }
    }
  }
  {
    sequence_length,
    head_count,
    head_dimension,
    output: ReadOnlyArray::from_array(output),
    final_state: ReadOnlyArray::from_array(state),
  }
}
