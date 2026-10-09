///|
/// Scalar reference for the depthwise causal short convolution used by each
/// projected Q/K/V channel. State stores the preceding `kernel_size - 1`
/// values per channel, oldest first. The output applies SiLU as GLM Flash does.
pub fn kda_causal_short_convolution(
  input : ArrayView[Double],
  weights : ArrayView[Double],
  initial_state : ArrayView[Double],
  sequence_length~ : Int,
  channel_count~ : Int,
  kernel_size~ : Int,
  limits~ : GlmHybridReferenceLimits,
) -> GlmKdaConvolutionResult raise GlmHybridReferenceError {
  limits.check_sequence(sequence_length)
  limits.check_channels(channel_count)
  if kernel_size <= 0 || kernel_size > sequence_length + 1 {
    raise InvalidShape
  }
  let history = kernel_size - 1
  if input.length() != sequence_length * channel_count ||
    weights.length() != channel_count * kernel_size ||
    initial_state.length() != channel_count * history {
    raise InvalidShape
  }
  require_finite(input)
  require_finite(weights)
  require_finite(initial_state)
  let output = Array::make(sequence_length * channel_count, 0.0)
  for token in 0..<sequence_length {
    for channel in 0..<channel_count {
      let mut sum = 0.0
      for offset in 0..<kernel_size {
        let source = token - history + offset
        let value = if source < 0 {
          initial_state[channel * history + history + source]
        } else {
          input[source * channel_count + channel]
        }
        sum += value * weights[channel * kernel_size + offset]
      }
      output[token * channel_count + channel] = sum / (1.0 + @math.exp(-sum))
    }
  }
  let final_state = Array::make(channel_count * history, 0.0)
  for channel in 0..<channel_count {
    for offset in 0..<history {
      let combined_position = sequence_length + offset
      let source = combined_position - history
      final_state[channel * history + offset] = if source < 0 {
        initial_state[channel * history + history + source]
      } else {
        input[source * channel_count + channel]
      }
    }
  }
  {
    sequence_length,
    channel_count,
    output: ReadOnlyArray::from_array(output),
    final_state: ReadOnlyArray::from_array(final_state),
  }
}
