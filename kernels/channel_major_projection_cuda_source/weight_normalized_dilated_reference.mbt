///|
pub fn f32_weight_normalized_dilated_conv1d_reference(
  batch_count~ : Int,
  channel_count~ : Int,
  time_count~ : Int,
  kernel_size~ : Int,
  dilation~ : Int,
  padding~ : Int,
  input : ArrayView[Float],
  weight_g : ArrayView[Float],
  weight_v : ArrayView[Float],
  bias : ArrayView[Float],
) -> ReadOnlyArray[Float] raise ChannelMajorProjectionCudaSourceError {
  if batch_count <= 0 ||
    channel_count <= 0 ||
    time_count <= 0 ||
    kernel_size <= 1 ||
    dilation <= 0 ||
    padding < 0 {
    raise InvalidTimeCount
  }
  let receptive = checked_mul(dilation.to_int64(), (kernel_size - 1).to_int64())
  let padded = checked_mul(padding.to_int64(), 2L)
  if receptive != padded {
    raise InvalidTimeCount
  }
  let elements = checked_mul(
    checked_mul(batch_count.to_int64(), channel_count.to_int64()),
    time_count.to_int64(),
  )
  let per_output = checked_mul(channel_count.to_int64(), kernel_size.to_int64())
  let weights = checked_mul(channel_count.to_int64(), per_output)
  if elements > 2147483647L ||
    weights > 2147483647L ||
    input.length() != elements.to_int() ||
    weight_g.length() != channel_count ||
    bias.length() != channel_count ||
    weight_v.length() != weights.to_int() {
    raise InvalidChannelCount
  }
  let normalized : Array[Float] = Array::make(weights.to_int(), 0.0F)
  for output_channel in 0..<channel_count {
    let base = output_channel * channel_count * kernel_size
    let mut sum_squares = 0.0F
    for input_channel in 0..<channel_count {
      for kernel_index in 0..<kernel_size {
        let value = weight_v[base + input_channel * kernel_size + kernel_index]
        sum_squares = sum_squares + value * value
      }
    }
    if sum_squares == 0.0F {
      raise InvalidChannelCount
    }
    let scale = weight_g[output_channel] / sum_squares.sqrt()
    for offset in 0..<per_output.to_int() {
      normalized[base + offset] = weight_v[base + offset] * scale
    }
  }
  let output : Array[Float] = Array::make(elements.to_int(), 0.0F)
  for batch in 0..<batch_count {
    for output_channel in 0..<channel_count {
      for time in 0..<time_count {
        let mut sum = bias[output_channel]
        for input_channel in 0..<channel_count {
          for kernel_index in 0..<kernel_size {
            let input_time = time + kernel_index * dilation - padding
            if input_time >= 0 && input_time < time_count {
              let input_index = (batch * channel_count + input_channel) *
                time_count +
                input_time
              let weight_index = (
                  output_channel * channel_count + input_channel
                ) *
                kernel_size +
                kernel_index
              sum = sum + input[input_index] * normalized[weight_index]
            }
          }
        }
        output[(batch * channel_count + output_channel) * time_count + time] = sum
      }
    }
  }
  ReadOnlyArray::from_array(output)
}
