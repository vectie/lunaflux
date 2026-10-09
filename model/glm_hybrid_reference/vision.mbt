///|
/// Applies the Flash vision Conv3D patch kernel to already partitioned
/// non-overlapping patches. Each row is one flattened `[C,T,H,W]` patch.
pub fn vision_patch_projection(
  patches : ArrayView[Double],
  weights : ArrayView[Double],
  bias : ArrayView[Double],
  patch_count~ : Int,
  patch_volume~ : Int,
  output_width~ : Int,
  limits~ : GlmHybridReferenceLimits,
) -> GlmVisionReferenceResult raise GlmHybridReferenceError {
  limits.check_vision_patches(patch_count)
  limits.check_channels(patch_volume)
  limits.check_channels(output_width)
  if patches.length() != patch_count * patch_volume ||
    weights.length() != output_width * patch_volume ||
    bias.length() != output_width {
    raise InvalidVisionGeometry
  }
  require_finite(patches)
  require_finite(weights)
  require_finite(bias)
  let output = Array::make(patch_count * output_width, 0.0)
  for patch in 0..<patch_count {
    for feature in 0..<output_width {
      let mut sum = bias[feature]
      for component in 0..<patch_volume {
        sum += patches[patch * patch_volume + component] *
          weights[feature * patch_volume + component]
      }
      output[patch * output_width + feature] = sum
    }
  }
  {
    row_count: patch_count,
    width: output_width,
    values: ReadOnlyArray::from_array(output),
  }
}

///|
/// Applies the stride-equals-kernel Conv2D downsample over contiguous
/// `merge_size × merge_size` patch groups in the official flattened order.
pub fn vision_spatial_merge(
  patch_features : ArrayView[Double],
  weights : ArrayView[Double],
  bias : ArrayView[Double],
  patch_count~ : Int,
  input_width~ : Int,
  output_width~ : Int,
  merge_size~ : Int,
  limits~ : GlmHybridReferenceLimits,
) -> GlmVisionReferenceResult raise GlmHybridReferenceError {
  limits.check_vision_patches(patch_count)
  limits.check_channels(input_width)
  limits.check_channels(output_width)
  if merge_size <= 0 {
    raise InvalidVisionGeometry
  }
  let area = merge_size * merge_size
  if patch_count % area != 0 ||
    patch_features.length() != patch_count * input_width ||
    weights.length() != output_width * input_width * area ||
    bias.length() != output_width {
    raise InvalidVisionGeometry
  }
  require_finite(patch_features)
  require_finite(weights)
  require_finite(bias)
  let row_count = patch_count / area
  let output = Array::make(row_count * output_width, 0.0)
  for row in 0..<row_count {
    for output_component in 0..<output_width {
      let mut sum = bias[output_component]
      for input_component in 0..<input_width {
        for spatial in 0..<area {
          sum += patch_features[(row * area + spatial) * input_width +
            input_component] *
            weights[(output_component * input_width + input_component) * area +
            spatial]
        }
      }
      output[row * output_width + output_component] = sum
    }
  }
  { row_count, width: output_width, values: ReadOnlyArray::from_array(output), }
}

///|
/// Bounded row-wise linear projection used to referee the vision merger's
/// `proj`, gate/up, and down matrices independently of nonlinear activation.
pub fn vision_project_rows(
  input : GlmVisionReferenceResult,
  weights : ArrayView[Double],
  bias : ArrayView[Double],
  output_width~ : Int,
  limits~ : GlmHybridReferenceLimits,
) -> GlmVisionReferenceResult raise GlmHybridReferenceError {
  limits.check_vision_patches(input.row_count)
  limits.check_channels(input.width)
  limits.check_channels(output_width)
  if weights.length() != output_width * input.width ||
    bias.length() != output_width {
    raise InvalidVisionGeometry
  }
  require_finite(weights)
  require_finite(bias)
  let output = Array::make(input.row_count * output_width, 0.0)
  for row in 0..<input.row_count {
    for output_component in 0..<output_width {
      let mut sum = bias[output_component]
      for input_component in 0..<input.width {
        sum += input.values[row * input.width + input_component] *
          weights[output_component * input.width + input_component]
      }
      output[row * output_width + output_component] = sum
    }
  }
  {
    row_count: input.row_count,
    width: output_width,
    values: ReadOnlyArray::from_array(output),
  }
}
