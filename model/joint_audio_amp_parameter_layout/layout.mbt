///|
pub fn JointAudioAmpParameterRegionInput::new(
  role~ : JointAudioAmpParameterRole,
  tensor_name~ : String,
  element_count~ : Int64,
  byte_offset~ : Int64,
) -> Self raise JointAudioAmpParameterLayoutError {
  if tensor_name.is_empty() ||
    tensor_name.length() > 256 ||
    element_count <= 0L ||
    byte_offset < 0L ||
    byte_offset % 64L != 0L {
    raise InvalidRegion(0)
  }
  { role, tensor_name, element_count, byte_offset, }
}

///|
fn expected_role(index : Int) -> JointAudioAmpParameterRole {
  let block = index / 42
  let within = index % 42
  if within < 24 {
    let activation = within / 4
    match within % 4 {
      0 => ActivationAlpha(block, activation)
      1 => ActivationBeta(block, activation)
      2 => ActivationDownsampleFilter(block, activation)
      _ => ActivationUpsampleFilter(block, activation)
    }
  } else {
    let convolution = (within - 24) / 3
    let pair = convolution / 2
    let side = convolution % 2
    match (within - 24) % 3 {
      0 => ConvolutionBias(block, pair, side)
      1 => ConvolutionWeightG(block, pair, side)
      _ => ConvolutionWeightV(block, pair, side)
    }
  }
}

///|
pub fn JointAudioAmpParameterLayout::new(
  inputs : ArrayView[JointAudioAmpParameterRegionInput],
) -> Self raise JointAudioAmpParameterLayoutError {
  if inputs.length() != 126 {
    raise InvalidRegionCount
  }
  let regions : Array[JointAudioAmpParameterRegion] = []
  let canonical = StringBuilder()
  canonical.write_string(
    "lunaflux.joint-audio-amp-parameter-layout.v1|dtype=f32|alignment=64",
  )
  let mut previous_end = 0L
  for index, input in inputs {
    if input.role != expected_role(index) {
      raise InvalidRegionOrder(index)
    }
    let bytes = if input.element_count > 9223372036854775807L / 4L {
      raise SizeOverflow
    } else {
      input.element_count * 4L
    }
    if input.byte_offset < previous_end ||
      (index > 0 && input.byte_offset - previous_end >= 64L) {
      raise InvalidRegionOffset(index)
    }
    for previous in regions {
      if previous.tensor_name == input.tensor_name {
        raise DuplicateRegion(index)
      }
    }
    if input.byte_offset > 9223372036854775807L - bytes {
      raise SizeOverflow
    }
    previous_end = input.byte_offset + bytes
    regions.push({
      role: input.role,
      tensor_name: input.tensor_name,
      element_count: input.element_count,
      byte_offset: input.byte_offset,
      byte_count: bytes,
    })
    canonical <+
      "|\{index}:\{input.tensor_name}:\{input.element_count}:\{input.byte_offset}:\{bytes}"
  }
  {
    regions: ReadOnlyArray::from_array(regions),
    total_arena_bytes: previous_end,
    digest: {
      hex: @crypto.bytes_to_hex_string(
        @crypto.sha256(@utf8.encode(canonical.to_string())),
      ),
    },
  }
}
