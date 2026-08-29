///|
/// Semantic scale grouping required by one stored tensor.
#valtype
pub struct ScaleGranularity {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
pub fn ScaleGranularity::absent() -> ScaleGranularity {
  { value: 0 }
}

///|
pub fn ScaleGranularity::per_tensor() -> ScaleGranularity {
  { value: 1 }
}

///|
pub fn ScaleGranularity::per_output_channel() -> ScaleGranularity {
  { value: 2 }
}

///|
/// Zero-point representation required by one stored tensor.
#valtype
pub struct ZeroPointMode {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
pub fn ZeroPointMode::absent() -> ZeroPointMode {
  { value: 0 }
}

///|
pub fn ZeroPointMode::implicit_symmetric_zero_v1() -> ZeroPointMode {
  { value: 1 }
}

///|
pub fn ZeroPointMode::per_output_channel_tensor_v1() -> ZeroPointMode {
  { value: 2 }
}

///|
/// Codebook geometry required by one stored tensor.
#valtype
pub struct CodebookLayout {
  priv value : Int
} derive(Debug, Eq, Hash)

///|
pub fn CodebookLayout::absent() -> CodebookLayout {
  { value: 0 }
}

///|
pub fn CodebookLayout::contiguous_f32_values_v1() -> CodebookLayout {
  { value: 1 }
}
