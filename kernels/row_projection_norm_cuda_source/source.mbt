///|
/// Two AOT stages preserve the explicit BF16 projection boundary. Geometry
/// and CUDA details are confined to lowering; no request-time source generation.
pub struct RowProjectionNormCudaSource {
  priv plan : @precision.RowProjectionNormPrecision
  priv source : Bytes
  priv symbols : ReadOnlyArray[String]
}

///|
pub fn row_projection_norm_source(
  plan : @precision.RowProjectionNormPrecision,
  symbol_prefix~ : String,
) -> RowProjectionNormCudaSource raise {
  guard plan.activation_quantization().amax_floor() == 0.0001 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let (input, output, rows) = plan.geometry()
  let symbols = [symbol_prefix + "_projection", symbol_prefix + "_norm"]
  let projection = @projection.block_fp8_ue8m0_projection_cuda_source(
    input_width=input,
    output_width=output,
    maximum_rows=rows,
    function_symbol=symbols[0],
  )
  let norm = @norm.prefix_rms_norm_bf16_with_optional_suffix_copy_source(
    prefix_width=output,
    suffix_width=0,
    maximum_rows=rows,
    epsilon=plan.normalization_epsilon(),
    function_symbol=symbols[1],
  )
  {
    plan,
    source: projection.source_bytes() + b"\n" + norm.source_bytes(),
    symbols: ReadOnlyArray::from_array(symbols),
  }
}

///|
pub fn RowProjectionNormCudaSource::plan(
  self : Self,
) -> @precision.RowProjectionNormPrecision {
  self.plan
}

///|
pub fn RowProjectionNormCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn RowProjectionNormCudaSource::symbol(self : Self, stage : Int) -> String {
  self.symbols[stage]
}

///|
pub fn RowProjectionNormCudaSource::launch(
  self : Self,
  stage : Int,
) -> (Int, Int, Int) {
  let (_, width, rows) = self.plan.geometry()
  (rows, if stage == 0 { (width + 255) / 256 } else { 1 }, 256)
}
