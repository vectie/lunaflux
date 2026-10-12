///|
pub struct HyperConnectionCudaSource {
  priv plan : @precision.HyperConnectionPrecision
  priv prefix : String
  priv source : Bytes
  priv normalization_threads : Int
}

///|
/// Preserve staged numerical rounds while composing a complete branch envelope.
/// Device scheduling and source-level helper names stay exclusively here.
pub fn hyper_connection_source(
  plan : @precision.HyperConnectionPrecision,
  symbol_prefix~ : String,
) -> HyperConnectionCudaSource raise {
  let (streams, hidden, rows) = plan.geometry()
  let function = @control.staged_function_and_collapse_bf16_f32_source(
    stream_width=streams,
    hidden_width=hidden,
    maximum_rows=rows,
    control_epsilon=plan.control_epsilon(),
    normalization_epsilon=plan.normalization_epsilon(),
    function_symbol=symbol_prefix + "_function",
    control_law=plan.control_law(),
    control_square_law=plan.control_square_law(),
    control_dot_law=plan.control_dot_law(),
    control_vector_affine_law=plan.control_vector_affine_law(),
    control_sigmoid_law=plan.control_sigmoid_law(),
    control_matrix_affine_law=plan.control_matrix_affine_law(),
    matrix_sum_law=plan.matrix_sum_law(),
    control_root_law=plan.control_root_law(),
  )
  // Multiple branch envelopes may share a module without duplicate helpers.
  let function_source = @utf8.encode(
    @utf8.decode(function.source_bytes())
    .replace_all(old="lf_hc_stage_sigmoid", new=symbol_prefix + "_sigmoid")
    .replace_all(old="lf_hc_stage_sum", new=symbol_prefix + "_matrix_sum"),
  )
  let sinkhorn = @control.staged_positive_sinkhorn_f32_source(
    stream_width=streams,
    maximum_rows=rows,
    sinkhorn_iterations=plan.sinkhorn_iterations(),
    control_epsilon=plan.control_epsilon(),
    function_symbol=symbol_prefix + "_sinkhorn",
    sum_law=plan.matrix_sum_law(),
  ).source_bytes()
  let sinkhorn = @utf8.encode(
    @utf8.decode(sinkhorn).replace_all(
      old="lf_hc_sinkhorn_sum",
      new=symbol_prefix + "_sinkhorn_sum",
    ),
  )
  let normalization = @norm.prefix_rms_norm_bf16_with_optional_suffix_copy_source(
    prefix_width=hidden,
    suffix_width=0,
    maximum_rows=rows,
    epsilon=plan.normalization_epsilon(),
    function_symbol=symbol_prefix + "_norm",
    square_law=plan.normalization_square_law(),
  )
  let post = match plan.mix_law() {
    RoundedBf16ControlsAndProducts =>
      @mix.hyper_connection_mix_bf16_source(
        hidden_width=hidden,
        stream_count=streams,
        maximum_rows=rows,
        function_symbol=symbol_prefix + "_post",
      ).source_bytes()
    OrderedF32SingleBf16Round | TransposedF32ResidualThenBranch =>
      @control.post_combine_v1_bf16_source(
        stream_width=streams,
        hidden_width=hidden,
        maximum_rows=rows,
        function_symbol=symbol_prefix + "_post",
        mix_law=plan.mix_law(),
      ).source_bytes()
  }
  {
    plan,
    prefix: symbol_prefix,
    source: function_source +
    b"\n" +
    sinkhorn +
    b"\n" +
    normalization.source_bytes() +
    b"\n" +
    post,
    normalization_threads: normalization.launch_threads(),
  }
}

///|
pub fn HyperConnectionCudaSource::plan(
  self : Self,
) -> @precision.HyperConnectionPrecision {
  self.plan
}

///|
pub fn HyperConnectionCudaSource::source_bytes(self : Self) -> Bytes {
  self.source
}

///|
pub fn HyperConnectionCudaSource::normalization_threads(self : Self) -> Int {
  self.normalization_threads
}

///|
pub fn HyperConnectionCudaSource::symbol(
  self : Self,
  stage : Int,
) -> String raise @precision.PrecisionPlanError {
  let suffix = match stage {
    0 => "function"
    1 => "sinkhorn"
    2 => "norm"
    3 => "post"
    _ => raise @precision.PrecisionPlanError::IllegalPlacement
  }
  self.prefix + "_" + suffix
}
