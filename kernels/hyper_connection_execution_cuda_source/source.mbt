///|
pub struct HyperConnectionCudaSource {
  priv plan : @precision.HyperConnectionPrecision
  priv prefix : String
  priv source : Bytes
  priv normalization_threads : Int
  priv control_schedule : @physical.ControlSchedule?
  priv post_square : Bool
}

///|
/// Preserve staged numerical rounds while composing a complete branch envelope.
/// Device scheduling and source-level helper names stay exclusively here.
pub fn hyper_connection_source(
  plan : @precision.HyperConnectionPrecision,
  symbol_prefix~ : String,
  parallel_controls? : Bool = true,
) -> HyperConnectionCudaSource raise {
  let (streams, hidden, rows) = plan.geometry()
  // Independent physical scheduling does not change the numerical IR. Keep
  // other arithmetic laws on their explicit existing implementations.
  let control_schedule : @physical.ControlSchedule? = if parallel_controls &&
    plan.control_law() is NormalizedBf16Projection &&
    plan.control_square_law() is OrderedFloat32 &&
    plan.control_dot_law() is OrderedFloat32 {
    Some(
      Ordered(
        @physical.OrderedControlPlan::new(plan, workers=256, product_tile=256),
      ),
    )
  } else if parallel_controls &&
    plan.control_law() is NormalizedBf16Projection &&
    plan.control_square_law() is VectorizedTreeFloat32(..) &&
    plan.control_dot_law() is StripedTreeFloat32(..) {
    Some(Tree(@physical.ParallelControlPlan::new(plan)))
  } else {
    None
  }
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
  let function_source = if control_schedule is Some(Ordered(schedule)) {
    @control.ordered_control_tiles_source(plan, schedule, symbol_prefix~)
  } else if control_schedule is Some(Tree(schedule)) {
    @control.parallel_control_tiles_source(plan, schedule, symbol_prefix~)
  } else {
    @utf8.encode(
      @utf8.decode(function.source_bytes())
      .replace_all(old="lf_hc_stage_sigmoid", new=symbol_prefix + "_sigmoid")
      .replace_all(old="lf_hc_stage_sum", new=symbol_prefix + "_matrix_sum"),
    )
  }
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
  let post_square = plan.mix_law() is RoundedBf16ControlsAndProducts &&
    control_schedule is Some(Tree(_))
  let fused = if post_square {
    @control.post_square_source(
      plan,
      plan,
      symbol=symbol_prefix + "_post_square",
    )
  } else {
    b""
  }
  let fused_tail = if control_schedule is Some(Tree(_)) {
    @control.fused_control_tail_source(
      plan,
      symbol_prefix=symbol_prefix + "_tail",
    )
  } else {
    b""
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
    post +
    b"\n" +
    fused +
    b"\n" +
    fused_tail,
    normalization_threads: normalization.launch_threads(),
    control_schedule,
    post_square,
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
pub fn HyperConnectionCudaSource::fuses_normalization(self : Self) -> Bool {
  match self.control_schedule {
    Some(schedule) => schedule.fuses_normalization()
    None => false
  }
}

///|
pub fn HyperConnectionCudaSource::control_schedule(
  self : Self,
) -> @physical.ControlSchedule? {
  self.control_schedule
}

///|
pub fn HyperConnectionCudaSource::supports_post_square(self : Self) -> Bool {
  self.post_square
}

///|
pub fn HyperConnectionCudaSource::symbol(
  self : Self,
  stage : Int,
) -> String raise @precision.PrecisionPlanError {
  let suffix = match stage {
    0 =>
      if self.control_schedule is Some(Tree(_)) {
        "function_tree_tiles_v1"
      } else if self.control_schedule is Some(_) {
        "function_ordered_tiles_v1"
      } else {
        "function"
      }
    1 => "sinkhorn"
    2 => "norm"
    3 => "post"
    4 if self.control_schedule is Some(Tree(_)) => "square_tree_tiles_v1"
    5 if self.control_schedule is Some(Tree(_)) => "project_tree_tiles_v1"
    4 if self.control_schedule is Some(_) => "square_ordered_tiles_v1"
    5 if self.control_schedule is Some(_) => "project_ordered_tiles_v1"
    6 if self.post_square => "post_square"
    7 if self.control_schedule is Some(Tree(_)) => "tail_function_tree_tiles_v1"
    _ => raise @precision.PrecisionPlanError::IllegalPlacement
  }
  self.prefix + "_" + suffix
}
