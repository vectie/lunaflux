///|
pub(all) enum NormSite {
  Decoder
  FullQuery
  FullKey
  RecurrentOutput
}

///|
/// Model-owned operator binding. Generic numeric plans contain no model names,
/// while this builder selects the selected checkpoint's layouts and arithmetic.
/// This is a decoder subset, not an assembled executable forward pass.
pub struct DecoderOperators {
  priv spec : @model.Spec
  priv maximum_rows : Int32
  priv decoder_norm : @precision.RmsNormPrecision
  priv query_norm : @precision.RmsNormPrecision
  priv key_norm : @precision.RmsNormPrecision
  priv recurrent_norm : @precision.RmsNormPrecision
  priv router : @precision.SoftmaxTopKPrecision
  priv control : @precision.GatedDeltaControlPrecision
  priv delta : @precision.RecurrentDeltaPrecision
  priv convolution : @precision.CausalConvolutionPrecision
  priv routed_experts : @precision.ExpertMlpPrecision
  priv shared_mlp : @precision.DenseSwiGluPrecision
  priv shared_combine : @precision.SharedExpertPrecision
  priv router_projection : @matrix.MatrixTilePlan
  priv shared_gate_projection : @matrix.MatrixTilePlan
  priv query_rotary : @precision.MultiAxisRotaryPrecision
  priv key_rotary : @precision.MultiAxisRotaryPrecision
  priv full_output_gate : @precision.PointwiseGatePrecision
}

///|
/// Shared immutable geometry used by enclosing decoder/state plan builders.
pub fn DecoderOperators::geometry(self : Self) -> @hybrid.Geometry {
  self.spec.decoder()
}

///|
pub fn DecoderOperators::new(
  spec : @model.Spec,
  maximum_rows~ : Int32,
  maximum_sequences~ : Int32 = 1,
  cache_slots~ : Int32 = maximum_sequences,
) -> DecoderOperators raise {
  let geometry = spec.decoder()
  let full = geometry.full_attention()
  let linear = geometry.recurrent_attention()
  let experts = geometry.experts()
  let channels = 2 * linear.key_heads * linear.key_width +
    linear.value_heads * linear.value_width
  let norm = fn(width, vectors, vector_stride, row_stride, mode) raise {
    @precision.RmsNormPrecision::new(
      vector_width=width,
      vectors_per_row=vectors,
      maximum_rows~,
      epsilon=0.000001,
      mode~,
      input_vector_stride=vector_stride,
      input_row_stride=row_stride,
    )
  }
  let decoder_norm = norm(
    geometry.hidden_width(),
    1,
    geometry.hidden_width(),
    geometry.hidden_width().to_int64(),
    Weighted(OnePlusGamma),
  )
  // q_proj is [head, query-or-gate, channel], not separate query/gate halves.
  let query_stride = full.head_width * 2
  let query_norm = norm(
    full.head_width,
    full.query_heads,
    query_stride,
    (full.query_heads * query_stride).to_int64(),
    Weighted(OnePlusGamma),
  )
  let key_norm = norm(
    full.head_width,
    full.kv_heads,
    full.head_width,
    (full.kv_heads * full.head_width).to_int64(),
    Weighted(OnePlusGamma),
  )
  let recurrent_norm = norm(
    linear.value_width,
    linear.value_heads,
    linear.value_width,
    (linear.value_heads * linear.value_width).to_int64(),
    Bf16NormBf16WeightSiluGate,
  )
  let router = @precision.SoftmaxTopKPrecision::new(
    expert_count=experts.routed,
    experts_per_token=experts.selected,
    maximum_rows~,
    weight_storage=Float32CarryingBf16,
  )
  let control = @precision.GatedDeltaControlPrecision::new(
    head_count=linear.value_heads,
    maximum_rows~,
  )
  let delta = @precision.RecurrentDeltaPrecision::new(
    head_count=linear.value_heads,
    query_key_heads=linear.key_heads,
    key_dimension=linear.key_width,
    value_dimension=linear.value_width,
    decay_layout=PerValueHead,
    maximum_rows~,
    maximum_sequences~,
    cache_slots~,
    normalization_epsilon=0.000001,
    maximum_state_bytes=geometry.fp32_recurrent_matrix_bytes(cache_slots),
    query_key_row_stride=channels.to_int64(),
    value_row_stride=channels.to_int64(),
  )
  let convolution = @precision.CausalConvolutionPrecision::new(
    channel_count=channels,
    kernel_size=linear.convolution_width,
    maximum_rows~,
    maximum_sequences~,
    cache_slots~,
    projection_boundary=Bf16BeforeSilu,
    maximum_state_bytes=cache_slots.to_int64() *
      channels.to_int64() *
      (linear.convolution_width - 1).to_int64() *
      2L,
  )
  let routed_experts = routed_plan(spec, maximum_rows~)
  let shared_mlp = @precision.DenseSwiGluPrecision::new(
    hidden_width=geometry.hidden_width(),
    intermediate_width=experts.shared_width,
    maximum_rows~,
  )
  let shared_combine = @precision.SharedExpertPrecision::new(
    hidden_width=geometry.hidden_width(),
    maximum_rows~,
  )
  let router_projection = @matrix.MatrixTilePlan::new(
    rows=maximum_rows,
    columns=experts.routed,
    inner=geometry.hidden_width(),
    law=ShapeAwareFloat32,
  )
  let shared_gate_projection = @matrix.MatrixTilePlan::new(
    rows=maximum_rows,
    columns=1,
    inner=geometry.hidden_width(),
    law=ShapeAwareFloat32,
  )
  let query_rotary = full_rotary(spec, full.query_heads, maximum_rows~)
  let key_rotary = full_rotary(spec, full.kv_heads, maximum_rows~)
  let full_output_gate = @precision.PointwiseGatePrecision::new(
    vector_width=full.head_width,
    vectors_per_row=full.query_heads,
    maximum_rows~,
    gate_vector_stride=full.head_width * 2,
    gate_row_stride=(full.head_width * full.query_heads * 2).to_int64(),
    gate_element_offset=full.head_width,
  )
  {
    spec,
    maximum_rows,
    decoder_norm,
    query_norm,
    key_norm,
    recurrent_norm,
    router,
    control,
    delta,
    convolution,
    routed_experts,
    shared_mlp,
    shared_combine,
    router_projection,
    shared_gate_projection,
    query_rotary,
    key_rotary,
    full_output_gate,
  }
}

///|
pub fn DecoderOperators::query_rotary(
  self : Self,
) -> @precision.MultiAxisRotaryPrecision {
  self.query_rotary
}

///|
pub fn DecoderOperators::key_rotary(
  self : Self,
) -> @precision.MultiAxisRotaryPrecision {
  self.key_rotary
}

///|
pub fn DecoderOperators::full_output_gate(
  self : Self,
) -> @precision.PointwiseGatePrecision {
  self.full_output_gate
}

///|
pub fn DecoderOperators::routed_experts(
  self : Self,
) -> @precision.ExpertMlpPrecision {
  self.routed_experts
}

///|
pub fn DecoderOperators::shared_mlp(
  self : Self,
) -> @precision.DenseSwiGluPrecision {
  self.shared_mlp
}

///|
pub fn DecoderOperators::shared_combine(
  self : Self,
) -> @precision.SharedExpertPrecision {
  self.shared_combine
}

///|
pub fn DecoderOperators::router_projection(
  self : Self,
) -> @matrix.MatrixTilePlan {
  self.router_projection
}

///|
pub fn DecoderOperators::shared_gate_projection(
  self : Self,
) -> @matrix.MatrixTilePlan {
  self.shared_gate_projection
}

///|
pub fn DecoderOperators::normalization(
  self : DecoderOperators,
  site : NormSite,
) -> @precision.RmsNormPrecision {
  match site {
    Decoder => self.decoder_norm
    FullQuery => self.query_norm
    FullKey => self.key_norm
    RecurrentOutput => self.recurrent_norm
  }
}

///|
pub fn DecoderOperators::routing(
  self : DecoderOperators,
) -> @precision.SoftmaxTopKPrecision {
  self.router
}

///|
pub fn DecoderOperators::controls(
  self : DecoderOperators,
) -> @precision.GatedDeltaControlPrecision {
  self.control
}

///|
pub fn DecoderOperators::recurrent(
  self : DecoderOperators,
) -> @precision.RecurrentDeltaPrecision {
  self.delta
}

///|
pub fn DecoderOperators::convolution(
  self : DecoderOperators,
) -> @precision.CausalConvolutionPrecision {
  self.convolution
}
