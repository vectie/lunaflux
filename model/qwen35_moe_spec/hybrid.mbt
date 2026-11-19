///|
/// Exposes semantic geometry without carrying this model's FP8 storage policy
/// into another family. The existing pinned public getters remain authoritative.
pub fn Qwen35MoeTextSpec::hybrid_geometry(
  self : Qwen35MoeTextSpec,
) -> @hybrid.Geometry {
  try! @hybrid.Geometry::new(
    hidden_width=self.hidden_size(),
    vocabulary_size=self.vocabulary_size(),
    context_length=self.context_length(),
    layer_count=self.layer_count(),
    full_interval=self.full_attention_interval(),
    full={
      query_heads: self.attention_head_count(),
      kv_heads: self.kv_head_count(),
      head_width: self.head_dimension(),
      output_gate: self.attention_output_gate(),
    },
    recurrent={
      key_heads: self.linear_key_head_count(),
      key_width: self.linear_key_head_dimension(),
      value_heads: self.linear_value_head_count(),
      value_width: self.linear_value_head_dimension(),
      convolution_width: self.linear_convolution_width(),
    },
    experts={
      routed: self.routed_expert_count(),
      selected: self.experts_per_token(),
      intermediate_width: self.expert_intermediate_size(),
      shared_width: self.shared_expert_intermediate_size(),
    },
  )
}
