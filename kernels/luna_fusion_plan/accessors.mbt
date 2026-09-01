///|
pub fn QkvQkNormPositionedRopePagedKvWritePlan::family(
  _self : QkvQkNormPositionedRopePagedKvWritePlan,
) -> FusionSemanticFamily {
  QkvQkNormPositionedRopePagedKvWrite
}

///|
pub fn QkvQkNormPositionedRopePagedKvWritePlan::operation_indices(
  self : QkvQkNormPositionedRopePagedKvWritePlan,
) -> (Int, Int, Int, Int) {
  (
    self.qkv_operation.as_int(),
    self.qk_norm_operation.as_int(),
    self.rope_operation.as_int(),
    self.attention_operation.as_int(),
  )
}

///|
pub fn QkvQkNormPositionedRopePagedKvWritePlan::kv_layer(
  self : QkvQkNormPositionedRopePagedKvWritePlan,
) -> @plan.KvLayerId {
  self.kv_layer
}

///|
pub fn QkvQkNormPositionedRopePagedKvWritePlan::projection_shape(
  self : QkvQkNormPositionedRopePagedKvWritePlan,
) -> (Int, Int) {
  (self.input_width, self.output_width)
}

///|
pub fn QkvQkNormPositionedRopePagedKvWritePlan::attention_shape(
  self : QkvQkNormPositionedRopePagedKvWritePlan,
) -> (Int, Int, Int) {
  (self.query_heads, self.kv_heads, self.head_dimension)
}

///|
pub fn QkvQkNormPositionedRopePagedKvWritePlan::norm_epsilon(
  self : QkvQkNormPositionedRopePagedKvWritePlan,
) -> Double {
  self.norm_epsilon
}

///|
pub fn QkvQkNormPositionedRopePagedKvWritePlan::rope_theta(
  self : QkvQkNormPositionedRopePagedKvWritePlan,
) -> Double {
  self.rope_theta
}

///|
/// Activation produced by the separately lowerable QKV projection. Keeping
/// this identity in semantic IR lets a backend fuse the post-projection work
/// without replacing a faster shape-selected projection kernel.
pub fn QkvQkNormPositionedRopePagedKvWritePlan::projected_qkv(
  self : QkvQkNormPositionedRopePagedKvWritePlan,
) -> @plan.ActivationRef {
  self.projected_qkv
}

///|
pub fn QkvQkNormPositionedRopePagedKvWritePlan::positioned_qkv(
  self : QkvQkNormPositionedRopePagedKvWritePlan,
) -> @plan.ActivationRef {
  self.positioned_qkv
}
