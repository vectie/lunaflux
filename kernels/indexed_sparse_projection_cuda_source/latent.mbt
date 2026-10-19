///|
/// Absorption/expansion are separately declared finite-precision operations.
/// This adapter selects generic strided matrices from the semantic geometry.
fn latent_head_plan(
  plan : @precision.IndexedSparseProjectionPrecision,
  expand : Bool,
) -> @matrix.HeadProjectionPlan raise {
  let (heads, kv_heads, keys, values) = plan.attention().attention_geometry()
  let rank = plan.geometry().2
  let stride = (keys + values).to_int64() * rank.to_int64()
  @matrix.HeadProjectionPlan::new(
    rows=plan.attention().capacity().0,
    heads~,
    weight_heads=kv_heads,
    columns=if expand { values } else { rank },
    inner=if expand { rank } else { keys },
    head_stride=stride,
    column_stride=if expand { rank.to_int64() } else { 1L },
    inner_stride=if expand { 1L } else { rank.to_int64() },
    offset=if expand { keys.to_int64() * rank.to_int64() } else { 0L },
    weight_elements=kv_heads.to_int64() * stride,
  )
}
