///|
fn d2_wb_content() -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
}

///|
fn d2_wb_plan() -> @plan.ModelPlan {
  let operations = [
    try! @plan.PlanOperation::new_stateless(
      id=@plan.OperationId::new(0),
      kind=Embedding,
      inputs=[ModelInput(TokenIds), Tensor(@plan.TensorRef::new(0))],
      outputs=[@plan.ActivationRef::new(0)],
      shape=EmbeddingShape(vocabulary_size=8, width=4),
      required_capability=@plan.KernelCapabilityId::embedding_lookup(),
      workspace_bytes=0L,
    ),
    try! @plan.PlanOperation::new_stateless(
      id=@plan.OperationId::new(1),
      kind=LanguageModelHead,
      inputs=[
        Activation(@plan.ActivationRef::new(0)),
        Tensor(@plan.TensorRef::new(1)),
      ],
      outputs=[@plan.ActivationRef::new(1)],
      shape=ProjectionShape(input_width=4, output_width=8),
      required_capability=@plan.KernelCapabilityId::dense_projection(),
      workspace_bytes=0L,
    ),
  ]
  try! @plan.ModelPlan::new_stateless(
    content_digest=d2_wb_content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=8,
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=8,
      attention_head_count=2,
      kv_head_count=1,
      head_dimension=2,
    ),
    kv_cache_geometry=@plan.KvCacheGeometry::new(
      layer_count=1,
      kv_head_count=1,
      head_dimension=2,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    workspace_bounds=@plan.WorkspaceBounds::new(max_temporary_bytes=0L),
    operations~,
    final_output=@plan.ActivationRef::new(1),
    numeric_schema=@plan_test_fixture.synthetic_bf16_numeric_schema(operations),
  )
}

///|
fn d2_wb_fact(
  reference : Int,
  offset_bytes : Int64,
  byte_length : Int64,
) -> WeightRegionFacts? {
  Some({
    reference: try! @plan.TensorRef::new(reference),
    offset_bytes,
    byte_length,
  })
}

///|
fn d2_wb_exact_facts() -> Array[WeightRegionFacts?] {
  [d2_wb_fact(0, 0L, 64L), d2_wb_fact(1, 64L, 64L)]
}
