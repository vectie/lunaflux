///|
fn numeric_layout_operations() -> Array[@plan.PlanOperation] {
  [
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
}

///|
fn numeric_layout_plan(mode : Int) -> @plan.ModelPlan {
  let operations = numeric_layout_operations()
  let selected = [try! @plan.OperationId::new(1)]
  let numeric_schema = match mode {
    1 => @plan_test_fixture.synthetic_bf16_numeric_schema(operations)
    2 =>
      @plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(operations, selected)
    3 =>
      @plan_test_fixture.synthetic_symmetric_i8_weight_only_numeric_schema_v1(
        operations, selected,
      )
    _ => abort("unknown numeric layout fixture mode")
  }
  try! @plan.ModelPlan::new_stateless(
    content_digest=test_model().content_digest(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=@plan.ShapeConstraints::new(
      max_batch_rows=1,
      max_sequence_tokens=16,
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=6,
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
    numeric_schema~,
  )
}
