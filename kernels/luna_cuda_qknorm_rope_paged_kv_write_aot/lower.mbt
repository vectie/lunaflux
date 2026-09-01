///|
pub fn lower_qknorm_rope_paged_kv_write_cuda_aot_candidate(
  plan : @plan.ModelPlan,
  semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  limits : QkNormRopePagedKvWriteAotLimits,
) -> QkNormRopePagedKvWriteCudaCandidate raise QkNormRopePagedKvWriteAotError {
  let (query_norm_weight, key_norm_weight) = postprocess_validate(
    plan, semantic, profile, layout, target, compiler, operands,
  )
  let (_, output_width) = semantic.projection_shape()
  let (query_heads, kv_heads, _) = semantic.attention_shape()
  let grid_x = Int::min(profile.max_query_tokens(), limits.maximum_grid_x)
  if grid_x <= 0 {
    raise Unsupported(Profile)
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x~,
    grid_y=query_heads + 2 * kv_heads,
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise Unsupported(Shape)
  }
  if output_width <= 0 {
    raise Unsupported(Shape)
  }
  let alias_contract = { first_writable_operand: 9, operand_count: 12 }
  let source = postprocess_source(semantic, layout, profile, grid_x)
  if source.length() > limits.max_source_bytes {
    raise LimitExceeded(Source)
  }
  let source_digest = QkNormRopePagedKvWriteDigest(postprocess_sha256(source))
  let recipe = postprocess_recipe(
    plan, semantic, query_norm_weight, key_norm_weight, profile, layout, target,
    compiler, operands, dimensions, alias_contract, source_digest,
  )
  if recipe.length() > limits.max_recipe_bytes {
    raise LimitExceeded(Recipe)
  }
  let recipe_digest = QkNormRopePagedKvWriteDigest(postprocess_sha256(recipe))
  let candidate_digest = QkNormRopePagedKvWriteDigest(
    postprocess_sha256(
      @utf8.encode(
        "schema=lunaflux-qknorm-rope-paged-kv-write-candidate-identity.v1\n" +
        "source_sha256=" +
        source_digest.as_hex() +
        "\n" +
        "recipe_sha256=" +
        recipe_digest.as_hex() +
        "\n",
      ),
    ),
  )
  {
    semantic,
    model_content: plan.identity().content().as_hex(),
    model_plan: plan.identity().plan().as_hex(),
    target,
    profile,
    compiler,
    layout,
    operands: ReadOnlyArray::from_array(operands),
    dimensions,
    symbol: postprocess_production_symbol,
    abi: QwenQkNormPositionedRopePagedKvWriteProductionV1,
    alias_contract,
    source,
    source_digest,
    recipe,
    recipe_digest,
    candidate_digest,
  }
}
