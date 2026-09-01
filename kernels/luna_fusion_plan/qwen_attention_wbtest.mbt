///|
fn fusion_qwen_plan() -> @plan.ModelPlan raise {
  let spec = @qwen3_spec.Qwen3ModelSpec::new(
    vocabulary_size=1024,
    hidden_size=128,
    intermediate_size=256,
    layer_count=2,
    attention_head_count=4,
    kv_head_count=2,
    head_dimension=32,
    context_length=256,
    rope_theta=1000000.0,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
    tied_embeddings=true,
  )
  @qwen3.build_paged(
    @qwen3_spec.Qwen3ModelMetadata::from_verified_content(
      spec,
      @spec.ContentDigest::from_sha256("3".repeat(64)),
    ),
    batch=@qwen3.Qwen3PagedBatchEnvelope::new(4),
  )
}

///|
test "Qwen fusion plan matches QKV QKNorm RoPE and paged KV write" {
  let plan = fusion_qwen_plan()
  let matches = find_qkv_qknorm_positioned_rope_paged_kv_write_plans(plan)
  assert_eq(matches.length(), 2)
  let first = matches[0]
  assert_eq(first.family(), QkvQkNormPositionedRopePagedKvWrite)
  assert_eq(first.operation_indices(), (2, 3, 4, 5))
  assert_eq(first.projection_shape(), (128, 256))
  assert_eq(first.attention_shape(), (4, 2, 32))
  assert_eq(first.norm_epsilon(), 0.000001)
  assert_eq(first.rope_theta(), 1000000.0)
  assert_eq(first.projected_qkv(), plan.operations()[2].outputs()[0])
  assert_eq(first.kv_layer().as_int(), 0)
}

///|
test "Qwen fusion plan rejects a non-QKV starting operation" {
  let plan = fusion_qwen_plan()
  try
    ignore(
      match_qkv_qknorm_positioned_rope_paged_kv_write(
        plan,
        plan.operations()[1].id(),
      ),
    )
  catch {
    InvalidFusionPlan(OperationOrder) => ()
    _ => fail("non-QKV start reported the wrong fusion failure")
  } noraise {
    _ => fail("non-QKV start was accepted")
  }
}
