///|
test "local MoE TP removes concatenation from both layer families and both ranks" {
  for rank in 0..<2 {
    let plans = [
      @precision_ir.ExpertPartialsFloat32,
      @precision_ir.LocalCombinedBFloat16,
    ].map(placement => {
      @glm53_packed_expert_upload.GlmDecoderSlice::new(
        @glm53_spec.Glm53FlashSpec::official(),
        first_layer=3,
        end_layer=5,
        maximum_rows=64,
        maximum_sequences=1,
        cache_slots=1,
        maximum_history=64,
        index_layer_norm_epsilon=1.0e-5,
        maximum_layer_state_bytes=100000000L,
        maximum_layer_workspace_bytes=100000000L,
        maximum_layer_expert_bytes=5000000000L,
        accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
        matrix_law=ShapeAwareFloat32,
        expert_scale_partitions=32,
        expert_scaled_dot_law=BlockScaledMatrixFloat32(block_columns=16),
        moe_reduction=placement,
        tensor_group=(rank, 2),
      )
    })
    defer (for plan in plans {
      plan.close()
    })
    let original = @utf8.decode(plans[0].source_bytes())
    let local = @utf8.decode(plans[1].source_bytes())
    for layer in [3, 4] {
      assert_true(original.contains("glm_moe_combine_\{layer}_collective_pack"))
      assert_false(local.contains("glm_moe_combine_\{layer}_collective_pack"))
      assert_false(local.contains("glm_moe_combine_\{layer}_collective_unpack"))
      assert_true(local.contains("glm_routed_\{layer}_combine"))
      assert_true(local.contains("glm_shared_\{layer}_combine"))
    }
    // Nothing else changes its persistent allocation: two 9 MiB joined buffers.
    assert_eq(
      plans[0].workspace_bytes() - plans[1].workspace_bytes(),
      2L * 64L * 4096L * 9L * 4L,
    )
    let contract = plans[1].tensor_contract(
      @device.TensorCollectiveContractDigest::from_sha256(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      ),
    )
    assert_eq(contract.site_count(), 4)
  }
}

///|
test "local MoE TP cannot accidentally reduce replicated shared work" {
  try
    @glm53_packed_expert_upload.GlmRecurrentLayer::with_moe_block(
      @glm53_spec.Glm53FlashSpec::official(),
      layer_index=4,
      maximum_rows=1,
      maximum_sequences=1,
      cache_slots=1,
      maximum_state_bytes=100000000L,
      maximum_expert_bytes=5000000000L,
      accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
      moe_reduction=LocalCombinedBFloat16,
    )
  catch {
    _ => ()
  } noraise {
    plan => {
      plan.close()
      fail("unpartitioned branch accepted")
    }
  }
}
