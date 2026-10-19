///|
test "projection rounding reaches all non-MoE families without changing expert arithmetic" {
  for rank in 0..<2 {
    let plans = [
      @precision_ir.ProjectionPublication::ReduceFloat32BeforeRound,
      @precision_ir.ReduceBFloat16AfterRound,
    ].map(publication => {
      @glm53_packed_expert_upload.GlmDecoderSlice::new(
        @glm53_spec.Glm53FlashSpec::official(),
        first_layer=0,
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
        expert_scaled_dot_law=BlockScaledMatrixFloat32(block_columns=16),
        moe_reduction=LocalCombinedBFloat16,
        tensor_publication=publication,
        tensor_group=(rank, 2),
      )
    })
    defer (for plan in plans {
      plan.close()
    })
    let old = @utf8.decode(plans[0].source_bytes())
    let next = @utf8.decode(plans[1].source_bytes())
    for
      symbol in [
        "glm_recurrent_0_16(", "glm_sparse_3_output_publish(", "glm_dense_ffn_0_publish(",
      ] {
      assert_true(old.contains(symbol))
      assert_false(next.contains(symbol))
    }
    for
      symbol in [
        "glm_routed_3_combine", "glm_shared_3_combine", "glm_moe_combine_3",
      ] {
      assert_true(old.contains(symbol))
      assert_true(next.contains(symbol))
    }
    // Five attention branches and three dense FFNs each drop their F32 arena.
    assert_eq(
      plans[0].workspace_bytes() - plans[1].workspace_bytes(),
      8L * 64L * 4096L * 4L,
    )
    let contract = plans[1].tensor_contract(
      @device.TensorCollectiveContractDigest::from_sha256(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      ),
    )
    assert_eq(contract.site_count(), 10)
  }
}

///|
test "partitioned projection cannot silently omit its collective" {
  try
    @glm53_packed_expert_upload.recurrent_block_plan(
      @glm53_spec.Glm53FlashSpec::official(),
      layer_index=0,
      maximum_rows=1,
      maximum_sequences=1,
      cache_slots=1,
      maximum_state_bytes=100000000L,
      tensor_publication=LocalBFloat16,
      tensor_group=(0, 2),
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("missing collective accepted")
  }
}
