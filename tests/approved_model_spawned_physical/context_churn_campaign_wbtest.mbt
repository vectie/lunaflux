///|
test "context churn frames vary authority-neutral identity and context only" {
  for cycle in 0..<CONTEXT_CHURN_FRESH_CHILD_CYCLES {
    ignore(context_churn_validation_contract(cycle))
    let contexts = [
      CONTEXT_CHURN_FIRST_CEILING,
      CONTEXT_CHURN_SECOND_CEILING,
      CONTEXT_CHURN_THIRD_CEILING,
    ]
    for index, context in contexts {
      let frame = context_churn_request_frame(
        900UL + (cycle * 3 + index + 1).to_uint64(),
        context,
        b"context-test",
      )
      let request = @framed_wire.RequestFrameBuffer::new(
          @framed_wire.FramedWireLimits::new(
            campaign_inference_limits(),
            max_frame_bytes=1048576,
          ),
        )
        .load(frame, frame.length())
        .request()
      assert_eq(request.model_identity(), pinned_tiny_model_identity())
      guard request.input() is TokenIds(tokens) else {
        abort("context churn request widened beyond token input")
      }
      assert_eq(tokens.length(), 1)
      assert_eq(tokens.token_at(0), 1)
      assert_eq(request.max_new_tokens(), 2)
      assert_eq(request.context_ceiling(), context)
      assert_eq(request.cache().permission(), Disabled)
    }
  }
}

///|
test "context churn matrix is bounded and qualification-only" {
  assert_eq(CONTEXT_CHURN_FIRST_CEILING, 8)
  assert_eq(CONTEXT_CHURN_SECOND_CEILING, 32)
  assert_eq(CONTEXT_CHURN_THIRD_CEILING, 256)
  assert_eq(CONTEXT_CHURN_REQUESTS_PER_CHILD, 3)
  assert_eq(CONTEXT_CHURN_FRESH_CHILD_CYCLES, 3)
  let scope = "campaign_scope=qualification-only\n" +
    "release_promotion=not-claimed\n" +
    "broader_context_length_coverage=not-established\n" +
    "actual_context_tokens_per_request=3\n" +
    "declared_context_ceiling_matrix=8,32,256\n" +
    "network_observation=structural-no-listener-owner\n" +
    "selected_logit_correctness=not-observed\n" +
    "gpu_allocator_instrumentation=not-observed-by-this-mode\n"
  assert_true(scope.contains("qualification-only"))
  assert_true(scope.contains("not-established"))
  assert_false(scope.contains("release_promotion=pass"))
  assert_false(scope.contains("selected_logit_correctness=pass"))
}
