///|
test "Flash text IO selects one prefill row without learned head parameters" {
  let plan = @glm53_packed_expert_upload.flash_text_io_plan(
    @glm53_spec.Glm53FlashSpec::official(),
    maximum_rows=4096,
    maximum_output_rows=1,
    maximum_workspace_bytes=627712L,
  )
  assert_eq(plan.fold(StreamUpdate).extent, 4)
  assert_true(plan.egress_program().happens_before(0, 2))
  assert_eq(plan.replay_boundaries()[1].label, "logits")
  let owner = @glm53_packed_expert_upload.GlmTextIo::new(
    @glm53_spec.Glm53FlashSpec::official(),
    maximum_rows=4096,
    maximum_output_rows=1,
    maximum_workspace_bytes=627712L,
  )
  defer owner.close()
  let text = @utf8.decode(owner.source_bytes())
  for
    symbol in [
      "glm_text_io_embed", "glm_text_io_norm", "glm_text_io_head", "glm_text_io_greedy",
    ] {
    assert_true(text.contains("void " + symbol + "("))
  }
  assert_true(text.contains("live > 4096"))
  assert_true(text.contains("n > 1"))
  assert_true(text.contains("stream<4"))
  assert_false(text.contains("head_fn"))
  try owner.replay_buffers() catch {
    _ => ()
  } noraise {
    _ => fail("fresh model head exposed replay")
  }
}
