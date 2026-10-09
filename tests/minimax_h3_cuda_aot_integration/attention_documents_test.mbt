///|
test "attention AOT carries document isolation into source and recipe identity" {
  let requirements = official_requirements(Fl2va)
  let (io, timestep, _, _) = packed_selection_fixture_with_io(Fl2va)
  fn lower(boundaries : ArrayView[Int]) raise {
    @joint_diffusion_cuda_aot.lower_transformer_attention_cuda_aot_candidate(
      requirements,
      request=io,
      packed_timestep=timestep,
      requirement_ordinal=transformer_attention_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
      document_boundaries=boundaries,
    )
  }
  let whole = lower([0, timestep.packed_row_count()])
  let split = lower([0, 137, timestep.packed_row_count()])
  assert_true(whole.source_digest() != split.source_digest())
  assert_true(whole.recipe_digest() != split.recipe_digest())
  let source = @utf8.decode(split.source_bytes())
  assert_true(source.contains("if (qr >= 137ULL) { key_begin = 137;"))
  assert_eq(whole.packed_row_count(), split.packed_row_count())
  assert_false(split.manifest_bindable())
}
