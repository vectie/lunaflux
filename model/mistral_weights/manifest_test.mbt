///|
test "Mistral binds the exact 21-operation two-layer weight vocabulary" {
  let metadata = mistral_fixture_metadata()
  let plan = @mistral.build_paged(metadata)
  let manifest = @mistral_weights.bind(
    metadata,
    plan,
    mistral_fixture_safetensors(),
  )
  assert_eq(plan.operations().length(), 21)
  assert_eq(manifest.length(), 21)
  assert_eq(manifest.identity(), plan.identity())
  assert_eq(
    manifest.binding(try! @plan.TensorRef::new(0)).unwrap().tensor().name(),
    "model.embed_tokens.weight",
  )
  assert_eq(
    manifest.binding(try! @plan.TensorRef::new(20)).unwrap().tensor().name(),
    "lm_head.weight",
  )
  assert_eq(manifest.manifest_sha256().length(), 64)
}

///|
test "Mistral rejects missing, extra, and mis-shaped manifests" {
  let metadata = mistral_fixture_metadata()
  let plan = @mistral.build_paged(metadata)
  try
    @mistral_weights.bind(metadata, plan, mistral_fixture_safetensors(rename=7))
  catch {
    MissingTensor(reference=7) => ()
    _ => fail("unexpected missing-name rejection")
  } noraise {
    _ => fail("missing Mistral tensor was admitted")
  }
  try
    @mistral_weights.bind(
      metadata,
      plan,
      mistral_fixture_safetensors(extra=true),
    )
  catch {
    Mismatch(TensorCount) => ()
    _ => fail("unexpected extra-tensor rejection")
  } noraise {
    _ => fail("extra Mistral tensor was admitted")
  }
  try
    @mistral_weights.bind(
      metadata,
      plan,
      mistral_fixture_safetensors(shape_change=8),
    )
  catch {
    InvalidTensor(reference=8, TensorRank | TensorDimension) => ()
    _ => fail("unexpected shape rejection")
  } noraise {
    _ => fail("mis-shaped Mistral tensor was admitted")
  }
}

///|
test "Mistral rejects content, plan, and sliding-window replay" {
  let metadata = mistral_fixture_metadata(window=16)
  let plan = @mistral.build_paged(metadata)
  let foreign_content = mistral_fixture_metadata(
    content="2222222222222222222222222222222222222222222222222222222222222222",
    window=16,
  )
  try
    @mistral_weights.bind(foreign_content, plan, mistral_fixture_safetensors())
  catch {
    Mismatch(ModelContent) => ()
    _ => fail("unexpected content replay rejection")
  } noraise {
    _ => fail("foreign content identity was admitted")
  }
  let foreign_window = mistral_fixture_metadata(window=32)
  try
    @mistral_weights.bind(foreign_window, plan, mistral_fixture_safetensors())
  catch {
    Mismatch(ModelPlan) => ()
    _ => fail("unexpected sliding-window replay rejection")
  } noraise {
    _ => fail("foreign sliding-window plan was admitted")
  }
  let stateless = @mistral.build(metadata)
  try
    @mistral_weights.bind(metadata, stateless, mistral_fixture_safetensors())
  catch {
    _ => fail("exact stateless plan should remain admissible")
  } noraise {
    manifest => assert_eq(manifest.length(), 21)
  }
}
