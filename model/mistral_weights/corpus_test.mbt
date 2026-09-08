///|
async test "current Mistral row-end corpus authenticates plan, manifest, and numeric artifact" {
  let metadata = mistral_fixture_metadata()
  let plan = @mistral.build_paged(metadata)
  let manifest = @mistral_weights.bind(
    metadata,
    plan,
    mistral_fixture_safetensors(),
  )
  let artifact = mistral_fixture_artifact_digest(
    mistral_fixture_numeric_file(plan),
  )
  let bytes = @fs.read_file(
    "model/mistral_weights/fixtures/tiny_mistral_v2.json",
  ).binary()
  guard @json.parse(@utf8.decode(bytes)) is Object(corpus) &&
    corpus["schema"] is String("lunaflux.mistral.synthetic-corpus.v1") &&
    corpus["license"] is String("CC0-1.0") &&
    corpus["redistributable"] is True &&
    corpus["synthetic"] is True &&
    corpus["operations"] is Number(21.0, ..) &&
    corpus["tensors"] is Number(21.0, ..) &&
    corpus["sliding_window"] is Number(16.0, ..) &&
    corpus["model_content_sha256"] is String(content) &&
    corpus["model_plan_sha256"] is String(plan_digest) &&
    corpus["weight_manifest_sha256"] is String(manifest_digest) &&
    corpus["numeric_artifact_sha256"] is String(artifact_digest) else {
    fail("invalid pinned Mistral corpus envelope")
  }
  assert_eq(content, metadata.content_digest().as_hex())
  assert_eq(plan_digest, plan.identity().plan().as_hex())
  assert_eq(manifest_digest, manifest.manifest_sha256())
  assert_eq(artifact_digest, artifact.as_hex())
}

///|
// Preserve the historical v1 corpus bytes: its head computed every token.
// Only this test fixture reconstructs that old, still-valid plan contract.
fn mistral_legacy_full_token_corpus_plan(
  source : @plan.ModelPlan,
) -> @plan.ModelPlan raise {
  let operations : Array[@plan.PlanOperation] = []
  for operation in source.operations() {
    if operation.kind() == LanguageModelHead {
      operations.push(
        @plan.PlanOperation::new_with_runtime_inputs(
          id=operation.id(),
          kind=operation.kind(),
          inputs=operation.inputs(),
          outputs=operation.outputs(),
          shape=operation.shape(),
          required_capability=operation.required_capability(),
          workspace_bytes=operation.workspace_bytes(),
          runtime_inputs=[StepCounts],
        ),
      )
    } else {
      operations.push(operation)
    }
  }
  @plan.ModelPlan::new_with_kv_execution(
    content_digest=source.identity().content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=source.shape_constraints(),
    kv_cache_geometry=source.kv_cache_geometry(),
    workspace_bounds=source.workspace_bounds(),
    operations~,
    final_output=source.final_output(),
    kv_execution=PagedKeyValue,
    numeric_schema=source.numeric_binding().schema(),
  )
}

///|
async test "historical Mistral corpus differs only by full-token head contract" {
  let metadata = mistral_fixture_metadata()
  let current = @mistral.build_paged(metadata)
  let plan = mistral_legacy_full_token_corpus_plan(current)
  let artifact = mistral_fixture_artifact_digest(
    mistral_fixture_numeric_file(plan),
  )
  let bytes = @fs.read_file(
    "model/mistral_weights/fixtures/tiny_mistral_v1.json",
  ).binary()
  guard @json.parse(@utf8.decode(bytes)) is Object(corpus) &&
    corpus["model_plan_sha256"] is String(plan_digest) &&
    corpus["numeric_artifact_sha256"] is String(artifact_digest) else {
    fail("invalid historical Mistral corpus envelope")
  }
  assert_eq(plan_digest, plan.identity().plan().as_hex())
  assert_eq(artifact_digest, artifact.as_hex())
  assert_not_eq(current.identity().plan(), plan.identity().plan())
  assert_eq(current.numeric_binding(), plan.numeric_binding())
  for operation in current.operations() {
    if operation.kind() == LanguageModelHead {
      assert_eq(operation.runtime_inputs(), [StepCounts, QueryRowOffsets])
    }
  }
  // The current binder must not accept historical, non-row-selected plans.
  try
    @mistral_weights.bind(metadata, plan, mistral_fixture_safetensors())
  catch {
    @mistral_weights.MistralWeightsError::Mismatch(_) => ()
    _ => fail("unexpected historical plan rejection")
  } noraise {
    _ => fail("historical full-token plan silently admitted as current")
  }
}
