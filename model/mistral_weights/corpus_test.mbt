///|
async test "pinned Mistral corpus authenticates plan, manifest, and numeric artifact" {
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
    "model/mistral_weights/fixtures/tiny_mistral_v1.json",
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
