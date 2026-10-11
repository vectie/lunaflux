///|
// CPU-only attribution tool. It reads small model configuration, never weights,
// and captures initial latent values independently of CUDA or media execution.
async fn main {
  guard @env.args() is [_, root, label, seed_text, output] else {
    fail("CONFIG_ROOT DECLARED_MODEL_LABEL SEED NEW_OUTPUT_DIRECTORY")
  }
  let pipeline = @config.parse_pipeline_index(
    @fs.read_file(root + "/model_index.json").text(),
    max_bytes=1048576,
    max_depth=16,
  )
  let metadata = @config.parse_transformer(
    @fs.read_file(root + "/transformer/config.json").text(),
    variant=pipeline.variant(),
    content_digest=@spec.ContentDigest::from_sha256(label),
    max_bytes=1048576,
    max_depth=16,
  )
  let model = @model.build(metadata)
  let request = model
    .shapes()
    .resolve_request(
      height=32,
      width=32,
      requested_frames=120,
      inference_steps=5,
    )
  let prepared = @loop.PreparedPlan::new(
    model,
    request,
    max_latent_elements=1048576L,
  )
  let storage = @loop.LatentStoragePlan::new(prepared, max_bytes=1048576L)
  let seed = @string.parse_int64(seed_text)
  let noise = storage.noise(seed~, max_chunk_elements=1048576)
  @fs.mkdir(output)
  let streams : Array[(String, @joint.JointDiffusionLatentKind)] = [
    ("video", VideoLatent),
    ("audio", AudioLatent),
  ]
  for (name, latent) in streams {
    let count = noise.element_count(latent).to_int()
    @fs.write_file(
      output + "/" + name + ".f32",
      noise.chunk(latent, start=0L, count~),
      create_mode=CreateNew,
    )
    println("initial_noise name=\{name} elements=\{count}")
  }
  // Diagnose the removed v1 key coupling, not payload authentication. This
  // allows a v2 explicit seed to replay a historical v1 initial tensor.
  let digest = model.plan_digest().as_hex()
  let legacy = for index in 0..<digest.length(); bits = 14695981039346656037UL {
    continue (bits ^ digest[index].to_uint64()) * 1099511628211UL
  } nobreak {
    bits ^ seed.reinterpret_as_uint64()
  }
  let metadata = "plan_label=\{digest}\nseed=\{seed}\nlegacy_effective_seed=\{legacy.reinterpret_as_int64()}\nGPU_execution=false\n"
  @fs.write_file(output + "/PLAN.txt", metadata, create_mode=CreateNew)
  println(metadata)
}
