///|
async fn run_qwen3_release_bind(
  raw_arguments : ArrayView[String],
) -> Unit raise Qwen3Bf16ReleaseBindError {
  let arguments = parse_qwen3_release_bind_arguments(raw_arguments)
  let inputs = load_release_bind_inputs(arguments)
  let candidates = release_candidates(inputs, arguments)
  let compilations = admit_compiled_set(arguments, candidates)
  let release = @luna_bf16_kernel_producer.produce_luna_bf16_kernel_release(
    inputs.plan,
    release_layout(inputs.plan, arguments),
    release_target(),
    compilations,
    producer_limits(),
  ) catch {
    error => {
      ignore(error)
      raise KernelBinding
    }
  }
  let evidence = publish_release_binding(arguments, inputs, release)
  println("schema=lunaflux-qwen3-bf16-release-bind.v1")
  println("recipe=dense_qwen3_bf16_paged_aot_v12")
  println("model_content_sha256=" + arguments.model_content_sha256)
  println(
    "weight_route_manifest_sha256=" + arguments.weight_route_manifest_sha256,
  )
  println("model_plan_sha256=" + evidence.model_plan_sha256)
  println("target=sm_120")
  println("tokens_per_page=" + arguments.tokens_per_page.to_string())
  println("total_page_count=" + arguments.total_page_count.to_string())
  println(
    "max_sequence_tokens=" +
    inputs.plan.shape_constraints().max_sequence_tokens().to_string(),
  )
  println(
    "max_page_table_entries=" + arguments.max_page_table_entries.to_string(),
  )
  println("kernel_manifest_relative=lunaflux.execution.json")
  println("kernel_manifest_sha256=" + evidence.manifest_sha256)
  println("admitted_bootstrap_sha256=" + evidence.bootstrap_sha256)
  println("bootstrap_source_sha256=" + evidence.bootstrap_source_sha256)
  println("kernel_inventory_sha256=" + evidence.inventory_sha256)
  println("kernel_root_plan_sha256=" + evidence.plan_sha256)
  println("compiler_invoked=0")
  println("device_opened=0")
  println("runtime_authority=0")
  println(
    "authenticated_embedded_greedy_sampling=" +
    (if release.has_authenticated_embedded_greedy_sampling() {
      "1"
    } else {
      "0"
    }),
  )
}

///|
async fn main {
  try! run_qwen3_release_bind(@env.args())
}
