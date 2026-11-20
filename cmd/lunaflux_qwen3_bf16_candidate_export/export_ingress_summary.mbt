///|
fn print_qwen3_ingress_summary(
  ingress_root : String,
  qwen_fused : QwenFusedProductionSet,
) -> Unit {
  println("reusable_qwen_ingress_source=" + ingress_root + "/kernel.cu")
  println("reusable_qwen_ingress_recipe=" + ingress_root + "/kernel.recipe")
  println(
    "reusable_qwen_ingress_candidate_sha256=" +
    qwen_fused.postprocess.candidate_digest().as_hex(),
  )
  println(
    "reusable_qwen_ingress_first_operation=" +
    qwen_fused.first_operation.to_string(),
  )
  println(
    "reusable_qwen_ingress_function_symbol=" +
    qwen_fused.postprocess.function_symbol(),
  )
  let ingress_launch = qwen_fused.postprocess.launch_dimensions()
  println(
    "reusable_qwen_ingress_launch=" +
    ingress_launch.grid_x().to_string() +
    "," +
    ingress_launch.grid_y().to_string() +
    "," +
    ingress_launch.block_x().to_string() +
    "," +
    ingress_launch.shared_memory_bytes().to_string(),
  )
}
