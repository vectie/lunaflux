///|
fn print_qwen_ingress_resource_policy(
  arguments : Qwen3CandidateExportArguments,
  fused : QwenFusedProductionSet,
  full_ingress_root : String,
) -> Unit {
  let device = match arguments.ingress_policy_records {
    Some((_, _, device)) => device
    None => "unmeasured-device"
  }
  // All alternatives are compared against the untuned baseline's scope, not
  // the winning schedule's digest. This makes saved tables replayable.
  let (_, toolchain, shape, program) = fused.ingress_variants[0]
    .resource_scope(device~, toolchain=arguments.toolchain_sha256)
    .unwrap()
    .identity()
  println("ingress_policy_scope_device=" + device)
  println("ingress_policy_scope_toolchain=" + toolchain)
  println("ingress_policy_scope_shape=" + shape)
  println("ingress_policy_scope_program=" + program)
  println("ingress_policy_selected=" + fused.full_ingress.ingress_policy().id())
  println(
    "ingress_policy_selection=" +
    (match fused.ingress_selection_origin {
      Measured(..) => "measured"
      UnmeasuredPreferred => "unmeasured-preferred"
      UnmeasuredEligible => "unmeasured-eligible"
    }),
  )
  for variant in fused.ingress_variants {
    println(
      "ingress_policy_candidate=" +
      variant.ingress_policy().id() +
      "," +
      variant.source_digest().as_hex() +
      "," +
      full_ingress_root +
      "-" +
      variant.ingress_policy().id(),
    )
  }
}
