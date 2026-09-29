///|
/// Recover the semantic origin from the selected artifact span once. Artifact
/// selection changes the start of a partial module, never the underlying cut.
fn selected_ingress_cut(
  args : RuntimeBundleArguments,
  choice : @luna_fusion_strategy.FusionSchedule,
) -> @luna_fusion_plan.AttentionIngressCutPlan raise Qwen3FusedRuntimeBundleExportError {
  let qkv_operation = if choice == ProducerSeparated {
    args.ingress.first_operation - 1
  } else {
    args.ingress.first_operation
  }
  @luna_fusion_plan.AttentionIngressCutPlan::new(choice, qkv_operation~) catch {
    _ => raise InvalidArguments
  }
}
