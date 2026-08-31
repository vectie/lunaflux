///|
/// Selects optional execution observers at executor construction time.
///
/// This policy never relaxes artifact authentication, bounds checks, resource
/// ownership, lifecycle checks, or deterministic cleanup. The production
/// default omits qualification-only host buffers, device transfers, and CPU
/// scans from each token step.
pub(all) enum PagedExecutionDiagnosticPolicy {
  ProductionFastPath
  QualificationDiagnostics
} derive(Eq, Debug)

///|
fn fused_dispatch_policy_allows_span(
  policy : PagedExecutionDiagnosticPolicy,
  abi : @luna_cuda_fused_parallel_aot.FusedParallelKernelAbi,
) -> Bool {
  match policy {
    ProductionFastPath => abi == ResidualRmsNormProductionAbiV2
    QualificationDiagnostics => abi == ResidualRmsNormAbiV1
  }
}

///|
/// Qualification ABI v1 retains a per-dispatch canary around eager launches.
/// Production ABI v2 has no observer or mutable diagnostic argument, so it is
/// compatible with every already-authenticated graph execution policy.
fn fused_dispatch_policy_allows_graph(
  policy : PagedExecutionDiagnosticPolicy,
  graph_policy : PagedGraphExecutionPolicy,
) -> Bool {
  match policy {
    ProductionFastPath => true
    QualificationDiagnostics => graph_policy == PagedOrderedEagerOnly
  }
}

///|
fn prepare_fp8_scale_diagnostics(
  policy : PagedExecutionDiagnosticPolicy,
  workspace_base : Int64,
  workspace_bytes : Int64,
) -> Fp8ScaleEvidenceV3? {
  match policy {
    ProductionFastPath => None
    QualificationDiagnostics =>
      Some({
        workspace_base,
        workspace_bytes,
        host: FixedArray::make(workspace_bytes.to_int(), b'\x00'),
      })
  }
}

///|
fn prepare_fused_dispatch_diagnostics(
  policy : PagedExecutionDiagnosticPolicy,
  spans : ReadOnlyArray[ApprovedFusedResidualRmsNormSpan],
) -> FusedDispatchCanaryState? {
  match (policy, spans.get(0)) {
    (QualificationDiagnostics, Some(approved)) if approved.uses_qualification_canary() =>
      Some({
        per_token: approved.dispatch_canary_per_token(),
        host: FixedArray::make(4, b'\x00'),
      })
    _ => None
  }
}
