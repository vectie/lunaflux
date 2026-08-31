///|
test "production execution policy allocates no diagnostic host state" {
  assert_true(
    prepare_fp8_scale_diagnostics(ProductionFastPath, 64L, 128L) is None,
  )
  assert_true(
    prepare_fused_dispatch_diagnostics(
      ProductionFastPath,
      ReadOnlyArray::from_array([]),
    )
    is None,
  )
}

///|
test "fused execution policy separates qualification and production ABIs" {
  assert_false(
    fused_dispatch_policy_allows_span(ProductionFastPath, ResidualRmsNormAbiV1),
  )
  assert_true(
    fused_dispatch_policy_allows_span(
      QualificationDiagnostics,
      ResidualRmsNormAbiV1,
    ),
  )
  assert_true(
    fused_dispatch_policy_allows_span(
      ProductionFastPath,
      ResidualRmsNormProductionAbiV2,
    ),
  )
  assert_false(
    fused_dispatch_policy_allows_span(
      QualificationDiagnostics,
      ResidualRmsNormProductionAbiV2,
    ),
  )
  for
    graph_policy in [
      PagedOrderedEagerOnly,
      PagedCapturedRequired,
      PagedCapturedWithEagerFallback,
    ] {
    assert_true(
      fused_dispatch_policy_allows_graph(ProductionFastPath, graph_policy),
    )
  }
  assert_true(
    fused_dispatch_policy_allows_graph(
      QualificationDiagnostics,
      PagedOrderedEagerOnly,
    ),
  )
  assert_false(
    fused_dispatch_policy_allows_graph(
      QualificationDiagnostics,
      PagedCapturedRequired,
    ),
  )
  assert_false(
    fused_dispatch_policy_allows_graph(
      QualificationDiagnostics,
      PagedCapturedWithEagerFallback,
    ),
  )
}

///|
test "production FP8 hooks return before device access or CPU scans" {
  let executor = dummy_paged_executor(GraphExecuting)
  arm_fp8_scale_evidence_v3(executor)
  validate_fp8_scale_evidence_v3(executor)
  assert_eq(executor.lifecycle(), GraphExecuting)
}

///|
test "qualification policy retains bounded FP8 diagnostic scratch" {
  guard prepare_fp8_scale_diagnostics(QualificationDiagnostics, 64L, 128L)
    is Some(evidence) else {
    fail("qualification diagnostics omitted FP8 scratch")
  }
  assert_eq(evidence.workspace_base, 64L)
  assert_eq(evidence.workspace_bytes, 128L)
  assert_eq(evidence.host.length(), 128)
}
