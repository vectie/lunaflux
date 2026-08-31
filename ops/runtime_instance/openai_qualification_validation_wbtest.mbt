///|
test "qualification provenance never grants production readiness" {
  assert_eq(health_for_phase(OpenAiQualificationReady), RuntimeHealthy)
  assert_eq(
    readiness_for_phase(OpenAiQualificationReady, true),
    RuntimeNotReady,
  )
}

///|
test "qualification admits exactly one OpenAI server or connection pool" {
  assert_true(openai_qualification_ingress_shape_valid(0, 0, 1, 0, 0))
  assert_true(openai_qualification_ingress_shape_valid(0, 0, 0, 1, 0))
  assert_false(openai_qualification_ingress_shape_valid(0, 0, 0, 0, 0))
  assert_false(openai_qualification_ingress_shape_valid(0, 0, 1, 1, 0))
  assert_false(openai_qualification_ingress_shape_valid(1, 0, 1, 0, 0))
  assert_false(openai_qualification_ingress_shape_valid(0, 1, 0, 1, 0))
  assert_false(openai_qualification_ingress_shape_valid(0, 0, 1, 0, 1))
}

///|
test "qualification singleton server rejects terminal or unhealthy state" {
  assert_true(
    openai_qualification_server_live_state_valid(
      LunaOnlineTcpServerListening,
      LunaOnlineTcpServerHealthy,
    ),
  )
  assert_true(
    openai_qualification_server_live_state_valid(
      LunaOnlineTcpServerConnected,
      LunaOnlineTcpServerHealthy,
    ),
  )
  assert_false(
    openai_qualification_server_live_state_valid(
      LunaOnlineTcpServerDraining,
      LunaOnlineTcpServerHealthy,
    ),
  )
  assert_false(
    openai_qualification_server_live_state_valid(
      LunaOnlineTcpServerClosed,
      LunaOnlineTcpServerHealthy,
    ),
  )
  assert_false(
    openai_qualification_server_live_state_valid(
      LunaOnlineTcpServerListening,
      LunaOnlineTcpServerFailed,
    ),
  )
}

///|
test "qualification pool live state requires truthful connection balance" {
  assert_true(
    openai_qualification_pool_live_state_valid(
      LunaOnlineOpenAIConnectionPoolListening,
      0,
      LunaOnlineOpenAIConnectionPoolReady,
      LunaOnlineTcpServerHealthy,
    ),
  )
  assert_true(
    openai_qualification_pool_live_state_valid(
      LunaOnlineOpenAIConnectionPoolActive,
      1,
      LunaOnlineOpenAIConnectionPoolReady,
      LunaOnlineTcpServerHealthy,
    ),
  )
  assert_false(
    openai_qualification_pool_live_state_valid(
      LunaOnlineOpenAIConnectionPoolListening,
      1,
      LunaOnlineOpenAIConnectionPoolReady,
      LunaOnlineTcpServerHealthy,
    ),
  )
  assert_false(
    openai_qualification_pool_live_state_valid(
      LunaOnlineOpenAIConnectionPoolActive,
      0,
      LunaOnlineOpenAIConnectionPoolReady,
      LunaOnlineTcpServerHealthy,
    ),
  )
  assert_false(
    openai_qualification_pool_live_state_valid(
      LunaOnlineOpenAIConnectionPoolDraining,
      0,
      LunaOnlineOpenAIConnectionPoolReady,
      LunaOnlineTcpServerHealthy,
    ),
  )
  assert_false(
    openai_qualification_pool_live_state_valid(
      LunaOnlineOpenAIConnectionPoolClosed,
      0,
      LunaOnlineOpenAIConnectionPoolReady,
      LunaOnlineTcpServerHealthy,
    ),
  )
  assert_false(
    openai_qualification_pool_live_state_valid(
      LunaOnlineOpenAIConnectionPoolListening,
      0,
      LunaOnlineOpenAIConnectionPoolNotReady,
      LunaOnlineTcpServerHealthy,
    ),
  )
  assert_false(
    openai_qualification_pool_live_state_valid(
      LunaOnlineOpenAIConnectionPoolListening,
      0,
      LunaOnlineOpenAIConnectionPoolReady,
      LunaOnlineTcpServerFailed,
    ),
  )
}

///|
fn openai_qualification_test_metric_sample(
  admissions : UInt64,
  completions : UInt64,
  prompt_tokens : UInt64,
  generated_tokens : UInt64,
  network_accepts : UInt64,
  network_disconnects : UInt64,
  network_rejections : UInt64,
) -> OpenAiQualificationMetricSample {
  {
    admissions,
    completions,
    cancellations: 0UL,
    deadlines: 0UL,
    failures: 0UL,
    prompt_tokens,
    generated_tokens,
    worker_failures: 0UL,
    network_accepts,
    network_disconnects,
    network_rejections,
    queue_depth: 0,
    active_requests: 0,
    kv_pages_used: 0,
    kv_pages_free: 32,
  }
}

///|
test "qualification initial metrics account exactly for embedded pool probes" {
  let server = openai_qualification_test_metric_sample(
    0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
  )
  let pool = openai_qualification_test_metric_sample(
    0UL, 0UL, 0UL, 0UL, 2UL, 2UL, 0UL,
  )
  let unbalanced_pool = openai_qualification_test_metric_sample(
    0UL, 0UL, 0UL, 0UL, 2UL, 1UL, 0UL,
  )
  assert_true(openai_qualification_initial_metrics_valid(server, 0UL))
  assert_true(openai_qualification_initial_metrics_valid(pool, 2UL))
  assert_false(openai_qualification_initial_metrics_valid(pool, 0UL))
  assert_false(openai_qualification_initial_metrics_valid(unbalanced_pool, 2UL))
}

///|
test "qualification terminal metrics require exact deltas from pool baseline" {
  let initial = openai_qualification_test_metric_sample(
    0UL, 0UL, 0UL, 0UL, 2UL, 2UL, 0UL,
  )
  let terminal = openai_qualification_test_metric_sample(
    1UL, 1UL, 2UL, 2UL, 5UL, 5UL, 2UL,
  )
  let missing_probe_baseline = openai_qualification_test_metric_sample(
    1UL, 1UL, 2UL, 2UL, 3UL, 3UL, 2UL,
  )
  let excess_rejection = openai_qualification_test_metric_sample(
    1UL, 1UL, 2UL, 2UL, 5UL, 5UL, 3UL,
  )
  assert_true(openai_qualification_terminal_metrics_valid(terminal, initial))
  assert_false(
    openai_qualification_terminal_metrics_valid(missing_probe_baseline, initial),
  )
  assert_false(
    openai_qualification_terminal_metrics_valid(excess_rejection, initial),
  )
}

///|
test "qualification contract rejects missing and header-hostile credentials" {
  let credentials : FixedArray[FixedArray[Byte]] = [
    [],
    [b'a', b'\r'],
    [b'a', b'\n'],
    [b'a', b' '],
  ]
  for credential in credentials {
    try
      RuntimeOpenAiQualificationValidationContract::new(credential, 1, 1, 1024)
    catch {
      RuntimeOpenAiQualificationValidationFailed(OpenAiQualificationContract) =>
        ()
      _ => abort("unexpected qualification contract rejection")
    } noraise {
      _ => abort("hostile qualification credential admitted")
    }
  }
}

///|
test "Responses evidence markers must remain in canonical order" {
  assert_true(
    qualification_markers_ordered(
      "response.created response.output_text.delta response.usage " +
      "response.completed data: [DONE]",
    ),
  )
  assert_false(
    qualification_markers_ordered(
      "response.created response.completed response.output_text.delta " +
      "response.usage data: [DONE]",
    ),
  )
}
