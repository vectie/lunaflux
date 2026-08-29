///|
async fn main {
  match @env.args() {
    [_, "inspect-fused-production-runtime", runtime_path, expected_sha256] =>
      inspect_fused_production_runtime(runtime_path, expected_sha256)
    [_, "long-context", deployment_argument, expected_worker_sha256] =>
      run_long_context_campaign(deployment_argument, expected_worker_sha256)
    [_, "context-churn", deployment_argument, expected_worker_sha256] =>
      run_context_churn_campaign(deployment_argument, expected_worker_sha256)
    [_, "prefix-reuse", deployment_argument, expected_worker_sha256] =>
      run_prefix_campaign(deployment_argument, expected_worker_sha256)
    [_, "benchmark", deployment_argument, expected_worker_sha256] =>
      run_benchmark_campaign(deployment_argument, expected_worker_sha256)
    [
      _,
      "device-greedy",
      embedded_deployment,
      host_deployment,
      expected_worker_sha256,
    ] =>
      run_device_greedy_campaign(
        embedded_deployment, host_deployment, expected_worker_sha256, false,
      )
    [
      _,
      "device-greedy-fused-v2",
      embedded_deployment,
      host_deployment,
      expected_worker_sha256,
    ] =>
      run_device_greedy_campaign(
        embedded_deployment, host_deployment, expected_worker_sha256, true,
      )
    [_, "serving", deployment_argument, expected_worker_sha256] =>
      run_serving_campaign(deployment_argument, expected_worker_sha256)
    [_, "broad-serving", deployment_argument, expected_worker_sha256] =>
      run_broad_serving_campaign(deployment_argument, expected_worker_sha256)
    [
      _,
      "openai-qualification",
      deployment_argument,
      expected_worker_sha256,
      credential,
    ] =>
      run_openai_qualification_campaign(
        deployment_argument, expected_worker_sha256, credential,
      )
    [_, deployment_argument, expected_worker_sha256] =>
      run_campaign(deployment_argument, expected_worker_sha256)
    _ => raise UsageRejected
  }
}
