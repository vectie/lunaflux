///|
test "worker graph telemetry rejects missing executor without fabricating data" {
  let resources : DeviceWorkerResources = {
    executor: None,
    weights: None,
    context: None,
    numeric_weights_owned_by_executor: false,
  }
  try retained_graph_runtime_telemetry(resources) catch {
    Invalid(ResourceState) => ()
    _ => fail("unexpected missing graph telemetry error")
  } noraise {
    _ => fail("worker fabricated graph telemetry without an executor")
  }
}
