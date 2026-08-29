///|
async fn main {
  let args = @env.args()
  require_graph(
    args.length() == 8,
    "usage: paged_bf16_graph_probe POINTWISE QKV DENSE MLP LM_HEAD PAGED CYCLES",
  )
  let cycles = @string.parse_int(args[7], base=10) catch {
    _ => raise GraphProbeFailure("CYCLES must be an integer")
  }
  require_graph(cycles > 0 && cycles <= 128, "CYCLES must be in 1..128")
  let devices = match @device.probe() {
    Available(devices~, ..) => devices
    Unavailable(_) => raise GraphProbeFailure("CUDA driver is unavailable")
  }
  require_graph(devices.length() > 0, "no CUDA device is available")
  let capability = devices[0].capability()
  require_graph(
    capability.compute_major() == 12 && capability.compute_minor() == 0,
    "complete graph fixture requires exact sm120 device 0",
  )
  require_graph(capability.supports_bf16(), "device 0 does not support BF16")
  let module_paths = FixedArray::makei(6, index => args[index + 1])
  let (token, maximum_error) = run_captured_graph(module_paths, cycles)
  println(
    "outcome=paged-bf16-graph-sm120-pass cycles=\{cycles} kernels=12 mixed_rows=2 live_tokens=3 token=\{token} max_abs_error=\{maximum_error} capture=required resources=closed",
  )
}
