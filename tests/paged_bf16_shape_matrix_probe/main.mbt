///|
async fn main {
  let args = @env.args()
  require_matrix(
    args.length() == 9,
    "usage: paged_bf16_shape_matrix_probe CONTRACT POINTWISE QKV DENSE MLP LM_HEAD PAGED CYCLES",
  )
  let cycles = @string.parse_int(args[8], base=10) catch {
    _ => raise ShapeMatrixFailure("CYCLES must be an integer")
  }
  require_matrix(cycles > 0 && cycles <= 32, "CYCLES must be in 1..32")
  let devices = match @device.probe() {
    Available(devices~, ..) => devices
    Unavailable(_) => raise ShapeMatrixFailure("CUDA driver is unavailable")
  }
  require_matrix(devices.length() > 0, "no CUDA device is available")
  let capability = devices[0].capability()
  require_matrix(
    capability.compute_major() == 12 && capability.compute_minor() == 0,
    "shape matrix requires exact sm120 device 0",
  )
  require_matrix(capability.supports_bf16(), "device 0 does not support BF16")
  let module_paths = FixedArray::makei(6, index => args[index + 2])
  let (launches, maximum_error) = run_shape_matrix(
    args[1],
    module_paths,
    cycles,
  )
  println(
    "outcome=paged-bf16-shape-matrix-sm120-pass cycles=\{cycles} cases=5 graph_launches=\{launches} kernels=12 max_abs_error=\{maximum_error} capture=required artifacts=authenticated resources=closed scope=synthetic-shape-matrix",
  )
}
