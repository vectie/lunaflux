///|
async fn run_i8_physical_probe() -> Unit {
  let args = @env.args()
  require_probe(
    args.length() == 4,
    "usage: i8_weight_scale_cuda_probe CUBIN sm_89|sm_90|sm_120 CYCLES",
  )
  let cycles = @string.parse_int(args[3], base=10) catch {
    _ => raise I8PhysicalProbeFailure("CYCLES must be an integer")
  }
  require_probe(cycles > 0 && cycles <= 128, "CYCLES must be in 1..128")
  let devices = match @device.probe() {
    Available(devices~, ..) => devices
    Unavailable(_) => raise I8PhysicalProbeFailure("CUDA driver is unavailable")
  }
  require_probe(devices.length() > 0, "no CUDA device is available")
  let capability = devices[0].capability()
  require_probe(
    (capability.compute_major() == 8 && capability.compute_minor() == 9) ||
    (capability.compute_major() == 9 && capability.compute_minor() == 0) ||
    (capability.compute_major() == 12 && capability.compute_minor() == 0),
    "production I8 v1 requires exact sm89, sm90, or sm120 device 0",
  )
  require_probe(
    capability.supports_bf16(),
    "device 0 has no BF16 boundary support",
  )
  let observed_target = "sm_" +
    capability.compute_major().to_string() +
    capability.compute_minor().to_string()
  require_probe(
    args[2] == observed_target,
    "offline CUBIN target does not equal observed device capability",
  )
  let module_bytes = @fs.read_file(args[1]).binary()
  let admission = admit_i8_probe(module_bytes, capability)
  let maximum_error = run_i8_probe(admission, cycles)
  println(
    "outcome=i8-weight-scale-physical-pass target=\{observed_target} cycles=\{cycles} rows=3 inner=4 outputs=4 abi=bf16-i8-f32scale-bf16 catalog=v4 capture=required max_abs_error=\{maximum_error} resources=closed scope=single-output-projection",
  )
}

///|
async fn main {
  run_i8_physical_probe() catch {
    I8PhysicalProbeFailure(reason) => {
      println("outcome=i8-physical-rejected reason=\{reason}")
      raise I8PhysicalProbeFailure(reason)
    }
    error => raise error
  }
}
