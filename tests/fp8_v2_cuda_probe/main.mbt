///|
async fn main {
  let args = @env.args()
  try {
    require_fp8_probe(args.length() >= 2, "missing mode")
    match args[1] {
      "export" => export_fp8_probe(args)
      "run" => run_fp8_probe(args)
      _ => raise Fp8V2CudaProbeError::Rejected("mode must be export or run")
    }
  } catch {
    Fp8V2CudaProbeError::Rejected(reason) => {
      println("outcome=fp8-v2-sm120-physical-rejected reason=\{reason}")
      raise Fp8V2CudaProbeError::Rejected(reason)
    }
    error => raise error
  }
}
