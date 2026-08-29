///|
async fn main {
  let args = @env.args()
  try {
    require_tensor_core_probe(args.length() >= 2, "missing probe mode")
    match args[1] {
      "export" => export_tensor_core_probe(args)
      "run" => run_tensor_core_probe(args)
      _ => raise LunaTensorCoreProbeRejected("probe mode must be export or run")
    }
  } catch {
    LunaTensorCoreProbeRejected(reason) => {
      println(
        "outcome=lunatile-tensor-core-qualification-rejected reason=\{reason}",
      )
      raise LunaTensorCoreProbeRejected(reason)
    }
    error => raise error
  }
}
