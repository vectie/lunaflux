///|
async fn main {
  let args = @env.args()
  try {
    require_luna_tile_parallel_probe(args.length() >= 2, "missing probe mode")
    match args[1] {
      "export" => export_luna_tile_parallel_probe(args)
      "run" => run_luna_tile_parallel_probe(args)
      _ =>
        raise LunaTileParallelCudaProbeRejected(
          "probe mode must be export or run",
        )
    }
  } catch {
    LunaTileParallelCudaProbeRejected(reason) => {
      println(
        "outcome=lunatile-parallel-qualification-rejected reason=\{reason}",
      )
      raise LunaTileParallelCudaProbeRejected(reason)
    }
    error => raise error
  }
}
