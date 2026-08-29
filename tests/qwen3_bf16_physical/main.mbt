///|
async fn main {
  let args = @env.args()
  qwen3_require(args.length() >= 2, "missing Qwen physical mode")
  match args[1] {
    "export" => qwen3_export(args)
    "materialize" => qwen3_materialize(args)
    "probe" => qwen3_probe(args)
    "serve-check" => qwen3_serve_check(args)
    "metrics-check" => qwen3_metrics_check(args)
    "plan-check" => qwen3_plan_check(args)
    _ => raise InvalidInput("unknown Qwen physical mode")
  }
}
