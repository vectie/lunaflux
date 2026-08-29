///|
async fn main {
  let config = bridge_config(@env.args()) catch {
    _ => {
      println("Qwen benchmark bridge startup rejected")
      @sys.exit(2)
      abort("unreachable after exit")
    }
  }
  bridge_run(config)
}
