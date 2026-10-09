///|
// Select overlap once at startup. The inherited protocol and all admitted
// model, device, and artifact bindings are shared with the ordered rank child.
fn main {
  if !@child.run(parallel_policy=ComputeCommunicationOverlap) {
    @process.exit_failure()
  }
}
