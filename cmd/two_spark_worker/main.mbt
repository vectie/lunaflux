///|
async fn main {
  let (root, digest, parallel_policy) = match @env.args() {
    [_, root, digest] => (root, digest, @parallel_ir.Ordered)
    [_, root, digest, "--overlap"] =>
      (root, digest, @parallel_ir.ComputeCommunicationOverlap)
    _ => {
      @stdio.stderr.write(
        "usage: two_spark_worker ABSOLUTE_DEPLOYMENT_ROOT CONFIG_SHA256 [--overlap]\n",
      )
      @process.exit_failure()
      return
    }
  }
  @entry.run(root, digest, parallel_policy~) catch {
    _ => {
      // Never print credentials, startup payloads, or private certificate paths.
      @stdio.stderr.write(
        "two-spark worker failed admission, transport, or cleanup\n",
      )
      @process.exit_failure()
    }
  }
}
