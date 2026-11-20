///|
async fn main {
  try run_qwen3_candidate_export(@env.args()) catch {
    error => {
      // Complete the diagnostic write before aborting: buffered stdout is
      // otherwise lost when a caller captures this offline export failure.
      @stdio.stderr.write(
        "outcome=failed\nerror=" + candidate_export_error_name(error) + "\n",
      )
      abort("Qwen3 BF16 candidate export failed")
    }
  } noraise {
    _ => ()
  }
}
