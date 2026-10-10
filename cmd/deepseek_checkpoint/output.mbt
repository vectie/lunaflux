///|
// Terminal-only telemetry, after reverse-order resource teardown. Context close
// refuses live children: print only after explicit release succeeds, including
// cancellation unwinding. This is not a request-success or commit receipt.
fn close_rank_context(context : @device.Context, role : String) -> Unit raise {
  context.close()
  println("rank_cleanup: role=\{role} context_closed=true")
}

///|
/// Diagnostic rendering belongs after both request owners retire, never in
/// decode/verification submission. The token source is retained host storage.
async fn write_generated(
  directory : String,
  count : Int,
  reason : @generation.GenerationFinish,
  token_at : (Int) -> Int raise,
) -> Unit {
  let rendered = StringBuilder()
  for index in 0..<count {
    rendered <+ "\{token_at(index)}\n"
  }
  @fs.write_file(
    directory + "/tokens.txt",
    rendered.to_string(),
    create_mode=CreateNew,
  )
  println("generated=\{count} finish=\{Repr(reason)}")
}
