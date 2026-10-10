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
