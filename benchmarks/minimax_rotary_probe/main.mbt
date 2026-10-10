///|
/// Offline diagnostic only. This calls the production pure table planner,
/// without uploading any weights or acquiring a CUDA context.
async fn main {
  guard @env.args() is [_, config_path, rows, output] else {
    fail("ORIGINAL_ENCODER_CONFIG ROWS NEW_OUTPUT_DIRECTORY")
  }
  let rows = @string.parse_int(rows)
  guard rows > 0 && rows <= 1024 else { fail("bounded table probe required") }
  let config = @encoder.parse_config(
    @fs.read_file(config_path).binary(),
    max_bytes=1048576,
  )
  let tables = config.rotary_tables(
    @encoder.text_positions(rows)[:],
    max_bytes=rows.to_int64() * 128L * 4L,
  )
  @fs.mkdir(output)
  @fs.write_file(output + "/cosine.bf16", tables.cosine, create_mode=CreateNew)
  @fs.write_file(output + "/sine.bf16", tables.sine, create_mode=CreateNew)
  println("native_rotary_tables_complete rows=\{rows} device=none")
}
