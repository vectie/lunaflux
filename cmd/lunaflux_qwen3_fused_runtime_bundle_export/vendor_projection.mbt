///|
fn parse_vendor_projection_option(
  args : ArrayView[String],
) -> (ArrayView[String], Int) raise Qwen3FusedRuntimeBundleExportError {
  match args {
    [.. rest, "--prefill-vendor-projection-rows", text] => {
      let rows = @string.parse_int(text) catch { _ => raise InvalidArguments }
      if rows < 2 || rows > 1048576 {
        raise InvalidArguments
      }
      (rest, rows)
    }
    _ => (args, 0)
  }
}
