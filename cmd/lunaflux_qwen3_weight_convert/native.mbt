///|
#borrow(source_path, output_path, header, source_regions)
extern "c" fn raw_convert_qwen3_numeric_file(
  source_path : Bytes,
  output_path : Bytes,
  header : Bytes,
  source_regions : Bytes,
  chunk_bytes : Int,
) -> Int = "lunaflux_convert_qwen3_numeric_file"

///|
fn publish_qwen3_numeric_file(
  source_path : String,
  output_path : String,
  plan : Qwen3NumericConversionPlan,
) -> Unit raise Qwen3WeightConvertError {
  let status = raw_convert_qwen3_numeric_file(
    @utf8.encode(source_path),
    @utf8.encode(output_path),
    plan.header,
    plan.source_regions,
    1024 * 1024,
  )
  if status != 0 {
    raise Publication(status~)
  }
}
