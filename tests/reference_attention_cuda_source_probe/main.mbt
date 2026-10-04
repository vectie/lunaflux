///|
fn main raise {
  guard @env.args() is [_, backend] else {
    fail("flash-prefill|flash-decode|infer-decode")
  }
  let backend : @reference.ReferenceAttentionBackend = match backend {
    "flash-prefill" => FlashAttentionPrefill
    "flash-decode" => FlashAttentionDecode
    "infer-decode" => FlashInferDecode
    _ => fail("backend")
  }
  let abi = @abi.CudaPagedAttentionTileAbi::new(
    query_heads=16,
    key_value_heads=8,
    head_dimension=128,
    tokens_per_page=8,
    input_row_width=4096,
    page_stride_values=8192L,
    physical_page_capacity=9216L,
    maximum_query_rows=32,
    maximum_query_tokens=2048,
    maximum_page_entries=9216,
  )
  let plan = @reference.ReferenceAttentionPlan::new(backend~, abi~)
  println(@utf8.decode(plan.source()))
  println("// symbol=" + plan.symbol())
  println("// numeric_law=" + plan.numeric_law())
  let (gx, gy, gz, bx, by, bz, shared) = plan.launch()
  println("// launch=\{gx},\{gy},\{gz},\{bx},\{by},\{bz},\{shared}")
}
