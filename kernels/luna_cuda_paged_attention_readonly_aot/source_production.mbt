///|
/// Distinct canary-free production source path. The common numerical body is
/// emitted without a diagnostic ABI argument or device-side publication.
fn readonly_production_cuda_source(
  symbol : String,
  reference : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering,
  input_row_width : Int,
) -> Bytes {
  readonly_cuda_source(symbol, reference, input_row_width, None)
}
