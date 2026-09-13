///|
fn cuda_query_tile_binding(
  abi : CudaPagedAttentionTileAbi,
  query_tile : Int,
) -> String raise CudaAttentionTileSourceError {
  if abi.query_metadata {
    let layout = @luna_attention_metadata.AttentionMetadataLayout::new(
      maximum_rows=abi.maximum_query_rows,
      maximum_tokens=abi.maximum_query_tokens,
    ).unwrap()
    guard layout.bucket(query_tile) is Some((bucket, offset, capacity)) else {
      raise InvalidCudaAttentionTileSource(UnsupportedLowering)
    }
    return (
      $|    // attention-tile-metadata-v1: offsets is a typed metadata view.
      $|    const int tiles=offsets[\{bucket}];
      $|    if (tiles<0 || tiles>\{capacity} || tile>=tiles) return;
      $|    const int *record=offsets+\{offset}+tile*8;
      $|    const int token_base=record[1], begin=record[2], end=record[3];
      $|    const int page_begin=record[5];
      $|    const int active=min(LF_QUERY_TILE,end-token_base);
      $|    if (thread==0) { control[0]=1; control[1]=record[7]; }
    )
  }
  let page_tokens = abi.tokens_per_page
  (
    $|    int row = -1, token_base = 0, prefix = 0;
    $|    for (int r = 0; r < rows; ++r) {
    $|      const int begin = offsets[r], end = offsets[r+1];
    $|      const int tiles = end > begin ? (end-begin+LF_QUERY_TILE-1)/LF_QUERY_TILE : 0;
    $|      if (tile < prefix + tiles) { row = r; token_base = begin + (tile-prefix)*LF_QUERY_TILE; break; }
    $|      prefix += tiles;
    $|    }
    $|    if (row < 0) return;
    $|    const int begin = offsets[row], end = offsets[row+1], length = lengths[row];
    $|    const int page_begin = page_offsets[row], page_end = page_offsets[row+1];
    $|    if (begin < 0 || end <= begin || end > tokens || length <= 0 ||
    $|        page_begin < 0 || page_end <= page_begin || page_end > counts[4] ||
    $|        page_end-page_begin != (length+\{page_tokens-1})/\{page_tokens}) return;
    $|    const int active = min(LF_QUERY_TILE, end-token_base);
    $|    if (thread == 0) {
    $|      control[0] = 1; control[1] = 0;
    $|      for (int i=0; i<active; ++i) {
    $|        const int position = positions[token_base+i];
    $|        if (position < 0 || position >= length) control[0] = 0;
    $|        control[1] = max(control[1], position+1);
    $|      }
    $|    }
  )
}
