///|
pub struct TiledDecoder {
  source : Bytes
  symbols : ReadOnlyArray[String]
  tile : TileDecoder
  temporal : TemporalTiles
  y : TileAxis
  x : TileAxis
  tile_count : Int
  input_bytes : Int64
  gather_bytes : Int64
  tile_store_bytes : Int64
  clip_bytes : Int64
  output_bytes : Int64
  output_height : Int
  output_width : Int
  workspace_bytes : Int64
}

///|
/// Pure startup plan. Default pixel tile and overlap match the released
/// decoder; caller may explicitly choose a different supported tile geometry.
pub fn TiledDecoder::new(
  latent_frames~ : Int,
  latent_height~ : Int,
  latent_width~ : Int,
  output_frames~ : Int,
  tile_size~ : Int,
  minimum_overlap~ : Int,
  prefix~ : String,
  max_jobs~ : Int,
  max_workspace_bytes~ : Int64,
) -> Self raise {
  guard latent_height > 0 &&
    latent_width > 0 &&
    latent_height <= 2147483647 / 16 &&
    latent_width <= 2147483647 / 16 &&
    max_jobs > 0 &&
    max_jobs <= 65536 else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  let height = latent_height * 16
  let width = latent_width * 16
  guard height.to_int64() * width.to_int64() <= 2147483647L / 84L else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  let temporal = TemporalTiles::new(latent_frames, output_frames~)
  let y = TileAxis::new(
    height,
    tile_size~,
    minimum_overlap~,
    max_tiles=max_jobs,
  )
  let x = TileAxis::new(width, tile_size~, minimum_overlap~, max_tiles=max_jobs)
  guard y.starts.length().to_int64() * x.starts.length().to_int64() <=
    max_jobs.to_int64() / temporal.chunks.to_int64() &&
    latent_frames.to_int64() * latent_height.to_int64() <=
    9223372036854775807L / 96L / latent_width.to_int64() &&
    output_frames.to_int64() * height.to_int64() <=
    9223372036854775807L / 12L / width.to_int64() else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  let tile_count = y.starts.length() * x.starts.length()
  let tile = TileDecoder::new(
    temporal=7,
    height=y.length / 16,
    width=x.length / 16,
    prefix=prefix + "_tile",
    max_workspace_bytes~,
  )
  let gather_bytes = tile.frontend.input_bytes
  let tile_store_bytes = tile.head.output_bytes * tile_count.to_int64()
  let clip_bytes = height.to_int64() * width.to_int64() * 84L * 4L
  let output_bytes = height.to_int64() *
    width.to_int64() *
    output_frames.to_int64() *
    12L
  let workspace_bytes = for
    bytes in [
      gather_bytes, tile_store_bytes, clip_bytes, clip_bytes, output_bytes,
    ]
    total = tile.workspace_bytes {
    guard bytes > 0L && bytes <= max_workspace_bytes - total else {
      raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
    }
    continue total + bytes
  } nobreak {
    total
  }
  let symbols = [prefix + "_copy", prefix + "_assemble"]
  let sources = [
    tile.source,
    @video_latent_embedding_cuda_source.tile_copy_source(
      elements=tile.head.output_bytes / 4L,
      symbol=symbols[0],
    ),
    @video_latent_embedding_cuda_source.spatial_assemble_source(
      height~,
      width~,
      tile_h=y.length,
      tile_w=x.length,
      y.starts[:],
      x.starts[:],
      symbol=symbols[1],
    ),
  ]
  for chunk in 0..<temporal.chunks {
    for row, start_y in y.starts {
      for col, start_x in x.starts {
        let symbol = prefix + "_gather_\{chunk}_\{row}_\{col}"
        symbols.push(symbol)
        sources.push(
          @video_latent_embedding_cuda_source.tile_gather_source(
            latent_t=latent_frames,
            latent_h=latent_height,
            latent_w=latent_width,
            start_t=chunk * 5,
            start_y=start_y / 16,
            start_x=start_x / 16,
            tile_h=y.length / 16,
            tile_w=x.length / 16,
            symbol~,
          ),
        )
      }
    }
  }
  for chunk in 0..<temporal.chunks {
    let symbol = prefix + "_publish_\{chunk}"
    symbols.push(symbol)
    sources.push(
      @video_latent_embedding_cuda_source.temporal_publish_source(
        height~,
        width~,
        chunk~,
        chunks=temporal.chunks,
        available_frames=temporal.available_frames,
        output_frames~,
        symbol~,
      ),
    )
  }
  let source = sources.fold(init=b"", (a, b) => a + b)
  {
    source,
    symbols: ReadOnlyArray::from_array(symbols),
    tile,
    temporal,
    y,
    x,
    tile_count,
    input_bytes: latent_frames.to_int64() *
    latent_height.to_int64() *
    latent_width.to_int64() *
    96L,
    gather_bytes,
    tile_store_bytes,
    clip_bytes,
    output_bytes,
    output_height: height,
    output_width: width,
    workspace_bytes,
  }
}
