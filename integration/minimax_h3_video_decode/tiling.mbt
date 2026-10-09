///|
pub struct TileAxis {
  starts : ReadOnlyArray[Int]
  length : Int
  overlaps : ReadOnlyArray[Int]
}

///|
/// Pixel-space split_tiles from the released VAE, with overlap redistributed
/// in 16-pixel units. Whole-tile decoder attention must not see adjacent tiles.
pub fn TileAxis::new(
  pixels : Int,
  tile_size~ : Int,
  minimum_overlap~ : Int,
  max_tiles? : Int = 4096,
) -> Self raise @video_latent_embedding_cuda_source.EmbeddingError {
  guard pixels > 0 &&
    pixels % 16 == 0 &&
    tile_size > 0 &&
    tile_size % 16 == 0 &&
    minimum_overlap >= 0 &&
    minimum_overlap % 16 == 0 &&
    minimum_overlap < tile_size &&
    max_tiles > 0 else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  if pixels <= tile_size {
    return {
      starts: ReadOnlyArray::from_array([0]),
      length: pixels,
      overlaps: ReadOnlyArray::from_array([]),
    }
  }
  let count = ((
    pixels.to_int64() -
    minimum_overlap.to_int64() +
    tile_size.to_int64() -
    minimum_overlap.to_int64() -
    1L
  ) /
  (tile_size.to_int64() - minimum_overlap.to_int64())).to_int()
  guard count <= max_tiles else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  let remaining = tile_size.to_int64() * count.to_int64() -
    minimum_overlap.to_int64() * (count - 1).to_int64() -
    pixels.to_int64()
  let units = remaining / 16L
  let overlaps = Array::makei(count - 1, i => {
    minimum_overlap +
    (units / (count - 1).to_int64() +
    (if i.to_int64() < units % (count - 1).to_int64() { 1L } else { 0L })).to_int() *
    16
  })
  let starts : Array[Int] = [0]
  for overlap in overlaps {
    starts.push(starts[starts.length() - 1] + tile_size - overlap)
  }
  {
    starts: ReadOnlyArray::from_array(starts),
    length: tile_size,
    overlaps: ReadOnlyArray::from_array(overlaps),
  }
}

///|
pub struct TemporalTiles {
  latent_frames : Int
  pad_tokens : Int
  chunks : Int
  available_frames : Int
  output_frames : Int
  trim_front : Int
}

///|
/// Official clip_length=17, ratio=4, token_drop=3: seven-token clips stride
/// five; each contributes 17 frames after three padding frames, final tail five.
pub fn TemporalTiles::new(
  latent_frames : Int,
  output_frames~ : Int,
) -> Self raise @video_latent_embedding_cuda_source.EmbeddingError {
  guard latent_frames >= 3 && latent_frames <= 536870900 && output_frames > 0 else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  let pad_tokens = (5 - (latent_frames + 3) % 5) % 5
  let chunks = (latent_frames + 3 + pad_tokens) / 5 - 1
  guard chunks > 0 else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  let pad_frames = for k in 0..<pad_tokens; sum = 0 {
    continue sum + (if (latent_frames + k) % 5 == 0 { 1 } else { 4 })
  } nobreak {
    sum
  }
  let available_frames = chunks * 17 + 5 - pad_frames
  guard output_frames <= available_frames else {
    raise @video_latent_embedding_cuda_source.EmbeddingError::InvalidGeometry
  }
  {
    latent_frames,
    pad_tokens,
    chunks,
    available_frames,
    output_frames,
    trim_front: available_frames - output_frames,
  }
}
