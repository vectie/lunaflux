///|
/// Causal temporal zero-padding and spatial reflect-padding convolution.
/// Last two tuple fields are output height/width; temporal padding is k-1.
pub fn conv3d_shape(
  frames : Int,
  height : Int,
  width : Int,
  kernel : Int,
  temporal_stride : Int,
  spatial_stride : Int,
  spatial_before : Int,
  spatial_after : Int,
) -> (Int, Int, Int) raise PlanError {
  guard frames > 0 &&
    height > 0 &&
    width > 0 &&
    kernel > 0 &&
    temporal_stride > 0 &&
    spatial_stride > 0 &&
    spatial_before >= 0 &&
    spatial_after >= 0 &&
    spatial_before < height &&
    spatial_before < width &&
    spatial_after < height &&
    spatial_after < width else {
    raise InvalidGeometry
  }
  let h = height.to_int64() +
    spatial_before.to_int64() +
    spatial_after.to_int64() -
    kernel.to_int64()
  let w = width.to_int64() +
    spatial_before.to_int64() +
    spatial_after.to_int64() -
    kernel.to_int64()
  guard h >= 0L &&
    w >= 0L &&
    h / spatial_stride.to_int64() < 2147483647L &&
    w / spatial_stride.to_int64() < 2147483647L else {
    raise InvalidGeometry
  }
  (
    (frames - 1) / temporal_stride + 1,
    (h / spatial_stride.to_int64() + 1L).to_int(),
    (w / spatial_stride.to_int64() + 1L).to_int(),
  )
}

///|
fn video_signature(
  op : Operation,
) -> (ReadOnlyArray[Int64], Int64) raise PlanError {
  match op {
    Zeros(elements~) => {
      guard elements > 0L && elements <= 2147483647L else {
        raise InvalidGeometry
      }
      ([], elements)
    }
    VideoSlice(
      frames~,
      height~,
      width~,
      start_frame~,
      start_y~,
      start_x~,
      tile_frames~,
      tile_height~,
      tile_width~
    ) => {
      guard start_frame >= 0 &&
        start_frame < frames &&
        start_y >= 0 &&
        start_y <= height - tile_height &&
        start_x >= 0 &&
        start_x <= width - tile_width else {
        raise InvalidGeometry
      }
      (
        [product([3, frames, height, width])],
        product([3, tile_frames, tile_height, tile_width]),
      )
    }
    Conv3d(
      input_channels~,
      output_channels~,
      frames~,
      height~,
      width~,
      kernel~,
      temporal_stride~,
      spatial_stride~,
      spatial_before~,
      spatial_after~
    ) => {
      let (t, h, w) = conv3d_shape(
        frames, height, width, kernel, temporal_stride, spatial_stride, spatial_before,
        spatial_after,
      )
      (
        [
          product([input_channels, frames, height, width]),
          product([output_channels, input_channels, kernel, kernel, kernel]),
          product([output_channels]),
        ],
        product([output_channels, t, h, w]),
      )
    }
    GroupNormSilu(channels~, frames~, height~, width~, groups~) => {
      guard groups > 0 && channels % groups == 0 else { raise InvalidGeometry }
      (
        [
          product([channels, frames, height, width]),
          product([channels]),
          product([channels]),
        ],
        product([channels, frames, height, width]),
      )
    }
    InsertTile(
      channels~,
      frames~,
      height~,
      width~,
      tile_height~,
      tile_width~,
      y~,
      x~,
      copy_height~,
      copy_width~,
      overlap_y~,
      overlap_x~
    ) => {
      guard y >= 0 &&
        x >= 0 &&
        copy_height > 0 &&
        copy_width > 0 &&
        copy_height <= tile_height &&
        copy_width <= tile_width &&
        y <= height - copy_height &&
        x <= width - copy_width &&
        overlap_y >= 0 &&
        overlap_y <= tile_height &&
        overlap_x >= 0 &&
        overlap_x <= tile_width else {
        raise InvalidGeometry
      }
      let canvas = product([channels, frames, height, width])
      let tile = product([channels, frames, tile_height, tile_width])
      ([canvas, tile, tile, tile], canvas)
    }
    InsertTime(
      channels~,
      frames~,
      height~,
      width~,
      tile_frames~,
      start~,
      count~
    ) => {
      guard start >= 0 &&
        count > 0 &&
        count <= tile_frames &&
        start <= frames - count else {
        raise InvalidGeometry
      }
      let canvas = product([channels, frames, height, width])
      ([canvas, product([channels, tile_frames, height, width])], canvas)
    }
    GaussianPatch(channels~, frames~, height~, width~) => {
      guard height % 2 == 0 && width % 2 == 0 else { raise InvalidGeometry }
      let output = product([channels, frames, height, width])
      (
        [
          product([2, channels, frames, height, width]),
          output,
          product([channels]),
          product([channels]),
        ],
        output,
      )
    }
    _ => raise InvalidGeometry
  }
}
