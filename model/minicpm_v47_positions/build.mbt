///|
/// Build one compacted prompt using the selected model's canvas law. Crop grids
/// are in occurrence order within each modality. Modality comes from the
/// processor, never from guessing placeholder token IDs. Vision execution and
/// image preprocessing remain separate owners.
pub fn Sequence::build(
  spec : @model.Spec,
  tokens : ArrayView[Int32],
  modalities : ArrayView[Modality],
  image_grids~ : ArrayView[Grid],
  video_grids~ : ArrayView[Grid],
  downsample? : @model.Downsample,
) -> Self raise {
  let capacity = spec.decoder().context_length()
  guard tokens.length() > 0 &&
    tokens.length() <= capacity &&
    modalities.length() == tokens.length() else {
    fail("invalid prompt modalities")
  }
  let merge = match downsample.unwrap_or(spec.downsample()) {
    FourX => 2
    SixteenX => 4
  }
  let markers = spec.markers()
  let groups = visual_groups(tokens, modalities, markers)
  let image_index = Ref(0)
  let video_index = Ref(0)
  let next_grid = fn(modality) raise {
    let (grids, index) = match modality {
      Image => (image_grids, image_index)
      Video => (video_grids, video_index)
      Text => fail("text crop has no visual grid")
    }
    guard grids.get(index.val) is Some(grid) else {
      fail("missing visual crop grid")
    }
    index.val += 1
    merged_grid(grid, merge, capacity)
  }
  let coordinates = Array::make(tokens.length() * 3, 0)
  let mut current_position = 0
  let mut current_index = 0
  for group in groups {
    let group_end = frame_end(group.frames.last().unwrap(), tokens, markers)
    if group.start > current_index {
      fill_scalar(coordinates, current_index, group.start, current_position)
      current_position += group.start - current_index
    }
    let mut frame_cursor = group.start
    for frame in group.frames {
      let thumbnail = next_grid(frame.modality)
      let frame_start = frame.thumbnail.start - 1
      if frame_start > frame_cursor {
        fill_scalar(coordinates, frame_cursor, frame_start, current_position)
        current_position += frame_start - frame_cursor + 1
      }
      let end = frame_end(frame, tokens, markers).min(group_end)
      let origin = current_position
      let halo = (origin - 1).max(0)
      let mut canvas = thumbnail
      let mut slice_grid = (0, 0)
      let mut columns = 0
      if !frame.slices.is_empty() {
        columns = frame.slices.length()
        for k in 0..<(frame.slices.length() - 1) {
          if frame.slices[k + 1].start - frame.slices[k].end > 2 {
            columns = k + 1
            break
          }
        }
        let mut rows = frame.slices.length() / columns
        if rows * columns != frame.slices.length() {
          rows = 1
          columns = frame.slices.length()
        }
        slice_grid = next_grid(frame.modality)
        let height = rows.to_int64() * slice_grid.0.to_int64()
        let width = columns.to_int64() * slice_grid.1.to_int64()
        guard height <= capacity.to_int64() && width <= capacity.to_int64() else {
          fail("canvas exceeds position capacity")
        }
        canvas = (height.to_int(), width.to_int())
      }
      for token in frame_start..<end {
        for axis in 0..<3 {
          coordinates[token * 3 + axis] = origin
        }
      }
      coordinates[frame_start * 3 + 1] = halo
      coordinates[frame_start * 3 + 2] = halo
      if frame.thumbnail.end < end {
        coordinates[frame.thumbnail.end * 3 + 1] = origin + canvas.0
        coordinates[frame.thumbnail.end * 3 + 2] = origin + canvas.1
      }
      fill_grid(coordinates, frame.thumbnail, thumbnail, origin, canvas, (0, 0))
      for k, span in frame.slices {
        let offset = (k / columns * slice_grid.0, k % columns * slice_grid.1)
        let grid = if k == 0 { slice_grid } else { next_grid(frame.modality) }
        let start_marker = span.start - 1
        if start_marker >= frame_start {
          coordinates[start_marker * 3 + 1] = origin + offset.0
          coordinates[start_marker * 3 + 2] = origin + offset.1
        }
        if span.end < end {
          coordinates[span.end * 3 + 1] = origin + offset.0 + grid.0 - 1
          coordinates[span.end * 3 + 2] = origin + offset.1 + grid.1 - 1
        }
        fill_grid(coordinates, span, grid, origin, grid, offset)
      }
      for k in 0..<(frame.slices.length() - 1) {
        let gap_start = frame.slices[k].end
        let gap_end = frame.slices[k + 1].start
        if gap_end - gap_start > 2 {
          let boundary_h = (k / columns + 1) * slice_grid.0 - 1
          let right_w = columns * slice_grid.1
          for newline in (gap_start + 1)..<(gap_end - 1) {
            coordinates[newline * 3 + 1] = origin + boundary_h
            coordinates[newline * 3 + 2] = origin + right_w
          }
        }
      }
      current_position = origin + canvas.0.max(canvas.1) + 1
      frame_cursor = end
    }
    if frame_cursor < group_end {
      fill_scalar(coordinates, frame_cursor, group_end, current_position)
      current_position += group_end - frame_cursor
    }
    current_index = group_end
  }
  if current_index < tokens.length() {
    fill_scalar(coordinates, current_index, tokens.length(), current_position)
  }
  guard image_index.val == image_grids.length() &&
    video_index.val == video_grids.length() else {
    fail("unused visual crop grids")
  }
  let mut maximum = 0
  for position in coordinates {
    guard position >= 0 && position < capacity else {
      fail("canvas coordinate exceeds rotary capacity")
    }
    maximum = maximum.max(position)
  }
  {
    coordinates: ReadOnlyArray::from_array(coordinates),
    delta: maximum + 1 - tokens.length(),
    capacity,
  }
}
