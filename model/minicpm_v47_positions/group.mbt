///|
fn is_marker(token : Int32, markers : @model.Markers) -> Bool {
  token == markers.slice_start ||
  token == markers.slice_end ||
  token == markers.image_start ||
  token == markers.image_end ||
  token == markers.newline
}

///|
fn visual_groups(
  tokens : ArrayView[Int32],
  modalities : ArrayView[Modality],
  markers : @model.Markers,
) -> Array[VisualGroup] raise {
  let groups : Array[VisualGroup] = []
  let mut previous_end = 0
  let mut cursor = 0
  while cursor < tokens.length() {
    let modality = modalities[cursor]
    if modality is Text {
      cursor += 1
      continue
    }
    let start = cursor
    while cursor < tokens.length() && modalities[cursor] == modality {
      cursor += 1
    }
    guard start > 0 &&
      cursor < tokens.length() &&
      modalities[start - 1] is Text &&
      modalities[cursor] is Text else {
      fail("visual crop needs text markers")
    }
    let span = { start, end: cursor, }
    if tokens[start - 1] == markers.slice_start &&
      groups.last() is Some(group) &&
      group.frames.last() is Some(frame) {
      guard frame.modality == modality else {
        fail("slice modality differs from thumbnail")
      }
      frame.slices.push(span)
    } else {
      guard tokens[start - 1] == markers.image_start else {
        fail("thumbnail needs image start marker")
      }
      let frame = { modality, thumbnail: span, slices: [], }
      if groups.last() is Some(group) &&
        tokens[previous_end:start - 1].all(token => is_marker(token, markers)) {
        group.frames.push(frame)
      } else {
        groups.push({ start: start - 1, frames: [frame], })
      }
    }
    previous_end = cursor
  }
  groups
}

///|
fn frame_end(
  frame : VisualFrame,
  tokens : ArrayView[Int32],
  markers : @model.Markers,
) -> Int32 {
  let mut end = match frame.slices.last() {
    Some(span) => span.end
    None => frame.thumbnail.end
  }
  // Preserve the reference's treatment of an opening marker for the next frame.
  while end < tokens.length() && is_marker(tokens[end], markers) {
    end += 1
  }
  end
}
