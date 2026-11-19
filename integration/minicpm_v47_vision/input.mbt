///|
priv struct Upload {
  plane : Plane
  bytes : Bytes
}

///|
/// Immutable host media packet. Construction/resize/metadata are outside native
/// submission and decoder scheduling. Pixels are borrowed immutable Bytes.
pub struct Input {
  priv uploads : ReadOnlyArray[Upload]
  priv capacity : (Int32, Int32, Int32)
  priv crop_capacity : Int32
  priv rows : Int32
  priv outputs : Int32
}

///|
fn i32_bytes(values : ArrayView[Int32]) -> Bytes {
  let bytes = FixedArray::make(values.length() * 4, b'\x00')
  for i in 0..<values.length() {
    let word = values[i].reinterpret_as_uint()
    for lane in 0..<4 {
      bytes[i * 4 + lane] = ((word >> (lane * 8)) & 255U).to_byte()
    }
  }
  Bytes::from_array(bytes[:])
}

///|
pub fn Input::new(
  program : Program,
  layout : @vision.Layout,
  pixels : @processor.Pixels,
) -> Self raise {
  Input::from_images(program, layout, [pixels])
}

///|
fn pack_images(images : ArrayView[@processor.Pixels], rows : Int32) -> Bytes {
  if images.length() == 1 {
    return images[0].data()
  }
  let columns = rows * 14
  let packed = FixedArray::make(rows * 588 * 2, b'\x00')
  // Concatenate the final axis of [1,3,14,columns], never whole image blobs.
  for plane in 0..<42 {
    let mut cursor = plane * columns * 2
    for image in images {
      let bytes = image.data()
      let length = image.plan().pixel_columns() * 2
      for i in 0..<length {
        packed[cursor + i] = bytes[plane * length + i]
      }
      cursor += length
    }
  }
  Bytes::from_array(packed[:])
}

///|
/// Pack multiple image preparations in prompt occurrence order. All crops
/// retain independent attention spans; pixel concatenation is along NaViT's
/// final axis. One image reuses its immutable payload without a copy.
pub fn Input::from_images(
  program : Program,
  layout : @vision.Layout,
  images : ArrayView[@processor.Pixels],
) -> Self raise {
  let (raw_capacity, encoded_capacity, output_capacity) = program.rows
  let raw = layout.raw_spans().rows()
  let encoded = layout.encoded_spans().rows()
  let output = layout.output_rows()
  guard !images.is_empty() &&
    raw > 0 &&
    raw <= raw_capacity &&
    encoded == raw / (raw_capacity / encoded_capacity) &&
    output == encoded / 4 &&
    output <= output_capacity else {
    fail("pixels/layout/program geometry mismatch")
  }
  let mut crop = 0
  let mut pixel_rows = 0L
  let mut feature_rows = 0L
  for image in images {
    let grids = image.plan().target_sizes()
    let mut image_rows = 0L
    for grid in grids {
      guard crop < layout.grids().length() &&
        grid.height == layout.grids()[crop].height &&
        grid.width == layout.grids()[crop].width else {
        fail("pixel crop order differs from encoder layout")
      }
      crop += 1
      image_rows += grid.height.to_int64() * grid.width.to_int64()
    }
    guard image.data().length().to_int64() == image_rows * 588L * 2L else {
      fail("invalid prepared pixel extent")
    }
    pixel_rows += image_rows
    feature_rows += image.plan().visual_tokens().to_int64()
  }
  guard crop == layout.grids().length() &&
    pixel_rows == raw.to_int64() &&
    feature_rows == output.to_int64() else {
    fail("image count/rows differ from prepared layout")
  }
  let boundaries = layout.raw_spans().boundaries()
  for i in 0..<(boundaries.length() - 1) {
    guard boundaries[i + 1] - boundaries[i] <= program.crop_rows else {
      fail("image crop exceeds AOT attention segment capacity")
    }
  }
  let uploads = [
    { plane: RawCounts, bytes: i32_bytes([0, 0, 1, raw]), },
    { plane: EncodedCounts, bytes: i32_bytes([0, 0, 1, encoded]), },
    { plane: OutputCounts, bytes: i32_bytes([0, 0, 1, output]), },
    { plane: Pixels, bytes: pack_images(images, raw), },
    { plane: PositionIds, bytes: i32_bytes(layout.position_ids()[:]), },
    { plane: RawStarts, bytes: i32_bytes(layout.raw_spans().starts()[:]), },
    { plane: RawEnds, bytes: i32_bytes(layout.raw_spans().ends()[:]), },
    {
      plane: EncodedStarts,
      bytes: i32_bytes(layout.encoded_spans().starts()[:]),
    },
    { plane: EncodedEnds, bytes: i32_bytes(layout.encoded_spans().ends()[:]), },
    {
      plane: WindowStarts,
      bytes: i32_bytes(layout.window_spans().starts()[:]),
    },
    { plane: WindowEnds, bytes: i32_bytes(layout.window_spans().ends()[:]), },
    { plane: WindowIndices, bytes: i32_bytes(layout.window_indices()[:]), },
    { plane: MergerIndices, bytes: i32_bytes(layout.merger_indices()[:]), },
  ]
  {
    uploads: ReadOnlyArray::from_array(uploads),
    capacity: program.rows,
    crop_capacity: program.crop_rows,
    rows: raw,
    outputs: output,
  }
}

///|
pub fn Input::patch_rows(self : Self) -> Int32 {
  self.rows
}

///|
pub fn Input::feature_rows(self : Self) -> Int32 {
  self.outputs
}
