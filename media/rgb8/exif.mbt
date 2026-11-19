///|
fn exif_u16(data : BytesView, offset : Int32, little : Bool) -> Int32? {
  guard offset >= 0 && offset <= data.length() - 2 else { return None }
  let a = data[offset].to_int()
  let b = data[offset + 1].to_int()
  Some(if little { a + b * 256 } else { a * 256 + b })
}

///|
fn exif_u32(data : BytesView, offset : Int32, little : Bool) -> Int64? {
  guard exif_u16(data, offset, little) is Some(a) &&
    exif_u16(data, offset + 2, little) is Some(b) else {
    return None
  }
  Some(
    if little {
      a.to_int64() + b.to_int64() * 65536L
    } else {
      a.to_int64() * 65536L + b.to_int64()
    },
  )
}

///|
fn tiff_orientation(data : BytesView) -> Int32 {
  guard data.length() >= 8 else { return 1 }
  let little = if data[:2] == b"II"[:] {
    true
  } else if data[:2] == b"MM"[:] {
    false
  } else {
    return 1
  }
  guard exif_u16(data, 2, little) == Some(42) &&
    exif_u32(data, 4, little) is Some(start) &&
    start >= 8L &&
    start <= (data.length() - 2).to_int64() &&
    exif_u16(data, start.to_int(), little) is Some(count) else {
    return 1
  }
  for i in 0..<count {
    let offset = start.to_int() + 2 + i * 12
    guard offset <= data.length() - 12 else { return 1 }
    if exif_u16(data, offset, little) != Some(274) {
      continue
    }
    guard exif_u32(data, offset + 4, little) == Some(1L) else { return 1 }
    let value = match exif_u16(data, offset + 2, little) {
      Some(3) =>
        exif_u16(data, offset + 8, little).map(value => value.to_int64())
      Some(4) => exif_u32(data, offset + 8, little)
      _ => None
    }
    if value is Some(value) && value >= 1L && value <= 8L {
      return value.to_int()
    }
    return 1
  }
  1
}

///|
fn encoded_orientation(data : BytesView) -> Int32 {
  if data.length() >= 2 && data[:2] == b"\xff\xd8"[:] {
    let mut offset = 2
    while offset < data.length() {
      guard data[offset] == 0xff else { return 1 }
      while offset < data.length() && data[offset] == 0xff {
        offset += 1
      }
      guard offset < data.length() else { return 1 }
      let marker = data[offset].to_int()
      offset += 1
      if marker == 0xda || marker == 0xd9 {
        return 1
      }
      if marker == 0xd8 || (marker >= 0xd0 && marker <= 0xd7) {
        continue
      }
      guard exif_u16(data, offset, false) is Some(length) &&
        length >= 2 &&
        length <= data.length() - offset else {
        return 1
      }
      let payload = offset + 2
      let end = offset + length
      if marker == 0xe1 &&
        end - payload >= 6 &&
        data[payload:payload + 6] == b"Exif\x00\x00"[:] {
        return tiff_orientation(data[payload + 6:end])
      }
      offset = end
    }
  } else if data.length() >= 8 && data[:8] == b"\x89PNG\r\n\x1a\n"[:] {
    let mut offset = 8
    while offset <= data.length() - 12 {
      guard exif_u32(data, offset, false) is Some(length) &&
        length <= (data.length() - offset - 12).to_int64() else {
        return 1
      }
      let end = offset + 8 + length.to_int()
      if data[offset + 4:offset + 8] == b"eXIf"[:] {
        return tiff_orientation(data[offset + 8:end])
      }
      if data[offset + 4:offset + 8] == b"IEND"[:] {
        return 1
      }
      offset = end + 4
    }
  }
  1
}

///|
/// PNG/JPEG RGB8 with standard EXIF orientation applied before model resize.
/// Raw decode semantics remain available through Image::decode. TIFF IFD0
/// orientation is read from JPEG APP1 or PNG eXIf; an unknown or unreadable
/// orientation leaves the stored raster unchanged. ICC/gamma and XMP are not
/// interpreted.
pub fn Image::decode_oriented(
  encoded : Bytes,
  maximum_encoded_bytes~ : Int32,
  maximum_pixels~ : Int32,
) -> Self raise ImageError {
  let image = Image::decode(encoded, maximum_encoded_bytes~, maximum_pixels~)
  let orientation = encoded_orientation(encoded[:])
  if orientation == 1 {
    return image
  }
  let width = if orientation >= 5 { image.height } else { image.width }
  let height = if orientation >= 5 { image.width } else { image.height }
  let data = Bytes::makei(image.data.length(), i => {
    let pixel = i / 3
    let x = pixel % width
    let y = pixel / width
    let (sx, sy) = match orientation {
      2 => (image.width - 1 - x, y)
      3 => (image.width - 1 - x, image.height - 1 - y)
      4 => (x, image.height - 1 - y)
      5 => (y, x)
      6 => (y, image.height - 1 - x)
      7 => (image.width - 1 - y, image.height - 1 - x)
      _ => (image.width - 1 - y, x)
    }
    image.data[(sy * image.width + sx) * 3 + i % 3]
  })
  { width, height, data, }
}
