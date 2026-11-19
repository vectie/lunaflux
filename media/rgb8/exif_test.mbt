///|
fn tiff(orientation : Int32, little : Bool) -> Bytes {
  (if little {
    b"II\x2a\x00\x08\x00\x00\x00\x01\x00\x12\x01\x03\x00\x01\x00\x00\x00"
  } else {
    b"MM\x00\x2a\x00\x00\x00\x08\x00\x01\x01\x12\x00\x03\x00\x00\x00\x01"
  }) +
  (if little {
    Bytes::from_array([orientation.to_byte(), 0, 0, 0])
  } else {
    Bytes::from_array([0, orientation.to_byte(), 0, 0])
  }) +
  b"\x00\x00\x00\x00"
}

///|
fn png_exif(image : Bytes, metadata : Bytes) -> Bytes {
  let output = @buffer.Buffer()
  output.write_bytes(image[:33])
  let length = metadata.length().reinterpret_as_uint()
  for shift in [24, 16, 8, 0] {
    output.write_byte((length >> shift).to_byte())
  }
  let all = b"eXIf" + metadata
  output.write_bytes(all)
  let mut crc = 0xffffffffU
  for byte in all {
    crc = crc ^ byte.to_uint()
    for _ in 0..<8 {
      crc = if (crc & 1U) == 1U { (crc >> 1) ^ 0xedb88320U } else { crc >> 1 }
    }
  }
  crc = crc ^ 0xffffffffU
  for shift in [24, 16, 8, 0] {
    output.write_byte((crc >> shift).to_byte())
  }
  output.write_bytes(image[33:])
  output.to_bytes()
}

///|
async test "oriented PNG covers EXIF orders while raw decoding stays unchanged" {
  let bytes = @fs.read_file("media/rgb8/fixtures/rgb.png").binary()
  for little in [false, true] {
    for orientation in 1..<=8 {
      let encoded = png_exif(bytes, tiff(orientation, little))
      let raw = @rgb8.Image::decode(
        encoded,
        maximum_encoded_bytes=4096,
        maximum_pixels=2,
      )
      assert_eq((raw.width(), raw.height()), (2, 1))
      assert_eq(raw.data(), b"\xff\x00\x00\x00\xff\x00")
      let image = @rgb8.Image::decode_oriented(
        encoded,
        maximum_encoded_bytes=4096,
        maximum_pixels=2,
      )
      assert_eq(
        (image.width(), image.height()),
        if orientation >= 5 {
          (1, 2)
        } else {
          (2, 1)
        },
      )
      assert_eq(
        image.data(),
        if [2, 3, 7, 8].contains(orientation) {
          b"\x00\xff\x00\xff\x00\x00"
        } else {
          raw.data()
        },
      )
    }
  }
}

///|
async test "oriented JPEG reads APP1 metadata and ignores malformed or unknown orientation" {
  let jpeg = @fs.read_file("media/rgb8/fixtures/baseline.jpg").binary()
  for little in [false, true] {
    let metadata = b"Exif\x00\x00" + tiff(6, little)
    let length = metadata.length() + 2
    let encoded = jpeg[:2].to_owned() +
      b"\xff\xe1" +
      Bytes::from_array([(length >> 8).to_byte(), length.to_byte()]) +
      metadata +
      jpeg[2:].to_owned()
    let image = @rgb8.Image::decode_oriented(
      encoded,
      maximum_encoded_bytes=4096,
      maximum_pixels=64,
    )
    assert_eq(
      (image.width(), image.height(), image.data().length()),
      (8, 8, 192),
    )
  }
  let png = @fs.read_file("media/rgb8/fixtures/rgb.png").binary()
  let valid = tiff(6, true)
  for length in 0..<valid.length() {
    let image = @rgb8.Image::decode_oriented(
      png_exif(png, valid[:length].to_owned()),
      maximum_encoded_bytes=4096,
      maximum_pixels=2,
    )
    assert_eq(image.width() * image.height(), 2)
    assert_eq(image.data().length(), 6)
  }
  for orientation in [0, 9] {
    let image = @rgb8.Image::decode_oriented(
      png_exif(png, tiff(orientation, false)),
      maximum_encoded_bytes=4096,
      maximum_pixels=2,
    )
    assert_eq((image.width(), image.height()), (2, 1))
  }
}
