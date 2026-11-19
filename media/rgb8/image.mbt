///|
pub suberror ImageError {
  InvalidDimensions
  EncodedTooLarge
  InvalidEncoding
  UnsupportedPrecision
}

///|
pub struct Image {
  priv width : Int32
  priv height : Int32
  priv data : Bytes
}

///|
pub fn Image::from_rgb(
  data : Bytes,
  width~ : Int32,
  height~ : Int32,
  maximum_pixels~ : Int32,
) -> Self raise ImageError {
  guard width > 0 &&
    height > 0 &&
    maximum_pixels > 0 &&
    width.to_int64() * height.to_int64() <= maximum_pixels.to_int64() &&
    width.to_int64() * height.to_int64() * 3L == data.length().to_int64() else {
    raise InvalidDimensions
  }
  { width, height, data, }
}

///|
pub fn Image::width(self : Self) -> Int32 {
  self.width
}

///|
pub fn Image::height(self : Self) -> Int32 {
  self.height
}

///|
pub fn Image::data(self : Self) -> Bytes {
  self.data
}

///|
#borrow(encoded, dimensions)
extern "C" fn decode_native(
  encoded : Bytes,
  maximum_pixels : Int32,
  dimensions : FixedArray[Int32],
) -> Bytes = "lunaflux_rgb8_decode"

///|
/// In-memory 8-bit PNG/JPEG conversion. All C allocations are released before
/// returning; decoding is outside the token scheduling/execution path.
pub fn Image::decode(
  encoded : Bytes,
  maximum_encoded_bytes~ : Int32,
  maximum_pixels~ : Int32,
) -> Self raise ImageError {
  guard maximum_encoded_bytes > 0 && encoded.length() <= maximum_encoded_bytes else {
    raise EncodedTooLarge
  }
  guard maximum_pixels > 0 && maximum_pixels <= 715827882 else {
    raise InvalidDimensions
  }
  let dimensions = FixedArray::make(3, 0)
  let data = decode_native(encoded, maximum_pixels, dimensions)
  match dimensions[2] {
    0 =>
      Image::from_rgb(
        data,
        width=dimensions[0],
        height=dimensions[1],
        maximum_pixels~,
      )
    2 => raise InvalidDimensions
    3 => raise UnsupportedPrecision
    _ => raise InvalidEncoding
  }
}
