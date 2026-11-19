///|
const ABSOLUTE_MAX_DESCRIPTOR_BYTES : Int64 = 1048576L

///|
const ABSOLUTE_MAX_JSON_DEPTH : Int32 = 16

///|
pub fn RuntimeDescriptorFileLimits::new(
  max_descriptor_bytes~ : Int64,
  max_json_depth~ : Int32,
) -> RuntimeDescriptorFileLimits raise RuntimeDescriptorFileError {
  if max_descriptor_bytes <= 0L ||
    max_descriptor_bytes > ABSOLUTE_MAX_DESCRIPTOR_BYTES {
    raise InvalidLimit(DescriptorBytes)
  }
  if max_json_depth <= 0 || max_json_depth > ABSOLUTE_MAX_JSON_DEPTH {
    raise InvalidLimit(JsonDepth)
  }
  { max_descriptor_bytes, max_json_depth, }
}

///|
pub fn RuntimeDescriptorFileLimits::max_descriptor_bytes(
  self : RuntimeDescriptorFileLimits,
) -> Int64 {
  self.max_descriptor_bytes
}

///|
pub fn RuntimeDescriptorFileLimits::max_json_depth(
  self : RuntimeDescriptorFileLimits,
) -> Int32 {
  self.max_json_depth
}
