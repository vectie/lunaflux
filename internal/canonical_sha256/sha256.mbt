///|
let canonical_sha256_round_constants : ReadOnlyArray[UInt] = ReadOnlyArray::from_array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
  ],
)

///|
/// Fixed-scratch streaming SHA-256 for bounded canonical encoders.
pub struct CanonicalSha256 {
  priv state : FixedArray[UInt]
  priv block : FixedArray[Byte]
  priv schedule : FixedArray[UInt]
  priv mut block_length : Int
  priv mut byte_count : UInt64
  priv mut transformed_blocks : Int
  priv mut finalized : Bool
  priv mut cached_hex : String?
}

///|
pub fn CanonicalSha256::new() -> CanonicalSha256 {
  {
    state: [
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
      0x5be0cd19,
    ],
    block: FixedArray::make(64, 0),
    schedule: FixedArray::make(64, 0U),
    block_length: 0,
    byte_count: 0UL,
    transformed_blocks: 0,
    finalized: false,
    cached_hex: None,
  }
}

///|
fn rotate_right(value : UInt, amount : Int) -> UInt {
  (value >> amount) | (value << (32 - amount))
}

///|
fn CanonicalSha256::transform(self : CanonicalSha256) -> Unit {
  for index in 0..<16 {
    let offset = index * 4
    self.schedule[index] = (self.block[offset].to_uint() << 24) |
      (self.block[offset + 1].to_uint() << 16) |
      (self.block[offset + 2].to_uint() << 8) |
      self.block[offset + 3].to_uint()
  }
  for index in 16..<64 {
    let previous15 = self.schedule[index - 15]
    let previous2 = self.schedule[index - 2]
    let sigma0 = rotate_right(previous15, 7) ^
      rotate_right(previous15, 18) ^
      (previous15 >> 3)
    let sigma1 = rotate_right(previous2, 17) ^
      rotate_right(previous2, 19) ^
      (previous2 >> 10)
    self.schedule[index] = self.schedule[index - 16] +
      sigma0 +
      self.schedule[index - 7] +
      sigma1
  }
  let mut a = self.state[0]
  let mut b = self.state[1]
  let mut c = self.state[2]
  let mut d = self.state[3]
  let mut e = self.state[4]
  let mut f = self.state[5]
  let mut g = self.state[6]
  let mut h = self.state[7]
  for index in 0..<64 {
    let upper1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25)
    let choose = (e & f) ^ ((e ^ 0xffffffffU) & g)
    let temporary1 = h +
      upper1 +
      choose +
      canonical_sha256_round_constants[index] +
      self.schedule[index]
    let upper0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22)
    let majority = (a & b) ^ (a & c) ^ (b & c)
    let temporary2 = upper0 + majority
    h = g
    g = f
    f = e
    e = d + temporary1
    d = c
    c = b
    b = a
    a = temporary1 + temporary2
  }
  self.state[0] = self.state[0] + a
  self.state[1] = self.state[1] + b
  self.state[2] = self.state[2] + c
  self.state[3] = self.state[3] + d
  self.state[4] = self.state[4] + e
  self.state[5] = self.state[5] + f
  self.state[6] = self.state[6] + g
  self.state[7] = self.state[7] + h
  self.transformed_blocks += 1
}

///|
fn CanonicalSha256::append_data_byte(
  self : CanonicalSha256,
  byte : Byte,
) -> Unit {
  self.block[self.block_length] = byte
  self.block_length += 1
  if self.block_length == 64 {
    self.transform()
    self.block_length = 0
  }
}

///|
pub fn CanonicalSha256::update_byte(
  self : CanonicalSha256,
  byte : Byte,
) -> Bool {
  if self.finalized {
    return false
  }
  self.append_data_byte(byte)
  self.byte_count += 1UL
  true
}

///|
pub fn CanonicalSha256::update_bytes(
  self : CanonicalSha256,
  bytes : BytesView,
) -> Bool {
  if self.finalized {
    return false
  }
  for byte in bytes {
    self.append_data_byte(byte)
    self.byte_count += 1UL
  }
  true
}

///|
pub fn CanonicalSha256::update_u64_le(
  self : CanonicalSha256,
  value : UInt64,
) -> Bool {
  if self.finalized {
    return false
  }
  for shift in 0..<8 {
    self.append_data_byte((value >> (shift * 8)).to_byte())
    self.byte_count += 1UL
  }
  true
}

///|
fn CanonicalSha256::finalize_bytes(self : CanonicalSha256) -> FixedArray[Byte] {
  let padding_transforms = if self.byte_count % 64UL < 56UL { 1 } else { 2 }
  let expected_transforms = (self.byte_count / 64UL).to_int() +
    padding_transforms
  let bit_length = self.byte_count * 8UL
  self.append_data_byte(0x80)
  if self.block_length > 56 {
    while self.block_length != 0 {
      self.append_data_byte(0x00)
    }
  }
  while self.block_length < 56 {
    self.append_data_byte(0x00)
  }
  for shift in 0..<8 {
    self.block[56 + shift] = (bit_length >> ((7 - shift) * 8)).to_byte()
  }
  self.transform()
  if self.transformed_blocks != expected_transforms {
    abort("canonical SHA-256 block accounting diverged")
  }
  self.block_length = 0
  let output : FixedArray[Byte] = FixedArray::make(32, 0)
  for index in 0..<8 {
    output.unsafe_write_uint32_be(index * 4, self.state[index])
  }
  output
}

///|
pub fn CanonicalSha256::finalize_hex(self : CanonicalSha256) -> String {
  guard self.cached_hex is None else { return self.cached_hex.unwrap() }
  self.finalized = true
  let digest = self.finalize_bytes()
  let encoded : FixedArray[Byte] = FixedArray::make(64, 0)
  let digits : Bytes = b"0123456789abcdef"
  for index, byte in digest {
    encoded[index * 2] = digits[(byte.to_int() >> 4) & 0x0f]
    encoded[index * 2 + 1] = digits[byte.to_int() & 0x0f]
  }
  let value = try! @utf8.decode(Bytes::from_array(encoded))
  self.cached_hex = Some(value)
  value
}
