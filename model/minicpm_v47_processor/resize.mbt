///|
// Separable antialiased cubic convolution, a=-0.5. RGB8 uses Pillow's
// 22-bit coefficient rounding and clips after each pass. The public processor
// chooses this numeric profile explicitly; torchvision parity is separate.
fn cubic(x : Double) -> Double {
  let x = x.abs()
  if x < 1.0 {
    (1.5 * x - 2.5) * x * x + 1.0
  } else if x < 2.0 {
    (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
  } else {
    0.0
  }
}

///|
fn coefficients(
  source : Int32,
  target : Int32,
  index : Int32,
) -> (Int32, Array[Int64]) {
  let scale = source.to_double() / target.to_double()
  let support_scale = scale.max(1.0)
  let center = (index.to_double() + 0.5) * scale
  let first = (center - 2.0 * support_scale + 0.5).to_int().max(0)
  let last = (center + 2.0 * support_scale + 0.5).to_int().min(source)
  let weights = Array::makei(last - first, i => {
    cubic((i.to_double() + first.to_double() - center + 0.5) / support_scale)
  })
  let total = weights.fold(init=0.0, (a, b) => a + b)
  (
    first,
    weights.map(weight => {
      let value = weight / total * 4194304.0
      (value + (if value < 0.0 { -0.5 } else { 0.5 })).to_int64()
    }),
  )
}

///|
fn clip(sum : Int64) -> Byte {
  (sum >> 22).max(0L).min(255L).to_int().to_byte()
}

///|
fn normalized_bf16(value : Byte) -> UInt32 {
  // PIL rescale produces F32, then mean/std arithmetic stays F32. Preserve
  // those stages before the model's explicit BF16 input cast.
  let scaled = Float::from_double(value.to_int().to_double() * (1.0 / 255.0))
  let normalized = (scaled - 0.5F) / 0.5F
  let bits = normalized.reinterpret_as_uint()
  (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16
}

///|
fn write_crop(
  plan : Plan,
  crop : Crop,
  rgb : BytesView,
  output : FixedArray[Byte],
  column_offset : Int32,
) -> Unit {
  let width = crop.size.width
  let height = crop.size.height
  // Only this crop's columns are retained, even when its canvas is wider.
  let horizontal = FixedArray::make(plan.input.height * width * 3, b'\x00')
  for x in 0..<width {
    let (first, weights) = coefficients(
      plan.input.width,
      crop.canvas.width,
      crop.left + x,
    )
    for y in 0..<plan.input.height {
      for channel in 0..<3 {
        let mut sum = 2097152L
        for i in 0..<weights.length() {
          sum += rgb[(y * plan.input.width + first + i) * 3 + channel]
            .to_int()
            .to_int64() *
            weights[i]
        }
        horizontal[(y * width + x) * 3 + channel] = clip(sum)
      }
    }
  }
  for y in 0..<height {
    let (first, weights) = coefficients(
      plan.input.height,
      crop.canvas.height,
      crop.top + y,
    )
    for x in 0..<width {
      for channel in 0..<3 {
        let mut sum = 2097152L
        for i in 0..<weights.length() {
          sum += horizontal[((first + i) * width + x) * 3 + channel]
            .to_int()
            .to_int64() *
            weights[i]
        }
        let word = normalized_bf16(clip(sum))
        let patch = y / 14 * (width / 14) + x / 14
        let column = column_offset + patch * 14 + x % 14
        let offset = ((channel * 14 + y % 14) * plan.columns + column) * 2
        output[offset] = (word & 255U).to_byte()
        output[offset + 1] = (word >> 8).to_byte()
      }
    }
  }
}

///|
/// Numeric payload ceiling for preparation: BF16 output and its immutable copy,
/// one crop's horizontal RGB8 scratch and resize coefficient arrays. Excludes
/// caller-owned input and small object headers; no image work is performed.
pub fn Plan::preparation_bytes(self : Self) -> Int64 {
  let mut horizontal = 0L
  for crop in self.crops {
    horizontal = horizontal.max(
      self.input.height.to_int64() * crop.size.width.to_int64() * 3L,
    )
  }
  let coefficient_bytes = self.input.height.max(self.input.width).to_int64() *
    32L
  let bytes = self.pixel_bytes()
  bytes * 2L + horizontal + coefficient_bytes
}

///|
/// BF16 NaViT pixels. max_bytes bounds output construction, resize intermediates
/// and coefficient payloads, excluding caller-owned RGB and object headers.
pub fn Plan::prepare_rgb(
  self : Self,
  rgb : BytesView,
  max_bytes~ : Int64,
) -> Pixels raise {
  guard rgb.length().to_int64() ==
    self.input.height.to_int64() * self.input.width.to_int64() * 3L else {
    fail("RGB8 length differs from planned image geometry")
  }
  let bytes = self.pixel_bytes()
  guard self.crops.all(crop => {
      self.input.height.to_int64() * crop.size.width.to_int64() * 3L <=
      2147483647L
    }) &&
    self.preparation_bytes() <= max_bytes else {
    fail("image preparation exceeds numeric payload budget")
  }
  let output = FixedArray::make(bytes.to_int(), b'\x00')
  let mut offset = 0
  for crop in self.crops {
    write_crop(self, crop, rgb, output, offset)
    offset += crop.size.height / 14 * crop.size.width
  }
  { plan: self, data: Bytes::from_array(output[:]), }
}
