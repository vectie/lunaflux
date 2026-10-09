///|
pub struct ProcessedPixels {
  grid : Grid
  patches : Bytes
}

///|
fn nearest_even(value : Double) -> Int {
  let lower = value.floor().to_int()
  let fraction = value - lower.to_double()
  if fraction > 0.5 || (fraction == 0.5 && lower % 2 != 0) {
    lower + 1
  } else {
    lower
  }
}

///|
/// Qwen smart-resize, factor32, with explicit admitted pixel bounds. Defaults
/// match the published32B processor; caller may lower max_pixels for a budget.
pub fn resized_shape(
  height : Int,
  width : Int,
  min_pixels? : Int = 65536,
  max_pixels? : Int = 16777216,
) -> (Int, Int) raise VisionError {
  guard height > 0 &&
    width > 0 &&
    height <= 2147483615 &&
    width <= 2147483615 &&
    min_pixels >= 1024 &&
    max_pixels >= min_pixels &&
    Int::max(height, width).to_double() / Int::min(height, width).to_double() <=
    200.0 else {
    raise InvalidGeometry
  }
  let h = Int::max(32, nearest_even(height.to_double() / 32.0) * 32)
  let w = Int::max(32, nearest_even(width.to_double() / 32.0) * 32)
  if h.to_int64() * w.to_int64() > max_pixels.to_int64() {
    let ratio = (height.to_double() * width.to_double() / max_pixels.to_double()).sqrt()
    (
      Int::max(32, (height.to_double() / ratio / 32.0).floor().to_int() * 32),
      Int::max(32, (width.to_double() / ratio / 32.0).floor().to_int() * 32),
    )
  } else if h.to_int64() * w.to_int64() < min_pixels.to_int64() {
    let ratio = (min_pixels.to_double() /
    (height.to_double() * width.to_double())).sqrt()
    (
      (height.to_double() * ratio / 32.0).ceil().to_int() * 32,
      (width.to_double() * ratio / 32.0).ceil().to_int() * 32,
    )
  } else {
    (h, w)
  }
}

///|
fn cubic(value : Double) -> Double {
  let x = value.abs()
  if x < 1.0 {
    (1.5 * x - 2.5) * x * x + 1.0
  } else if x < 2.0 {
    ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
  } else {
    0.0
  }
}

///|
fn filter_weights(
  source : Int,
  target : Int,
  index : Int,
) -> (Int, ReadOnlyArray[Double]) {
  let scale = source.to_double() / target.to_double()
  let support = Double::max(scale, 1.0)
  let center = (index.to_double() + 0.5) * scale
  let first = Int::max(0, (center - 2.0 * support + 0.5).floor().to_int())
  let last = Int::min(source, (center + 2.0 * support + 0.5).floor().to_int())
  let weights = Array::makei(last - first, i => {
    cubic((first.to_double() + i.to_double() + 0.5 - center) / support)
  })
  let total = weights.fold(init=0.0, (a, b) => a + b)
  (first, ReadOnlyArray::from_array(weights.map(w => w / total)))
}

///|
fn rounded_rgb(value : Double) -> Int {
  Int::max(0, Int::min(255, (value + 0.5).floor().to_int()))
}

///|
/// RGB8 decoded frames -> antialiased bicubic resize -> [0,1] rescale and
/// mean/std0.5 -> merge-group-ordered BF16 patches. Odd frame count repeats
/// the final frame; image input is one frame. Codecs and video frame selection
/// belong to ingress; this function never interprets compressed data.
/// Numeric payload budget includes horizontal resize Float64 plus both the
/// mutable patch construction buffer and immutable output conversion.
pub fn preprocess_rgb(
  frames : ArrayView[Bytes],
  height~ : Int,
  width~ : Int,
  min_pixels? : Int = 65536,
  max_pixels? : Int = 16777216,
  max_bytes~ : Int64,
) -> ProcessedPixels raise VisionError {
  guard !frames.is_empty() &&
    height > 0 &&
    width > 0 &&
    width.to_int64() * height.to_int64() <= 2147483647L / 3L else {
    raise InvalidGeometry
  }
  for frame in frames {
    guard frame.length().to_int64() == height.to_int64() * width.to_int64() * 3L else {
      raise InvalidGeometry
    }
  }
  let (h, w) = resized_shape(height, width, min_pixels~, max_pixels~)
  let times = (frames.length().to_int64() + 1L) / 2L
  let pixels = h.to_int64() * w.to_int64()
  guard pixels > 0L && times <= 2147483647L / pixels / 12L else {
    raise BudgetExceeded
  }
  let output_bytes = times * pixels * 12L
  let temporary = height.to_int64() * w.to_int64() * 3L * 8L
  guard temporary <= 2147483647L &&
    output_bytes <= max_bytes / 2L &&
    temporary <= max_bytes - output_bytes * 2L else {
    raise BudgetExceeded
  }
  let output : FixedArray[Byte] = FixedArray::make(output_bytes.to_int(), 0)
  let horizontal : FixedArray[Double] = FixedArray::make(
    (temporary / 8L).to_int(),
    0.0,
  )
  // Each physical input frame is processed once; final-frame repetition is
  // written into a second temporal slot without materializing a resized copy.
  for frame_index, frame in frames {
    for x in 0..<w {
      let (first, weights) = filter_weights(width, w, x)
      for y in 0..<height {
        for c in 0..<3 {
          let sample = for i, weight in weights; value = 0.0 {
            continue value +
              frame[(y * width + first + i) * 3 + c].to_int().to_double() *
              weight
          } nobreak {
            value
          }
          horizontal[(y * w + x) * 3 + c] = sample
        }
      }
    }
    for y in 0..<h {
      let (first, weights) = filter_weights(height, h, y)
      for x in 0..<w {
        for c in 0..<3 {
          let sample = for i, weight in weights; value = 0.0 {
            continue value + horizontal[((first + i) * w + x) * 3 + c] * weight
          } nobreak {
            value
          }
          let normalized = (
              Float::from_int(rounded_rgb(sample)) / 255.0F - 0.5F
            ) /
            0.5F
          let bits = normalized.reinterpret_as_uint()
          let bf16 = (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16
          let py = y / 16
          let px = x / 16
          let patch = (
              (frame_index / 2 * (h / 32) + py / 2) * (w / 32) + px / 2
            ) *
            4 +
            py % 2 * 2 +
            px % 2
          let base = patch * 1536 + c * 512 + y % 16 * 16 + x % 16
          let slot = frame_index % 2
          let offset = (base + slot * 256) * 2
          output[offset] = (bf16 & 255U).to_byte()
          output[offset + 1] = (bf16 >> 8).to_byte()
          if frame_index == frames.length() - 1 && frames.length() % 2 != 0 {
            let repeated = (base + 256) * 2
            output[repeated] = output[offset]
            output[repeated + 1] = output[offset + 1]
          }
        }
      }
    }
  }
  {
    grid: { time: times.to_int(), height: h / 16, width: w / 16, },
    patches: Bytes::from_array(output[:]),
  }
}
