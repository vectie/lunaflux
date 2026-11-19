///|
fn word(value : Byte) -> UInt32 {
  let scaled = (value.to_int().to_double() * (1.0 / 255.0)).to_float()
  let normalized = (scaled - 0.5F) / 0.5F
  let bits = normalized.reinterpret_as_uint()
  (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16
}

///|
async fn main {
  guard @env.args()[1:] is [oracle, output] else {
    fail("usage: minicpm_v47_processor_oracle PILLOW_C_ORACLE NEW_OUTPUT")
  }
  guard !@fs.exists(output) else { fail("oracle output already exists") }
  @fs.mkdir(output, recursive=true)
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let mut compared = 0L
  for
    index, (height, width) in [
      (56, 56),
      (640, 960),
      (896, 448),
      (1792, 1792),
      (97, 613),
    ] {
    let plan = @processor.Plan::build(spec, height~, width~)
    // Nonuniform high-frequency RGB values make interpolation, clipping,
    // horizontal-pass rounding and crop offsets observable.
    let source = Bytes::from_array(
      Array::makei(height * width * 3, i => {
        let x = i / 3 % width
        let y = i / 3 / width
        ((x * 37 + y * 61 + i % 3 * 79 + x % 7 * (y % 11) * 19) % 256).to_byte()
      }),
    )
    let input = output + "/input-\{index}.rgb"
    @fs.write_file(input, source, create_mode=CreateNew)
    let pixels = plan.prepare_rgb(source[:], max_bytes=128000000L).data()
    let mut column_start = 0
    for crop_index in 0..<plan.crops().length() {
      let crop = plan.crops()[crop_index]
      let resized = output + "/canvas-\{index}-\{crop_index}.rgb"
      guard @process.run(oracle, [
          input,
          resized,
          width.to_string(),
          height.to_string(),
          crop.canvas.width.to_string(),
          crop.canvas.height.to_string(),
        ]) ==
        0 else {
        fail("Pillow C oracle failed")
      }
      let expected = @fs.read_file(resized).binary()
      guard expected.length() == crop.canvas.width * crop.canvas.height * 3 else {
        fail("Pillow C output size")
      }
      for y in 0..<crop.size.height {
        for x in 0..<crop.size.width {
          for channel in 0..<3 {
            let raw = expected[(
                (y + crop.top) * crop.canvas.width + x + crop.left
              ) *
              3 +
              channel]
            let patch = y / 14 * (crop.size.width / 14) + x / 14
            let offset = (
                (channel * 14 + y % 14) * plan.pixel_columns() +
                column_start +
                patch * 14 +
                x % 14
              ) *
              2
            let observed = pixels[offset].to_uint() |
              (pixels[offset + 1].to_uint() << 8)
            guard observed == word(raw) else {
              fail(
                "Pillow BF16 mismatch case=\{index} crop=\{crop_index} y=\{y} x=\{x} channel=\{channel}: \{observed} != \{word(raw)}",
              )
            }
            compared += 1L
          }
        }
      }
      column_start += crop.size.height / 14 * crop.size.width
    }
    println(
      "Pillow C case \{index}: \{height}x\{width}, \{plan.crops().length()} crops passed",
    )
  }
  println("Pillow C oracle: \{compared} BF16 pixel words matched")
}
