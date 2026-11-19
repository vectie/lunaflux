///|
async fn main {
  guard @env.args()[1:] is [oracle, python, reference, output] &&
    !@fs.exists(output) else {
    fail(
      "usage: minicpm_v47_pil_pipeline_oracle ORACLE PYTHON REFERENCE NEW_OUTPUT",
    )
  }
  @fs.mkdir(output, recursive=true)
  let config = @fs.read_file("model/minicpm_v47/config.fixture.json").text()
  let cases = [
    (56, 56, 9),
    (640, 960, 9),
    (896, 448, 9),
    (1792, 1792, 9),
    (97, 613, 9),
    (448, 448, 9),
    (449, 448, 9),
    (640, 960, 1),
    (640, 960, 2),
    (517, 1301, 7),
    (1301, 517, 7),
    (3, 227, 1),
  ]
  let records : Array[Json] = []
  let mut words = 0L
  for index, (height, width, max_slices) in cases {
    let spec = @model.Spec::parse(@utf8.encode(config)[:])
    let plan = @processor.Plan::build(spec, height~, width~, max_slices~)
    let source = Bytes::from_array(
      Array::makei(height * width * 3, i => {
        let x = i / 3 % width
        let y = i / 3 / width
        ((x * 37 + y * 61 + i % 3 * 79 + x % 7 * (y % 11) * 19) % 256).to_byte()
      }),
    )
    // Reuse only this verifier's scratch paths, so the complete corpus does
    // not retain copies of large pixel arrays or require checkpoint storage.
    @fs.write_file(output + "/input.rgb", source)
    guard @process.run(oracle, [
        python,
        reference,
        output + "/input.rgb",
        height.to_string(),
        width.to_string(),
        max_slices.to_string(),
        output + "/pixels.bf16",
        output + "/metadata.json",
      ]) ==
      0 else {
      fail("upstream PIL pipeline failed case=\{index}")
    }
    let metadata = @json.parse(@fs.read_file(output + "/metadata.json").text())
    let (grid_h, grid_w) = plan.slice_grid()
    let sizes : Json = Json(
      plan.target_sizes().map(grid => [grid.height, grid.width]),
    )
    guard metadata
      is {
        "target_sizes": expected_sizes,
        "grids": expected_grids,
        "num_patches_per_image": expected_count,
        "shape": shape,
        ..
      } &&
      expected_sizes == sizes &&
      expected_grids == Json([[grid_h, grid_w]]) &&
      expected_count == Json([plan.crops().length()]) &&
      shape == Json([1, 3, 14, plan.pixel_columns()]) else {
      fail(
        "upstream PIL geometry/packing shape differs case=\{index}: " +
        metadata.stringify(),
      )
    }
    let expected = @fs.read_file(output + "/pixels.bf16").binary()
    let pixels = plan.prepare_rgb(source[:], max_bytes=128000000L).data()
    assert_eq(expected.length(), pixels.length())
    for offset in 0..<pixels.length() {
      guard pixels[offset] == expected[offset] else {
        fail(
          "full PIL BF16 mismatch case=\{index} byte=\{offset}: native=\{pixels[offset]} upstream=\{expected[offset]}",
        )
      }
    }
    words += pixels.length().to_int64() / 2L
    records.push({
      "index": index,
      "height": height,
      "width": width,
      "max_slices": max_slices,
      "upstream": metadata,
      "bf16_words": pixels.length() / 2,
    })
    println(
      "upstream PIL case \{index}: \{height}x\{width}, max_slices=\{max_slices} passed",
    )
  }
  let report : Json = {
    "passed": true,
    "upstream": "Transformers5.18.0 unchanged PIL geometry/preprocess/backend/transforms",
    "cases": records,
    "bf16_words_compared": words,
    "scope": "decoded RGB8; thumbnail/detail geometry, row-major crops, normalize, NaViT packing, BF16 cast",
    "torchvision_profile_parity": false,
    "codec_parity": false,
    "gpu_execution": false,
  }
  @fs.write_file(
    output + "/report.json",
    report.stringify(indent=2),
    create_mode=CreateNew,
  )
  println(
    "full upstream PIL pipeline: \{cases.length()} cases, \{words} BF16 words matched",
  )
}
