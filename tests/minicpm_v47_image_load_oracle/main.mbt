///|
async fn main {
  guard @env.args()[1:] is [oracle, python, reference, output] &&
    !@fs.exists(output) else {
    fail("usage: image_load_oracle ORACLE PYTHON PINNED_LOADER NEW_OUTPUT")
  }
  @fs.mkdir(output, recursive=true)
  let mut cases = 0
  let mut png_cases = 0
  let mut jpeg_cases = 0
  let mut orientation_bytes = 0L
  let mut stored_raster_mismatches = 0
  let mut jpeg_different_bytes = 0L
  let mut jpeg_maximum_error = 0
  let mut jpeg_exact_cases = 0
  let mut pillow = ""
  let codec_modes : Array[Json] = []
  for
    (format, modes) in [
      ("PNG", ["RGB", "RGBA", "L", "LA"]),
      ("JPEG", ["RGB", "L", "CMYK"]),
    ] {
    for mode in modes {
      let mut mode_maximum = 0
      let mut mode_different = 0L
      let mut mode_exact = 0
      for endian in ["little", "big"] {
        for orientation in 0..<=8 {
          let (width, height) = [
              (1, 1),
              (1, 9),
              (9, 1),
              (7, 7),
              (8, 8),
              (9, 9),
              (17, 11),
              (31, 33),
              (64, 48),
            ][orientation]
          let channels = match mode {
            "L" => 1
            "LA" => 2
            "RGB" => 3
            _ => 4
          }
          let raw = Bytes::makei(width * height * channels, i => {
            ((i * 37 + i / 7 * 23 + cases * 11) % 256).to_byte()
          })
          let prefix = output + "/case-\{cases}"
          let raw_path = prefix + ".input"
          @fs.write_file(raw_path, raw, create_mode=CreateNew)
          guard @process.run(oracle, [
              python,
              reference,
              "make",
              mode,
              format,
              width.to_string(),
              height.to_string(),
              orientation.to_string(),
              endian,
              (orientation % 2 == 0).to_string(),
              raw_path,
              prefix,
            ]) ==
            0 else {
            fail("Pillow fixture/load reference failed")
          }
          let encoded = @fs.read_file(prefix + ".encoded").binary()
          let stored = @rgb8.Image::decode(
            encoded,
            maximum_encoded_bytes=1048576,
            maximum_pixels=100000,
          )
          let actual = @rgb8.Image::decode_oriented(
            encoded,
            maximum_encoded_bytes=1048576,
            maximum_pixels=100000,
          )
          let native_input = prefix + ".native.rgb"
          @fs.write_file(native_input, stored.data(), create_mode=CreateNew)
          let native_reference = prefix + ".native-reference"
          guard @process.run(oracle, [
              python,
              reference,
              "orient",
              width.to_string(),
              height.to_string(),
              orientation.to_string(),
              endian,
              native_input,
              native_reference,
            ]) ==
            0 else {
            fail("upstream orientation on native RGB failed")
          }
          guard @fs.read_file(native_reference + ".json").json()
            is {
              "width": Number(expected_width, ..),
              "height": Number(expected_height, ..),
              "pillow_version": String(version),
              ..
            } else {
            fail("image loader reference metadata missing")
          }
          pillow = version
          let expected = @fs.read_file(native_reference + ".rgb").binary()
          guard actual.width() == expected_width.to_int() &&
            actual.height() == expected_height.to_int() &&
            actual.data() == expected else {
            fail(
              "EXIF orientation differs from pinned upstream: \{format}/\{mode}/\{endian}/\{orientation}",
            )
          }
          orientation_bytes += expected.length().to_int64()
          if stored.width() != actual.width() ||
            stored.height() != actual.height() ||
            stored.data() != expected {
            stored_raster_mismatches += 1
          }
          let decoded_reference = @fs.read_file(prefix + ".stored.rgb").binary()
          assert_eq(stored.data().length(), decoded_reference.length())
          let mut different = 0
          for i in 0..<decoded_reference.length() {
            let error = (stored.data()[i].to_int() -
            decoded_reference[i].to_int()).abs()
            mode_maximum = mode_maximum.max(error)
            if error != 0 {
              different += 1
            }
            if format == "JPEG" {
              jpeg_maximum_error = jpeg_maximum_error.max(error)
            }
          }
          mode_different += different.to_int64()
          if different == 0 {
            mode_exact += 1
          }
          if format == "PNG" {
            guard different == 0 &&
              actual.data() == @fs.read_file(prefix + ".rgb").binary() else {
              fail("PNG RGB conversion differs from pinned upstream")
            }
            png_cases += 1
          } else {
            jpeg_cases += 1
            jpeg_different_bytes += different.to_int64()
            if different == 0 {
              jpeg_exact_cases += 1
            }
          }
          cases += 1
        }
      }
      codec_modes.push({
        "format": format,
        "mode": mode,
        "cases": 18,
        "exact_cases": mode_exact,
        "different_bytes": Json(mode_different),
        "maximum_absolute_byte_error": mode_maximum,
      })
    }
  }
  guard cases == 126 && stored_raster_mismatches > 0 else {
    fail("image-load corpus coverage differs")
  }
  let report : Json = {
    "orientation_passed": true,
    "cases": cases,
    "orientation_compared_bytes": Json(orientation_bytes),
    "stored_raster_mismatches_reproduced": stored_raster_mismatches,
    "png_exact_cases": png_cases,
    "jpeg_cases": jpeg_cases,
    "jpeg_exact_cases": jpeg_exact_cases,
    "jpeg_different_bytes": Json(jpeg_different_bytes),
    "jpeg_maximum_absolute_byte_error": jpeg_maximum_error,
    "transformers_version": "5.18.0",
    "pillow_version": pillow,
    "codec_modes": Json(codec_modes),
    "orientation_scope": "unchanged upstream loader applied to native decoded RGB; both EXIF byte orders, all8 transforms, JPEG APP1/PNG eXIf",
    "codec_scope": "separate full PNG/JPEG decode comparison; JPEG differences are measured, not accepted as parity",
    "hardware_access": false,
    "production_python": false,
  }
  @fs.write_file(
    output + "/report.json",
    report.stringify(indent=2),
    create_mode=CreateNew,
  )
  println(report.stringify(indent=2))
}
