///|
fn integer(value : Json) -> Int32 raise {
  guard value is Number(n, ..) && n == n.trunc() && n.abs() < 2147483648.0 else {
    fail("oracle integer required")
  }
  n.to_int()
}

///|
async fn main {
  guard @env.args()[1:] is [assets, oracle, python, reference, output] else {
    fail(
      "usage: minicpm_v47_canvas_oracle ASSETS ORACLE PYTHON_EXECUTABLE REFERENCE_SOURCE NEW_OUTPUT",
    )
  }
  @fs.mkdir(output, recursive=true)
  let config = @fs.read_file(assets + "/config.json").text()
  let spec = @model.Spec::parse(@utf8.encode(config)[:])
  let cases = corpus(config)
  let input = output + "/corpus.json"
  @fs.write_file(
    input,
    Json(cases.map(case => case.json())).stringify(),
    create_mode=CreateNew,
  )
  let observed = output + "/upstream.jsonl"
  guard @process.run(oracle, [
      python,
      reference,
      assets + "/config.json",
      input,
      observed,
    ]) ==
    0 else {
    fail("upstream coordinate oracle failed")
  }
  let lines = @fs.read_file(observed).text().trim().split("\n").to_array()
  guard lines.length() == cases.length() + 1 else {
    fail("incomplete upstream sequence corpus")
  }
  let mut compared_tokens = 0
  for index, case in cases {
    let sequence = @positions.Sequence::build(
      spec,
      case.tokens,
      case.modalities,
      image_grids=case.images,
      video_grids=case.videos,
      downsample=case.downsample,
    )
    guard @json.parse(lines[index + 1])
      is {
        "name": String(name),
        "positions": Array(axes),
        "rope_delta": [[delta]],
        ..
      } &&
      name == case.name &&
      axes.length() == 3 else {
      fail("invalid sequence reference")
    }
    let expected = Array::make(case.tokens.length() * 3, 0)
    for axis in 0..<3 {
      guard axes[axis] is [Array(values)] &&
        values.length() == case.tokens.length() else {
        fail("upstream coordinate shape differs")
      }
      for token, value in values {
        expected[token * 3 + axis] = integer(value)
        guard sequence.coordinates()[token * 3 + axis] ==
          expected[token * 3 + axis] else {
          fail(
            "canvas differs case=\{case.name} token=\{token} axis=\{axis} native=\{sequence.coordinates()[token * 3 + axis]} upstream=\{expected[token * 3 + axis]}",
          )
        }
      }
    }
    let delta = integer(delta)
    guard sequence.rope_delta() == delta else {
      fail("decode rope delta differs in " + case.name)
    }
    let canvas = FixedArray::make(96, 0)
    let causal = FixedArray::make(32, 0)
    for start = 0; start < case.tokens.length(); start = start + 32 {
      let rows = (case.tokens.length() - start).min(32)
      sequence.write_prompt_into(start~, rows~, canvas, causal)
      for row in 0..<rows {
        assert_eq(causal[row], start + row)
        for axis in 0..<3 {
          assert_eq(canvas[row * 3 + axis], expected[(start + row) * 3 + axis])
        }
      }
    }
    sequence.write_decode_into(
      causal_start=case.tokens.length(),
      rows=4,
      canvas,
      causal,
    )
    for row in 0..<4 {
      assert_eq(causal[row], case.tokens.length() + row)
      for axis in 0..<3 {
        assert_eq(canvas[row * 3 + axis], case.tokens.length() + row + delta)
      }
    }
    compared_tokens += case.tokens.length()
  }
  let report : Json = {
    "passed": true,
    "oracle": @json.parse(lines[0]),
    "upstream": "Transformers5.18.0 unchanged canvas methods",
    "cases": cases.length(),
    "prompt_tokens_compared": compared_tokens,
    "coordinates_compared": compared_tokens * 3,
    "decode_coordinates_compared": cases.length() * 12,
    "scope": "unpadded batch-one T/H/W groups/markers/slices and rope delta; prompt/decode staging",
    "gpu_execution": false,
    "pixel_processor_parity": false,
  }
  @fs.write_file(
    output + "/report.json",
    report.stringify(indent=2),
    create_mode=CreateNew,
  )
  println(report.stringify(indent=2))
}
