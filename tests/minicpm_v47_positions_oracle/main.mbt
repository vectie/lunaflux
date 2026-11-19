///|
fn integer(value : Json) -> Int32 raise {
  guard value is Number(n, ..) && n >= 0 && n == n.trunc() else {
    fail("nonnegative oracle integer required")
  }
  n.to_int()
}

///|
fn compare(
  spec : @model.Spec,
  count : Int32,
  extent : Int32,
  expected : Array[Json],
  vertical~ : Bool,
) -> Unit raise {
  let m = spec.markers()
  let tokens = [m.image_start]
  let modalities : Array[@positions.Modality] = [Text]
  for _ in 0..<count {
    tokens.push(m.image)
    modalities.push(Image)
  }
  tokens.push(m.image_end)
  tokens.push(m.slice_start)
  modalities.push(Text)
  modalities.push(Text)
  for _ in 0..<extent {
    tokens.push(m.image)
    modalities.push(Image)
  }
  tokens.push(m.slice_end)
  modalities.push(Text)
  let grids : Array[@positions.Grid] = if vertical {
    [{ height: count * 4, width: 4, }, { height: extent * 4, width: 4, }]
  } else {
    [{ height: 4, width: count * 4, }, { height: 4, width: extent * 4, }]
  }
  let sequence = @positions.Sequence::build(
    spec,
    tokens,
    modalities,
    image_grids=grids,
    video_grids=[],
  )
  let axis = if vertical { 1 } else { 2 }
  for index in 0..<count {
    let actual = sequence.coordinates()[(index + 1) * 3 + axis]
    let reference = integer(expected[index])
    guard actual == reference else {
      fail(
        "canvas mismatch count=\{count} extent=\{extent} index=\{index} vertical=\{vertical}: native=\{actual}, ATen=\{reference}",
      )
    }
  }
}

///|
async fn main {
  guard @env.args()[1:] is [oracle, output] else {
    fail(
      "usage: minicpm_v47_positions_oracle ORACLE_EXECUTABLE NEW_OUTPUT_DIRECTORY",
    )
  }
  @fs.mkdir(output, recursive=true)
  let observations = output + "/aten.jsonl"
  guard @process.run(oracle, [observations]) == 0 else {
    fail("ATen oracle failed")
  }
  let lines = @fs.read_file(observations).text().trim().split("\n").to_array()
  let identity = @json.parse(lines[0])
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let mut cases = 0
  let mut coordinates = 0
  for line in lines[1:] {
    guard @json.parse(line)
      is {
        "count": count,
        "extent": extent,
        "coordinates": Array(expected),
        ..
      } else {
      fail("invalid ATen observation")
    }
    let count = integer(count)
    let extent = integer(extent)
    guard expected.length() == count else { fail("oracle count mismatch") }
    compare(spec, count, extent, expected, vertical=false)
    compare(spec, count, extent, expected, vertical=true)
    cases += 1
    coordinates += count * 2
  }
  guard cases == 8196 else { fail("incomplete oracle matrix") }
  let report : Json = {
    "passed": true,
    "oracle": identity,
    "scope": "public canvas thumbnail positions; both spatial axes",
    "cases": cases,
    "coordinates_compared": coordinates,
    "gpu_execution": false,
    "full_processor_parity": false,
  }
  @fs.write_file(
    output + "/report.json",
    report.stringify(indent=2),
    create_mode=CreateNew,
  )
  println(report.stringify(indent=2))
}
