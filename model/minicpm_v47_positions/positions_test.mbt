///|
async fn selected_spec() -> @model.Spec {
  @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
}

///|
async test "plain text chunks and decode stage distinct causal and rotary buffers" {
  let spec = selected_spec()
  let sequence = @minicpm_v47_positions.Sequence::build(
    spec,
    [10, 20, 30],
    [Text, Text, Text],
    image_grids=[],
    video_grids=[],
  )
  assert_eq(sequence.prompt_tokens(), 3)
  assert_eq(sequence.rope_delta(), 0)
  assert_eq(sequence.coordinates(), [0, 0, 0, 1, 1, 1, 2, 2, 2])
  let canvas = FixedArray::make(9, -1)
  let causal = FixedArray::make(3, -1)
  sequence.write_prompt_into(start=1, rows=2, canvas, causal)
  assert_eq(canvas[:].to_owned(), [1, 1, 1, 2, 2, 2, -1, -1, -1])
  assert_eq(causal[:].to_owned(), [1, 2, -1])
  sequence.write_decode_into(causal_start=3, rows=2, canvas, causal)
  assert_eq(canvas[:].to_owned(), [3, 3, 3, 4, 4, 4, -1, -1, -1])
  assert_eq(causal[:].to_owned(), [3, 4, -1])
}

///|
async test "thumbnail canvas shrinks rotary length without moving cache addresses" {
  let spec = selected_spec()
  let m = spec.markers()
  let sequence = @minicpm_v47_positions.Sequence::build(
    spec,
    [
      100,
      m.image_start,
      m.image,
      m.image,
      m.image,
      m.image,
      m.image,
      m.image,
      m.image_end,
      200,
    ],
    [Text, Text, Image, Image, Image, Image, Image, Image, Text, Text],
    image_grids=[{ height: 8, width: 12, }],
    video_grids=[],
  )
  assert_eq(sequence.coordinates(), [
    0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 2, 1, 1, 2, 2, 1, 2, 3, 1, 3,
    4, 5, 5, 5,
  ])
  assert_eq(sequence.rope_delta(), -4)
  let canvas = FixedArray::make(6, 0)
  let causal = FixedArray::make(2, 0)
  sequence.write_decode_into(causal_start=10, rows=2, canvas, causal)
  assert_eq(canvas[:].to_owned(), [6, 6, 6, 7, 7, 7])
  assert_eq(causal[:].to_owned(), [10, 11])
}

///|
async test "thumbnail stretches over slices with nearest even half coordinates" {
  let spec = selected_spec()
  let m = spec.markers()
  let sequence = @minicpm_v47_positions.Sequence::build(
    spec,
    [
      100,
      m.image_start,
      m.image,
      m.image,
      m.image,
      m.image_end,
      m.slice_start,
      m.image,
      m.image,
      m.slice_end,
      m.slice_start,
      m.image,
      m.image,
      m.slice_end,
      m.slice_start,
      m.image,
      m.image,
      m.slice_end,
      200,
    ],
    [
      Text,
      Text,
      Image,
      Image,
      Image,
      Text,
      Text,
      Image,
      Image,
      Text,
      Text,
      Image,
      Image,
      Text,
      Text,
      Image,
      Image,
      Text,
      Text,
    ],
    image_grids=[
      { height: 4, width: 12, },
      { height: 4, width: 8, },
      { height: 4, width: 8, },
      { height: 4, width: 8, },
    ],
    video_grids=[],
  )
  assert_eq(sequence.coordinates(), [
    0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 3, 1, 1, 6, 1, 2, 7, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 1, 1, 2, 1, 1, 3, 1, 1, 3, 1, 1, 4, 1, 1, 4, 1, 1, 5, 1, 1, 5, 1, 1, 6, 1,
    1, 6, 8, 8, 8,
  ])
  assert_eq(sequence.rope_delta(), -10)
}

///|
async test "slice rows place newline beyond the right canvas edge" {
  let spec = selected_spec()
  let m = spec.markers()
  let sequence = @minicpm_v47_positions.Sequence::build(
    spec,
    [
      m.image_start,
      m.image,
      m.image_end,
      m.slice_start,
      m.image,
      m.slice_end,
      m.slice_start,
      m.image,
      m.slice_end,
      m.newline,
      m.slice_start,
      m.image,
      m.slice_end,
      m.slice_start,
      m.image,
      m.slice_end,
      200,
    ],
    [
      Text,
      Image,
      Text,
      Text,
      Image,
      Text,
      Text,
      Image,
      Text,
      Text,
      Text,
      Image,
      Text,
      Text,
      Image,
      Text,
      Text,
    ],
    image_grids=Array::make(5, { height: 4, width: 4, }),
    video_grids=[],
  )
  assert_eq(sequence.coordinates(), [
    0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
    1, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 3, 3, 3,
  ])
  assert_eq(sequence.rope_delta(), -13)
}

///|
async test "adjacent video and image frames consume independent grid streams" {
  let spec = selected_spec()
  let m = spec.markers()
  let sequence = @minicpm_v47_positions.Sequence::build(
    spec,
    [
      m.image_start,
      m.video,
      m.image_end,
      m.newline,
      m.image_start,
      m.image,
      m.image_end,
      200,
    ],
    [Text, Video, Text, Text, Text, Image, Text, Text],
    image_grids=[{ height: 4, width: 4, }],
    video_grids=[{ height: 4, width: 4, }],
  )
  assert_eq(sequence.coordinates(), [
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4,
  ])
  assert_eq(sequence.rope_delta(), -3)
}

///|
async test "per-call downsample controls placeholder count and missing grids fail" {
  let spec = selected_spec()
  let m = spec.markers()
  let tokens = [m.image_start, m.image, m.image, m.image, m.image, m.image_end]
  let modalities : Array[@minicpm_v47_positions.Modality] = [
    Text,
    Image,
    Image,
    Image,
    Image,
    Text,
  ]
  let sequence = @minicpm_v47_positions.Sequence::build(
    spec,
    tokens,
    modalities,
    image_grids=[{ height: 4, width: 4, }],
    video_grids=[],
    downsample=FourX,
  )
  assert_eq(sequence.coordinates(), [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 2, 2,
  ])
  let mismatch = try
    @minicpm_v47_positions.Sequence::build(
      spec,
      tokens,
      modalities,
      image_grids=[{ height: 4, width: 4, }],
      video_grids=[],
    )
  catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(mismatch)
  let missing = try
    @minicpm_v47_positions.Sequence::build(
      spec,
      tokens,
      modalities,
      image_grids=[],
      video_grids=[],
    )
  catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(missing)
}

///|
async test "ordinary intervening text separates visual groups and advances the halo" {
  let spec = selected_spec()
  let m = spec.markers()
  let sequence = @minicpm_v47_positions.Sequence::build(
    spec,
    [
      m.image_start,
      m.image,
      m.image_end,
      100,
      200,
      m.image_start,
      m.image,
      m.image_end,
      300,
    ],
    [Text, Image, Text, Text, Text, Text, Image, Text, Text],
    image_grids=Array::make(2, { height: 4, width: 4, }),
    video_grids=[],
  )
  assert_eq(sequence.coordinates(), [
    0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6,
    6,
  ])
  assert_eq(sequence.rope_delta(), -2)
}
