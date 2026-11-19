///|
async fn selected_spec() -> @model.Spec {
  @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
}

///|
async test "full selected vision program joins all459 tensor names in318 launches" {
  let spec = selected_spec()
  let program = @minicpm_v47_vision.Program::new(
    spec,
    maximum_patches=8192,
    maximum_crop_patches=2048,
  )
  assert_eq(program.row_capacities(), (8192, 2048, 512))
  assert_eq(program.entries().length(), 25)
  assert_eq(program.stages().length(), 318)
  assert_eq(program.workspace_bytes(), 202637312L)
  let names = program.weight_names()
  assert_eq(names.length(), 459)
  let vision = @weights.requirements(spec).filter(requirement => {
    match requirement.component {
      VisionEmbedding
      | VisionLayer(_)
      | VisionFinalNorm
      | EarlyVisionMerger
      | OutputVisionMerger => true
      _ => false
    }
  })
  assert_eq(vision.length(), names.length())
  let mut bytes = 0L
  for requirement in vision {
    assert_true(names.contains(requirement.name))
    bytes += requirement.shape.fold(init=2L, (a, b) => a * b)
  }
  assert_eq(bytes, 1105972064L)
  let source = @utf8.decode(program.source_bytes())
  for entry in program.entries() {
    assert_true(source.contains("void " + entry.symbol() + "("))
    assert_true(entry.shared_bytes() <= 49152)
  }
  for stage in program.stages() {
    assert_eq(
      stage.arguments().length(),
      program.entries()[stage.entry()].parameters(),
    )
  }
  let mut offset = 0L
  for region in program.regions() {
    assert_eq(region.offset, offset)
    offset += region.bytes
  }
  assert_eq(offset, program.workspace_bytes())
}

///|
async test "vision program applies early compression after layer6 and final GELU erf" {
  let program = @minicpm_v47_vision.Program::new(
    selected_spec(),
    maximum_patches=8192,
    maximum_crop_patches=2048,
  )
  let stages = program.stages()
  assert_true(stages[79].arguments()[0] is Buffer(RawCounts))
  assert_true(
    stages[80].arguments()[2]
    is Weight("model.vision_tower.vit_merger.layer_norm1.weight"),
  )
  assert_true(stages[92].arguments()[3] is Buffer(State))
  assert_eq(
    program.entries()[stages[92].entry()].symbol(),
    "vision_early_mean_residual",
  )
  assert_true(stages[93].arguments()[0] is Buffer(EncodedCounts))
  assert_true(
    stages[93].arguments()[2]
    is Weight("model.vision_tower.encoder.layers.7.layer_norm1.weight"),
  )
  assert_true(stages[317].arguments()[4] is Buffer(Features))
  let text = @utf8.decode(program.source_bytes())
  assert_true(text.contains("tanhf("))
  assert_true(text.contains("erff("))
}

///|
async test "4x program skips early weights and keeps all27 layers at full rows" {
  let config = @utf8.decode(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary(),
  )
  let spec = @model.Spec::parse(
    @utf8.encode(config.replace(old="\"16x\"", new="\"4x\""))[:],
  )
  let program = @minicpm_v47_vision.Program::new(
    spec,
    maximum_patches=8192,
    maximum_crop_patches=2048,
  )
  assert_eq(program.row_capacities(), (8192, 8192, 2048))
  assert_eq(program.entries().length(), 13)
  assert_eq(program.stages().length(), 305)
  assert_eq(program.weight_names().length(), 443)
  assert_true(!@utf8.decode(program.source_bytes()).contains("vision_early"))
  assert_true(program.weight_names().all(name => !name.contains("vit_merger")))
}
