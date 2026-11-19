///|
async fn programs() -> (@multimodal.Program, @vision.Program) {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let operators = @operators.DecoderOperators::new(spec, maximum_rows=32)
  let text = @bootstrap.text_program(
    operators,
    @bootstrap.compile_spark_attention(operators),
  )
  (
    @multimodal.Program::new(text),
    @vision.Program::new(spec, maximum_patches=16384, maximum_crop_patches=8192),
  )
}

///|
async test "combined payload includes both weight owners and all mutable allocations" {
  let (program, vision) = programs()
  let budget = @minicpm_v47_image_runtime.Budget::new(
    program,
    vision,
    text_weight_bytes=69319779584L,
    vision_weight_bytes=1105972064L,
  )
  assert_eq(budget.text.total_allocation_bytes, 69473047868L)
  assert_eq(budget.vision_storage_bytes, 429178928L)
  assert_eq(budget.image_metadata_bytes, 16388L)
  assert_eq(budget.mutable_bytes, 582463600L)
  assert_eq(budget.total_allocation_bytes, 71008215248L)
  assert_true(
    (try? @minicpm_v47_image_runtime.Budget::new(
      program,
      vision,
      text_weight_bytes=0L,
      vision_weight_bytes=-1L,
    ))
    is Err(_),
  )
  assert_true(
    (try? @minicpm_v47_image_runtime.Budget::new(
      program,
      vision,
      text_weight_bytes=0L,
      vision_weight_bytes=9223372036854775807L,
    ))
    is Err(_),
  )
}

///|
async test "fresh and closed owners cannot drive work" {
  let (program, vision) = programs()
  let execution = @minicpm_v47_image_runtime.Execution::new(program, vision)
  assert_true((try? execution.submit_next()) is Err(_))
  assert_true((try? execution.poll()) is Err(_))
  assert_true((try? execution.wait()) is Err(_))
  assert_true((try? execution.cancel()) is Err(_))
  execution.close()
  execution.close()
  assert_true((try? execution.submit_next()) is Err(_))
}

///|
async test "resident plans both complete weight sets without allocating device memory" {
  @fixture.with_checkpoint((spec, checkpoint, _) => {
    let operators = @operators.DecoderOperators::new(spec, maximum_rows=32)
    let text = @bootstrap.text_program(
      operators,
      @bootstrap.compile_spark_attention(operators),
    )
    let vision = @vision.Program::new(
      spec,
      maximum_patches=16384,
      maximum_crop_patches=8192,
    )
    let resident = @minicpm_v47_image_runtime.Resident::new(
      @multimodal.Program::new(text),
      vision,
      @weights.Plan::new(operators, checkpoint),
      @vision.WeightPlan::new(spec, vision, checkpoint),
    )
    assert_eq(resident.budget().total_allocation_bytes, 71008215248L)
    assert_true((try? resident.execution()) is Err(_))
    resident.close()
    resident.close()
    assert_true((try? resident.execution()) is Err(_))
  })
}

///|
async test "nine bounded AOT files are read without retaining file handles" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-v47-image-aot"))
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    group.add_defer(() => root.close())
    for
      name in [
        "decoder_envelope", "recurrent_branch", "moe", "attention_common", "attention_prefill",
        "attention_decode", "text_io", "vision_encoder", "image_scatter",
      ] {
      @fs.write_file(
        directory + "/" + name + ".cubin",
        b"fixture",
        create_mode=CreateNew,
      )
    }
    let images = @minicpm_v47_image_runtime.ModuleImages::read(
      root,
      maximum_module_bytes=32L,
    )
    assert_eq(images.vision, b"fixture")
    assert_eq(images.image_scatter, b"fixture")
    assert_eq(images.text.envelope, b"fixture")
    assert_true(
      (try? @minicpm_v47_image_runtime.ModuleImages::read(
        root,
        maximum_module_bytes=3L,
      ))
      is Err(_),
    )
    @fs.write_file(directory + "/image_scatter.cubin", b"")
    assert_true(
      (try? @minicpm_v47_image_runtime.ModuleImages::read(
        root,
        maximum_module_bytes=32L,
      ))
      is Err(_),
    )
    root.close()
  }
}
