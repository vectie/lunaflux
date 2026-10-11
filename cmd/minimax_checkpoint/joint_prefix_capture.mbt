///|
fn joint_float_bytes(values : ArrayView[Float]) -> Bytes {
  Bytes::from_array(
    Array::makei(values.length() * 4, i => {
      let bits = values[i / 4].reinterpret_as_uint()
      (bits >> (i % 4 * 8)).to_byte()
    }),
  )
}

///|
// Diagnostic only: execute serving producers, expose literal initial
// packed input and raw timestep embedding, stop before denoising or VAEs.
fn capture_joint_prefix(
  context : @device.Context,
  stream : @device.Stream,
  layout : JointLayout,
  storage : @latent.LatentStorage,
  text : @program.TextConditioningProgram,
  conditioning : @program.ConditioningProgram,
  input : @program.InputProgram,
  variants : ArrayView[@program.PredictionVariant],
  rotary : @latent.RotaryBindings,
  weights : @program.PredictionWeights,
  embeddings : @device.Allocation,
  parameters : @device.Allocation,
  blocks~ : Int,
) -> Array[(String, Bytes)] raise {
  guard blocks > 0 && blocks <= 50 else {
    fail("invalid diagnostic block extent")
  }
  text.execute()
  conditioning.execute()
  let predictions = layout.metadata.bind_conditioned(
    variants,
    input,
    storage,
    rotary,
    weights,
    conditioning,
    timestep_embeddings=embeddings,
    timestep_embeddings_offset=0L,
    parameters~,
    parameter_offset=0L,
  )
  let schedule = layout.sequence.schedule(
    video=layout.plan.steps()[:].map(step => step.video_timestep),
    audio=layout.plan.steps()[:].map(step => step.audio_timestep),
    visual_noise=0.0,
    audio_noise=0.0,
    vision_text_rows=[],
    max_steps=layout.plan.steps().length(),
    max_index_bytes=1048576L,
  )
  let count = schedule.steps()[0].values.length()
  let counts = layout.metadata.distinct_timestep_counts()
  let selected = for i in 0..<counts.length() {
    if counts[i] == count {
      break i
    }
  } nobreak {
    fail("missing prepared timestep variant")
  }
  let variant = variants[selected]
  let prefix = variant.input_launch_count(input)
  let queue = context.create_ordered_kernel_executor(
    stream,
    predictions[0][:prefix],
    policy=OrderedKernelEagerOnly,
  )
  defer queue.close()
  for index in 0..<queue.kernel_count() {
    queue.enqueue(index)
  }
  queue.record_completion()
  queue.wait_completion()
  let packed = input
    .packed_state()
    .with_state(
      layout.sequence.assembly().mapping(),
      width=layout.sequence.assembly().width(),
      (allocation, offset) => {
        allocation.copy_to_host(
          offset,
          layout.sequence.assembly().element_count() * 2L,
        )
      },
    )
  let embedding = variant.with_timestep_embedding((allocation, offset, size) => {
    allocation.copy_to_host(offset, size)
  })
  let captures : Array[(String, Bytes)] = []
  for block in 0..<blocks {
    let first = layout.metadata.bind_initial_block(
      variants,
      input,
      rotary,
      weights,
      timestep_embeddings=embeddings,
      parameters~,
      block~,
    )
    let block_queue = context.create_ordered_kernel_executor(
      stream,
      first[:],
      policy=OrderedKernelEagerOnly,
    )
    defer block_queue.close()
    for index in 0..<block_queue.kernel_count() {
      block_queue.enqueue(index)
    }
    block_queue.record_completion()
    block_queue.wait_completion()
    let first_output = input
      .packed_state()
      .with_state(
        layout.sequence.assembly().mapping(),
        width=layout.sequence.assembly().width(),
        (allocation, offset) => {
          allocation.copy_to_host(
            offset,
            layout.sequence.assembly().element_count() * 2L,
          )
        },
      )
    captures.push(("block-\{block}.bf16", first_output))
    println("joint_prefix_block=\{block}")
  }
  let positions : Array[Float] = []
  for position in layout.positions.positions() {
    positions.append([
      Float::from_double(position.time),
      Float::from_double(position.height),
      Float::from_double(position.width),
    ])
  }
  let files = [
    ("packed-input.bf16", packed),
    ("timestep.f32", embedding),
    ("positions.f32", joint_float_bytes(positions[:])),
    (
      "combined.i32",
      matched_joint_ints(
        schedule.patterns()[schedule.steps()[0].pattern].combined_adaln[:],
      ),
    ),
    (
      "documents.i32",
      matched_joint_ints(layout.sequence.attention_boundaries()[:]),
    ),
    (
      "PREFIX.txt",
      @utf8.encode(
        "scope=first_real_denoiser_input_and_blocks\npacked_rows=\{layout.sequence.assembly().rows().length()}\ndistinct_times=\{count}\nblock_count=\{blocks}\nno_media_output=true\n",
      ),
    ),
  ]
  files.append(captures)
  files
}

///|
fn matched_joint_ints(values : ArrayView[Int]) -> Bytes {
  Bytes::from_array(
    Array::makei(values.length() * 4, i => {
      (values[i / 4].reinterpret_as_uint() >> (i % 4 * 8)).to_byte()
    }),
  )
}

///|
test "literal joint diagnostic metadata preserves all F32 and I32 bits" {
  assert_eq(
    joint_float_bytes([0.0F, 1.0F]),
    b"\x00\x00\x00\x00\x00\x00\x80\x3f",
  )
  assert_eq(
    matched_joint_ints([0, 1009, 1024]),
    b"\x00\x00\x00\x00\xf1\x03\x00\x00\x00\x04\x00\x00",
  )
}
