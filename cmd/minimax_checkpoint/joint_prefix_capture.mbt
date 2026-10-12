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
fn joint_initial_capture_regions() -> Array[(String, @loop.LatentBuffer)] {
  [
    ("initial-video-rows.f32", VideoState),
    ("initial-audio-rows.f32", AudioState),
  ]
}

///|
test "every prefix retains packed initial noise independently of final head" {
  assert_eq(joint_initial_capture_regions().length(), 2)
  assert_eq(joint_initial_capture_regions()[0].0, "initial-video-rows.f32")
  assert_eq(joint_initial_capture_regions()[1].0, "initial-audio-rows.f32")
}

///|
fn capture_joint_packed_state(
  input : @program.InputProgram,
  layout : JointLayout,
) -> Bytes raise {
  input
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
  video_update : @device.Function,
  audio_update : @device.Function,
  refiners : ArrayView[(@weights.RefinerWeightPacking, @device.Allocation)],
  blocks~ : Int,
) -> Array[(String, Bytes)] raise {
  guard blocks > 0 && blocks <= 50 else {
    fail("invalid diagnostic block extent")
  }
  let text_captures = text.capture_stages(stream, refiners)
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
  let packed = capture_joint_packed_state(input, layout)
  let embedding = variant.with_timestep_embedding((allocation, offset, size) => {
    allocation.copy_to_host(offset, size)
  })
  let captures : Array[(String, Bytes)] = []
  captures.append(text_captures)
  // A one-block prefix must also demonstrate literal input identity. Capture
  // after producer retirement and before any flow update; final-head captures
  // keep their existing names. Ordinary serving never enters this diagnostic.
  for (name, region) in joint_initial_capture_regions() {
    captures.push((name, storage.copy_to_host(region)))
  }
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
    let frontiers = if block == 0 { joint_block_frontiers() } else { [] }
    guard frontiers.is_empty() || block_queue.kernel_count() == 13 else {
      fail("bound block changed; diagnostic frontiers must be updated")
    }
    for index in 0..<block_queue.kernel_count() {
      block_queue.enqueue(index)
      if block == 0 && index == 0 {
        stream.synchronize()
        captures.push(
          (
            "block-0-parameters.bf16",
            parameters.copy_to_host(
              0L,
              joint_parameter_capture_bytes(
                count,
                layout.sequence.assembly().width(),
              ),
            ),
          ),
        )
      }
      for (name, end, stage) in frontiers {
        if index + 1 == end {
          // Completion events belong to the complete executor. Retire only
          // this stream at intermediate frontiers, without replaying a kernel.
          stream.synchronize()
          let bytes = match stage {
            AttentionResidual => capture_joint_packed_state(input, layout)
            _ => variant.capture_block_stage(stage)
          }
          captures.push(("block-0-\{name}.bf16", bytes))
        }
      }
    }
    block_queue.record_completion()
    block_queue.wait_completion()
    let first_output = capture_joint_packed_state(input, layout)
    captures.push(("block-\{block}.bf16", first_output))
    println("joint_prefix_block=\{block}")
  }
  // Only the complete-stack diagnostic may execute the final head. Retain
  // literal packed velocities and pre/post-update states for an independent
  // original-head/flow replay; ordinary media serving has no observers.
  if blocks == weights.layers.length() {
    let head_count = variant.output_launch_count()
    let tail = joint_prediction_tail(
      predictions[0].length(),
      prefix,
      head_count,
    )
    let head = context.create_ordered_kernel_executor(
      stream,
      predictions[0][tail:],
      policy=OrderedKernelEagerOnly,
    )
    defer head.close()
    for index in 0..<head.kernel_count() {
      head.enqueue(index)
    }
    head.record_completion()
    head.wait_completion()
    for
      (name, region) in [
        ("video-state.f32", @loop.LatentBuffer::VideoState),
        ("audio-state.f32", AudioState),
        ("video-velocity.f32", VideoVelocity),
        ("audio-velocity.f32", AudioVelocity),
      ] {
      captures.push((name, storage.copy_to_host(region)))
    }
    let flow = context.create_ordered_kernel_executor(
      stream,
      [
        storage.flow_launch(
          video_update,
          step=0,
          latent=VideoLatent,
          block_size=256,
        ),
        storage.flow_launch(
          audio_update,
          step=0,
          latent=AudioLatent,
          block_size=256,
        ),
      ],
      policy=OrderedKernelEagerOnly,
    )
    defer flow.close()
    for index in 0..<flow.kernel_count() {
      flow.enqueue(index)
    }
    flow.record_completion()
    flow.wait_completion()
    captures.push(("video-updated.f32", storage.copy_to_host(VideoState)))
    captures.push(("audio-updated.f32", storage.copy_to_host(AudioState)))
    captures.push(("flow-table.f32", layout.plan.flow_table().bytes()))
    let step = layout.plan.steps()[0]
    captures.push(
      (
        "sigmas.f32",
        joint_float_bytes([
          Float::from_double(step.video_sigma),
          Float::from_double(step.video_next_sigma),
          Float::from_double(step.video_timestep),
          Float::from_double(step.audio_sigma),
          Float::from_double(step.audio_next_sigma),
          Float::from_double(step.audio_timestep),
        ]),
      ),
    )
    println("joint_prefix_head_and_first_update=true")
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
      "inverse.i32",
      matched_joint_ints(
        schedule.patterns()[schedule.steps()[0].pattern].indices[:],
      ),
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
fn joint_prediction_tail(total : Int, prefix : Int, head : Int) -> Int raise {
  guard prefix > 0 && head > 0 && total > prefix + head else {
    fail("complete prediction must include input, blocks and final head")
  }
  total - head
}

///|
test "head capture follows the complete stack and never replays input producers" {
  assert_eq(joint_prediction_tail(107, 7, 6), 101)
  for
    (total, prefix, head) in [(13, 7, 6), (12, 7, 6), (107, 0, 6), (107, 7, 0)] {
    try joint_prediction_tail(total, prefix, head) catch {
      _ => ()
    } noraise {
      _ => fail("invalid head capture extent accepted")
    }
  }
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
