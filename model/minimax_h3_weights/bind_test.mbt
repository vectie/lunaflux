///|
fn h3_weight_metadata(
  variant? : @minimax_h3_spec.MiniMaxH3Variant = Ref2va,
  seed? : Char = '5',
) -> @minimax_h3_spec.MiniMaxH3ModelMetadata {
  @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
    @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
      variant~,
      checkpoint_format=DiffusersV1,
    ),
    try! @spec.ContentDigest::from_sha256(String::make(64, seed)),
  )
}

///|
fn one_tensor(
  name : String,
  shape? : ArrayView[Int64] = [1L, 1L],
  dtype? : String = "BF16",
) -> @safetensors.SafetensorsMetadata raise @safetensors.SafetensorsError {
  let mut elements = 1L
  for dimension in shape {
    elements = elements * dimension
  }
  let bytes = elements * (if dtype == "F32" { 4L } else { 2L })
  @safetensors.validate_metadata(
    [
      @safetensors.RawTensorMetadata::new(
        name~,
        dtype~,
        shape~,
        data_start=0L,
        data_end=bytes,
      ),
    ],
    encoded_header_bytes=512,
    payload_bytes=bytes,
    limits=@safetensors.HeaderLimits::new(
      max_header_bytes=4096,
      max_tensor_count=8,
      max_rank=8,
      max_name_length=256,
      max_payload_bytes=1073741824L,
      max_materialized_tensor_bytes=1073741824L,
    ),
  )
}

///|
test "MiniMax H3 binds denoiser shard to semantic plan identity" {
  let metadata = h3_weight_metadata()
  let plan = @minimax_h3.build(metadata)
  let bindings = @minimax_h3_weights.bind(
    metadata,
    plan,
    Ref2vaDenoiser,
    one_tensor("transformer_blocks.49.attn.to_q.weight", shape=[7168L, 5376L]),
  )
  assert_eq(bindings.component(), Ref2vaDenoiser)
  assert_eq(bindings.schema_version(), 1)
  assert_eq(bindings.tensor_count(), 1)
  assert_eq(bindings.identity().content(), metadata.content_digest())
  assert_eq(bindings.identity().plan(), plan.plan_digest())
  guard bindings.tensor(0) is Some(tensor) else { fail("missing bound tensor") }
  assert_eq(tensor.source_index(), 0)
  assert_eq(tensor.role(), DenoiserAttention)
  assert_eq(@minimax_h3_weights.Ref2vaDenoiser.conditioning_uses(), [
    Ref2vaImageConditioning,
    Ref2vaVideoConditioning,
    Ref2vaAudioConditioning,
  ])
  try bindings.require_materialization() catch {
    MaterializationUnavailable(Ref2vaDenoiser) => ()
    _ => fail("unexpected materialization boundary")
  } noraise {
    _ => fail("unimplemented materialization was admitted")
  }
}

///|
test "MiniMax H3 binds text video and audio component vocabularies" {
  let metadata = h3_weight_metadata()
  let plan = @minimax_h3.build(metadata)
  let text = @minimax_h3_weights.bind(
    metadata,
    plan,
    TextConditioner,
    one_tensor("model.language_model.layers.49.self_attn.q_proj.weight"),
  )
  let video = @minimax_h3_weights.bind(
    metadata,
    plan,
    VideoVae,
    one_tensor(
      "encoder.down_blocks.0.resnets.0.conv1.weight",
      shape=[128L, 128L, 3L, 3L, 3L],
      dtype="F32",
    ),
  )
  let audio = @minimax_h3_weights.bind(
    metadata,
    plan,
    AudioVae,
    one_tensor("encoder.block.0.weight_v", shape=[64L, 1L, 7L], dtype="F32"),
  )
  assert_eq(text.tensor(0).unwrap().role(), TextAttention)
  assert_eq(video.tensor(0).unwrap().role(), VideoEncoder)
  assert_eq(audio.tensor(0).unwrap().role(), AudioEncoder)
  assert_true(
    @minimax_h3_weights.TextConditioner
    .conditioning_uses()
    .contains(PromptConditioning),
  )
  assert_true(
    @minimax_h3_weights.VideoVae
    .conditioning_uses()
    .contains(Fl2vaKeyframeConditioning),
  )
  assert_eq(@minimax_h3_weights.AudioVae.conditioning_uses(), [
    Ref2vaAudioConditioning,
  ])
}

///|
test "MiniMax H3 rejects denoiser variant substitution" {
  let metadata = h3_weight_metadata(variant=Fl2va)
  let plan = @minimax_h3.build(metadata)
  let tensors = one_tensor("proj_in.weight", dtype="F32")
  try @minimax_h3_weights.bind(metadata, plan, Ref2vaDenoiser, tensors) catch {
    ComponentVariantMismatch(Ref2vaDenoiser) => ()
    _ => fail("unexpected partition substitution error")
  } noraise {
    _ => fail("Ref2VA denoiser was admitted for FL2VA content")
  }
}

///|
test "MiniMax H3 rejects unknown prefixed and rank-confused tensors" {
  let metadata = h3_weight_metadata()
  let plan = @minimax_h3.build(metadata)
  let prefixed = one_tensor("transformer.transformer_blocks.0.attn.to_q.weight")
  try @minimax_h3_weights.bind(metadata, plan, Ref2vaDenoiser, prefixed) catch {
    UnsupportedRawLayout(index=0) => ()
    _ => fail("unexpected raw-layout error")
  } noraise {
    _ => fail("consolidated raw prefix was admitted")
  }
  let out_of_range = one_tensor("transformer_blocks.50.attn.to_q.weight")
  try
    @minimax_h3_weights.bind(metadata, plan, Ref2vaDenoiser, out_of_range)
  catch {
    UnknownSemanticTensor(index=0) => ()
    _ => fail("unexpected layer-bound error")
  } noraise {
    _ => fail("out-of-range denoiser layer was admitted")
  }
  let confused = one_tensor("transformer_blocks.0.attn.to_q.weight", shape=[
    1L, 1L, 1L,
  ])
  try @minimax_h3_weights.bind(metadata, plan, Ref2vaDenoiser, confused) catch {
    InvalidSemanticRank(index=0) => ()
    _ => fail("unexpected semantic-rank error")
  } noraise {
    _ => fail("rank-confused denoiser projection was admitted")
  }
}

///|
test "MiniMax H3 rejects a plan from different authenticated content" {
  let metadata = h3_weight_metadata(seed='5')
  let foreign = @minimax_h3.build(h3_weight_metadata(seed='6'))
  let tensors = one_tensor("proj_out.weight", dtype="F32")
  try
    @minimax_h3_weights.bind(metadata, foreign, Ref2vaDenoiser, tensors)
  catch {
    PlanContentMismatch => ()
    _ => fail("unexpected content-plan error")
  } noraise {
    _ => fail("foreign plan identity was admitted")
  }
}

///|
test "complete vocabulary boundary exposes exact counts" {
  assert_eq(@minimax_h3_weights.Ref2vaDenoiser.required_tensor_count(), 638)
  assert_eq(@minimax_h3_weights.TextConditioner.required_tensor_count(), 1058)
  assert_eq(@minimax_h3_weights.VideoVae.required_tensor_count(), 703)
  assert_eq(@minimax_h3_weights.AudioVae.required_tensor_count(), 1087)
  let metadata = h3_weight_metadata()
  let shard = @minimax_h3_weights.bind(
    metadata,
    @minimax_h3.build(metadata),
    Ref2vaDenoiser,
    one_tensor("proj_in.weight", shape=[5376L, 96L], dtype="F32"),
  )
  try
    @minimax_h3_weights.join_complete_component(Ref2vaDenoiser, [shard])
  catch {
    IncompleteComponent(expected=638, actual=1) => ()
    _ => fail("unexpected partial-manifest result")
  } noraise {
    _ => fail("one-tensor shard was admitted as a complete denoiser")
  }
}

///|
test "MiniMax H3 enforces official mixed precision by tensor name" {
  let metadata = h3_weight_metadata()
  let plan = @minimax_h3.build(metadata)
  let projection = @minimax_h3_weights.bind(
    metadata,
    plan,
    Ref2vaDenoiser,
    one_tensor("proj_in.weight", shape=[5376L, 96L], dtype="F32"),
  )
  assert_eq(projection.tensor(0).unwrap().tensor().dtype(), Float32)
  let final_norm = @minimax_h3_weights.bind(
    metadata,
    plan,
    Ref2vaDenoiser,
    one_tensor("norm_out.norm.weight", shape=[5376L]),
  )
  let final_norm_tensor = final_norm.tensor(0).unwrap().tensor()
  assert_eq(final_norm_tensor.name(), "norm_out.norm.weight")
  assert_eq(final_norm_tensor.shape(), ReadOnlyArray::from_array([5376L]))
  assert_eq(final_norm_tensor.dtype(), BFloat16)
  for
    item in [
      ("time_embedder.linear_1.weight", [5376L, 256L]),
      ("time_embedder.linear_1.bias", [5376L]),
      ("time_embedder.linear_2.weight", [2688L, 5376L]),
      ("time_embedder.linear_2.bias", [2688L]),
    ] {
    let (name, shape) = item
    let timestep_mlp = @minimax_h3_weights.bind(
      metadata,
      plan,
      Ref2vaDenoiser,
      one_tensor(name, shape~, dtype="F32"),
    )
    let tensor = timestep_mlp.tensor(0).unwrap().tensor()
    assert_eq(tensor.shape(), ReadOnlyArray::from_array(shape))
    assert_eq(tensor.dtype(), Float32)
  }
  for
    item in [
      ("norm_out.linear.weight", [10752L, 2688L]),
      ("norm_out.linear.bias", [10752L]),
    ] {
    let (name, shape) = item
    let affine = @minimax_h3_weights.bind(
      metadata,
      plan,
      Ref2vaDenoiser,
      one_tensor(name, shape~),
    )
    let tensor = affine.tensor(0).unwrap().tensor()
    assert_eq(tensor.shape(), ReadOnlyArray::from_array(shape))
    assert_eq(tensor.dtype(), BFloat16)
  }
  for
    item in [
      ("proj_out.weight", [96L, 5376L]),
      ("proj_out.bias", [96L]),
      ("audio_proj_out.weight", [32L, 5376L]),
      ("audio_proj_out.bias", [32L]),
    ] {
    let (name, shape) = item
    let output_head = @minimax_h3_weights.bind(
      metadata,
      plan,
      Ref2vaDenoiser,
      one_tensor(name, shape~, dtype="F32"),
    )
    let tensor = output_head.tensor(0).unwrap().tensor()
    assert_eq(tensor.name(), name)
    assert_eq(tensor.shape(), ReadOnlyArray::from_array(shape))
    assert_eq(tensor.dtype(), Float32)
  }
  let wrong_projection = one_tensor("proj_in.weight", shape=[5376L, 96L])
  try
    @minimax_h3_weights.bind(metadata, plan, Ref2vaDenoiser, wrong_projection)
  catch {
    InvalidSemanticDType(index=0) => ()
    _ => fail("unexpected FP32 projection dtype rejection")
  } noraise {
    _ => fail("BF16 projection tensor was admitted")
  }
  let wrong_timestep_mlp = one_tensor("time_embedder.linear_1.weight", shape=[
    5376L, 256L,
  ])
  try
    @minimax_h3_weights.bind(metadata, plan, Ref2vaDenoiser, wrong_timestep_mlp)
  catch {
    InvalidSemanticDType(index=0) => ()
    _ => fail("unexpected timestep MLP dtype rejection")
  } noraise {
    _ => fail("BF16 timestep MLP tensor was admitted")
  }
  let wrong_output = one_tensor("audio_proj_out.weight", shape=[32L, 5376L])
  try
    @minimax_h3_weights.bind(metadata, plan, Ref2vaDenoiser, wrong_output)
  catch {
    InvalidSemanticDType(index=0) => ()
    _ => fail("unexpected F32 output-head dtype rejection")
  } noraise {
    _ => fail("BF16 output-head tensor was admitted")
  }
  let wrong_final_norm = one_tensor(
    "norm_out.norm.weight",
    shape=[5376L],
    dtype="F32",
  )
  try
    @minimax_h3_weights.bind(metadata, plan, Ref2vaDenoiser, wrong_final_norm)
  catch {
    InvalidSemanticDType(index=0) => ()
    _ => fail("unexpected BF16 final-RMSNorm dtype rejection")
  } noraise {
    _ => fail("F32 final-RMSNorm tensor was admitted")
  }
  let wrong_final_adaln_dtype = one_tensor(
    "norm_out.linear.weight",
    shape=[10752L, 2688L],
    dtype="F32",
  )
  try
    @minimax_h3_weights.bind(
      metadata,
      plan,
      Ref2vaDenoiser,
      wrong_final_adaln_dtype,
    )
  catch {
    InvalidSemanticDType(index=0) => ()
    _ => fail("unexpected final-AdaLN dtype rejection")
  } noraise {
    _ => fail("F32 final-AdaLN tensor was admitted")
  }
  let wrong_final_adaln_shape = one_tensor("norm_out.linear.weight", shape=[
    10752L, 2687L,
  ])
  try
    @minimax_h3_weights.bind(
      metadata,
      plan,
      Ref2vaDenoiser,
      wrong_final_adaln_shape,
    )
  catch {
    InvalidSemanticRank(index=0) => ()
    _ => fail("unexpected final-AdaLN shape rejection")
  } noraise {
    _ => fail("wrong final-AdaLN shape was admitted")
  }
  let wrong_core = one_tensor(
    "transformer_blocks.0.attn.to_q.weight",
    shape=[7168L, 5376L],
    dtype="F32",
  )
  try
    @minimax_h3_weights.bind(metadata, plan, Ref2vaDenoiser, wrong_core)
  catch {
    InvalidSemanticDType(index=0) => ()
    _ => fail("unexpected BF16 core dtype rejection")
  } noraise {
    _ => fail("F32 transformer-core tensor was admitted")
  }
}

///|
test "MiniMax H3 authenticates all 50 plus 2 bias-free SwiGLU weight pairs" {
  let metadata = h3_weight_metadata()
  let plan = @minimax_h3.build(metadata)
  for index in 0..<52 {
    let prefix = if index < 50 {
      "transformer_blocks.\{index}"
    } else {
      "token_refiner.blocks.\{index - 50}"
    }
    for
      item in [
        (prefix + ".ff.net.0.proj.weight", [28672L, 5376L]),
        (prefix + ".ff.net.2.weight", [5376L, 14336L]),
      ] {
      let (name, shape) = item
      let binding = @minimax_h3_weights.bind(
        metadata,
        plan,
        Ref2vaDenoiser,
        one_tensor(name, shape~),
      )
      let bound = binding.tensor(0).unwrap()
      let tensor = bound.tensor()
      assert_eq(bound.role(), DenoiserFeedForward)
      assert_eq(tensor.shape(), ReadOnlyArray::from_array(shape))
      assert_eq(tensor.dtype(), BFloat16)
    }
  }
  for
    hostile in [
      one_tensor("transformer_blocks.0.ff.net.0.proj.weight", shape=[
        28671L, 5376L,
      ]),
      one_tensor(
        "transformer_blocks.0.ff.net.2.weight",
        shape=[5376L, 14336L],
        dtype="F32",
      ),
    ] {
    try
      @minimax_h3_weights.bind(metadata, plan, Ref2vaDenoiser, hostile)
    catch {
      InvalidSemanticRank(index=0) | InvalidSemanticDType(index=0) => ()
      _ => fail("unexpected feed-forward weight rejection")
    } noraise {
      _ => fail("hostile feed-forward weight admitted")
    }
  }
  let out_of_range_refiner = try! one_tensor(
    "token_refiner.blocks.2.ff.net.2.weight",
    shape=[5376L, 14336L],
  )
  try
    @minimax_h3_weights.bind(
      metadata,
      plan,
      Ref2vaDenoiser,
      out_of_range_refiner,
    )
  catch {
    UnknownSemanticTensor(index=0) => ()
    _ => fail("unexpected refiner layer bound rejection")
  } noraise {
    _ => fail("third refiner MLP layer admitted")
  }
}
