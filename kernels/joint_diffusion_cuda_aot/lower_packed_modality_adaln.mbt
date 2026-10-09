///|
pub fn lower_packed_modality_adaln_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  timestep~ : @joint_diffusion_io_plan.JointPackedTimestepSelection,
  modality~ : @joint_diffusion_io_plan.JointPackedModalitySelection,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
) -> JointDiffusionPackedModalityAdaLnCudaCandidate raise JointDiffusionCudaAotError {
  if !target.supports_bf16() {
    raise Invalid(Target)
  }
  if compiler.reassociate() || compiler.fused_multiply_add() {
    raise Invalid(Compiler)
  }
  if timestep.identity() != requirements.source_identity() ||
    modality.identity() != requirements.source_identity() ||
    modality.packed_timestep_digest() != timestep.digest() ||
    modality.packed_row_count() != timestep.packed_row_count() {
    raise Invalid(RequestShape)
  }
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let hidden = match (requirement.operation(), requirement.capability()) {
    (PackedModalityAdaLnRows(value), PackedModalityAdaLnRowsAot) => value
    _ => raise UnsupportedRequirement(requirement_ordinal)
  }
  if hidden != requirements.denoiser_shape().hidden_size() {
    raise Invalid(Requirement)
  }
  let packed = timestep.packed_row_count()
  let video = modality.video_indices().length()
  let audio = modality.audio_indices().length()
  let selected_elements = checked_mul(
    (video + audio).to_int64(),
    hidden.to_int64(),
  )
  let packed_elements = checked_mul(packed.to_int64(), hidden.to_int64())
  if packed_elements > limits.maximum_elements {
    raise Invalid(Limit)
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=((selected_elements + 255L) / 256L).to_int(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let operands = [
    launch_operand(PackedAdaLnShiftInput, checked_mul(packed_elements, 2L), 16L),
    launch_operand(PackedAdaLnScaleInput, checked_mul(packed_elements, 2L), 16L),
    launch_operand(VideoPackedRowIndices, video.to_int64() * 4L, 16L),
    launch_operand(AudioPackedRowIndices, audio.to_int64() * 4L, 16L),
    launch_operand(
      VideoAdaLnShiftOutput,
      video.to_int64() * hidden.to_int64() * 2L,
      16L,
    ),
    launch_operand(
      VideoAdaLnScaleOutput,
      video.to_int64() * hidden.to_int64() * 2L,
      16L,
    ),
    launch_operand(
      AudioAdaLnShiftOutput,
      audio.to_int64() * hidden.to_int64() * 2L,
      16L,
    ),
    launch_operand(
      AudioAdaLnScaleOutput,
      audio.to_int64() * hidden.to_int64() * 2L,
      16L,
    ),
  ]
  let symbol = "lunaflux_joint_diffusion_packed_modality_adaln_gather_v1"
  let rendered = @affine_modulation_cuda_source.packed_modality_adaln_gather_source(
    packed_rows=packed,
    video_rows=video,
    audio_rows=audio,
    hidden_size=hidden,
    function_symbol=symbol,
  ) catch {
    error => raise AffineSource(error)
  }
  let source = rendered.source_bytes()
  let source_digest = JointDiffusionCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let compiler_id = compiler.identity()
  let recipe = @utf8.encode(
    "lunaflux.joint-packed-modality-adaln.v1|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
    "|timestep=" +
    timestep.digest().as_hex() +
    "|modality=" +
    modality.digest().as_hex() +
    "|geometry=" +
    packed.to_string() +
    ":" +
    video.to_string() +
    ":" +
    audio.to_string() +
    ":" +
    hidden.to_string() +
    "|sm=" +
    target.compute_major().to_string() +
    "." +
    target.compute_minor().to_string() +
    ":" +
    target.supports_bf16().to_string() +
    ":" +
    target.supports_cublas_lt().to_string() +
    "|compiler=" +
    compiler_id.toolchain_digest_sha256() +
    ":" +
    compiler_id.major().to_string() +
    "." +
    compiler_id.minor().to_string() +
    "." +
    compiler_id.patch().to_string() +
    "|opt=" +
    compiler.optimization_level().to_string() +
    "|fma=" +
    compiler.fused_multiply_add().to_string() +
    "|reassociate=" +
    compiler.reassociate().to_string() +
    "|max-registers=" +
    compiler.max_registers().to_string() +
    "|launch=" +
    dimensions.grid_x().to_string() +
    ":256:0|abi=packed-bf16-shift,packed-bf16-scale,video-i32,audio-i32,video-bf16-shift,video-bf16-scale,audio-bf16-shift,audio-bf16-scale" +
    "|source=" +
    source_digest.as_hex(),
  )
  if source.length() > limits.maximum_source_bytes ||
    recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: BitExactBf16PackedModalityGatherV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    timestep_digest: timestep.digest(),
    modality_digest: modality.digest(),
    hidden_size: hidden,
    packed_rows: packed,
    video_rows: video,
    audio_rows: audio,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    source,
    source_digest,
    recipe,
    recipe_digest: JointDiffusionCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(recipe)),
    ),
  }
}
