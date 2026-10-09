///|
fn attention_dimensions(
  elements : Int64,
  block_x : Int,
) -> @launch_contract.AotLaunchDimensions raise JointDiffusionCudaAotError {
  let grid = (elements + block_x.to_int64() - 1L) / block_x.to_int64()
  if grid > 2147483647L {
    raise Invalid(Limit)
  }
  @launch_contract.AotLaunchDimensions::new(
    grid_x=grid.to_int(),
    grid_y=1,
    grid_z=1,
    block_x~,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
}

///|
fn attention_stage(
  scope : JointDiffusionAttentionRowScope,
  kind : JointDiffusionAttentionStageKind,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : Array[
    @joint_diffusion_artifact_admission.JointDiffusionLaunchOperand,
  ],
) -> JointDiffusionAttentionCudaStage {
  { scope, kind, dimensions, operands: ReadOnlyArray::from_array(operands), }
}

///|
fn attention_scope_stages(
  scope : JointDiffusionAttentionRowScope,
  rows : Int,
  hidden : Int,
  heads : Int,
  head_dim : Int,
  rotary : Int,
  projection_weights : Int64,
) -> Array[JointDiffusionAttentionCudaStage] raise JointDiffusionCudaAotError {
  let inner = checked_mul(heads.to_int64(), head_dim.to_int64())
  let hidden_elements = checked_mul(rows.to_int64(), hidden.to_int64())
  let inner_elements = checked_mul(rows.to_int64(), inner)
  let hidden_bytes = checked_mul(hidden_elements, 2L)
  let inner_bytes = checked_mul(inner_elements, 2L)
  let weight_bytes = checked_mul(projection_weights, 2L)
  let qkv = attention_stage(
    scope,
    QkvProjection,
    attention_dimensions(inner_elements, 256),
    [
      launch_operand(JointHiddenInput, hidden_bytes, 16L),
      launch_operand(AttentionQueryWeight, weight_bytes, 16L),
      launch_operand(AttentionKeyWeight, weight_bytes, 16L),
      launch_operand(AttentionValueWeight, weight_bytes, 16L),
      launch_operand(AttentionQueryBuffer, inner_bytes, 16L),
      launch_operand(AttentionKeyBuffer, inner_bytes, 16L),
      launch_operand(AttentionValueBuffer, inner_bytes, 16L),
    ],
  )
  let norm_operands = [
    launch_operand(AttentionQueryBuffer, inner_bytes, 16L),
    launch_operand(AttentionKeyBuffer, inner_bytes, 16L),
    launch_operand(AttentionQueryNormWeight, head_dim.to_int64() * 2L, 16L),
    launch_operand(AttentionKeyNormWeight, head_dim.to_int64() * 2L, 16L),
  ]
  let norm_kind = match scope {
    PackedRows => {
      let rotary_bytes = checked_mul(
        checked_mul(rows.to_int64(), rotary.to_int64()),
        4L,
      )
      norm_operands.push(
        launch_operand(AttentionRotaryCosInput, rotary_bytes, 16L),
      )
      norm_operands.push(
        launch_operand(AttentionRotarySinInput, rotary_bytes, 16L),
      )
      QkNormRotary
    }
    RefinerRows => QkNorm
  }
  norm_operands.push(
    launch_operand(AttentionNormalizedQueryBuffer, inner_bytes, 16L),
  )
  norm_operands.push(
    launch_operand(AttentionNormalizedKeyBuffer, inner_bytes, 16L),
  )
  let norm = attention_stage(
    scope,
    norm_kind,
    attention_dimensions(checked_mul(rows.to_int64(), heads.to_int64()), 1),
    norm_operands,
  )
  let attention = attention_stage(
    scope,
    FullAttention,
    attention_dimensions(inner_elements, 256),
    [
      launch_operand(AttentionNormalizedQueryBuffer, inner_bytes, 16L),
      launch_operand(AttentionNormalizedKeyBuffer, inner_bytes, 16L),
      launch_operand(AttentionValueBuffer, inner_bytes, 16L),
      launch_operand(AttentionOutputBuffer, inner_bytes, 16L),
    ],
  )
  let output = attention_stage(
    scope,
    OutputProjection,
    attention_dimensions(hidden_elements, 256),
    [
      launch_operand(AttentionOutputBuffer, inner_bytes, 16L),
      launch_operand(AttentionOutputWeight, weight_bytes, 16L),
      launch_operand(JointHiddenOutput, hidden_bytes, 16L),
    ],
  )
  [qkv, norm, attention, output]
}

///|
fn attention_stage_text(
  stages : ArrayView[JointDiffusionAttentionCudaStage],
) -> String {
  let out = StringBuilder()
  for index, stage in stages {
    if index > 0 {
      out.write_char(';')
    }
    out.write_string(
      match stage.scope {
        PackedRows => "packed"
        RefinerRows => "refiner"
      },
    )
    out.write_char(':')
    out.write_string(
      match stage.kind {
        QkvProjection => "qkv"
        QkNormRotary => "qk-norm-rotary"
        QkNorm => "qk-norm"
        FullAttention => "attention"
        OutputProjection => "output"
      },
    )
    out.write_char('@')
    out.write_string(stage.dimensions.grid_x().to_string())
    out.write_char('[')
    for operand_index, operand in stage.operands {
      if operand_index > 0 {
        out.write_char(',')
      }
      out.write_string(role_tag(operand.role()))
      out.write_char('@')
      out.write_string(operand.byte_count().to_string())
      out.write_char('/')
      out.write_string(operand.alignment().to_string())
    }
    out.write_char(']')
  }
  out.to_string()
}

///|
pub fn lower_transformer_attention_cuda_aot_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  request~ : @joint_diffusion_io_plan.JointDiffusionIoPlan,
  packed_timestep~ : @joint_diffusion_io_plan.JointPackedTimestepSelection,
  requirement_ordinal~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : JointDiffusionCudaAotLimits,
  document_boundaries? : ArrayView[Int] = [],
) -> JointDiffusionTransformerAttentionCudaCandidate raise JointDiffusionCudaAotError {
  if request.identity() != requirements.source_identity() ||
    packed_timestep.identity() != request.identity() ||
    packed_timestep.request_digest() != request.request_digest() {
    raise Invalid(RequestShape)
  }
  if !target.supports_bf16() {
    raise Invalid(Target)
  }
  if compiler.reassociate() || compiler.fused_multiply_add() {
    raise Invalid(Compiler)
  }
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let (layers, refiners, heads, head_dim, hidden) = match
    (requirement.operation(), requirement.capability()) {
    (JointTransformerAttention(l, r, h, d, width), JointTransformerAttentionAot) =>
      (l, r, h, d, width)
    _ => raise UnsupportedRequirement(requirement_ordinal)
  }
  let denoiser = requirements.denoiser_shape()
  if layers != denoiser.layer_count() ||
    refiners != denoiser.refiner_layer_count() ||
    heads != denoiser.attention_head_count() ||
    head_dim != denoiser.attention_head_dimension() ||
    hidden != denoiser.hidden_size() {
    raise Invalid(Requirement)
  }
  let inner = checked_mul(heads.to_int64(), head_dim.to_int64())
  let projection_weights = checked_mul(inner, hidden.to_int64())
  let packed_rows = packed_timestep.packed_row_count()
  let boundaries = if document_boundaries.is_empty() {
    [0, packed_rows]
  } else {
    document_boundaries.to_array()
  }
  let refiner_rows = request.text().token_count()
  let packed_elements = checked_mul(packed_rows.to_int64(), inner)
  let refiner_elements = checked_mul(refiner_rows.to_int64(), inner)
  if projection_weights > limits.maximum_elements ||
    packed_elements > limits.maximum_elements ||
    refiner_elements > limits.maximum_elements {
    raise Invalid(Limit)
  }
  let rotary = denoiser.attention_rotary_dimension()
  let stages = attention_scope_stages(
    PackedRows,
    packed_rows,
    hidden,
    heads,
    head_dim,
    rotary,
    projection_weights,
  )
  stages.append(
    attention_scope_stages(
      RefinerRows,
      refiner_rows,
      hidden,
      heads,
      head_dim,
      0,
      projection_weights,
    ),
  )
  let packed_source = @joint_attention_cuda_source.staged_joint_attention_bf16_source(
    rows=packed_rows,
    hidden_size=hidden,
    head_count=heads,
    head_dimension=head_dim,
    rotary_dimension=rotary,
    epsilon_numerator=denoiser.query_key_norm_epsilon_numerator(),
    epsilon_denominator=denoiser.query_key_norm_epsilon_denominator(),
    apply_rotary=true,
    function_prefix="lunaflux_joint_attention_packed_bf16_v1",
    document_boundaries=boundaries,
  ) catch {
    error => raise AttentionSource(error)
  }
  let refiner_source = @joint_attention_cuda_source.staged_joint_attention_bf16_source(
    rows=refiner_rows,
    hidden_size=hidden,
    head_count=heads,
    head_dimension=head_dim,
    rotary_dimension=0,
    epsilon_numerator=denoiser.query_key_norm_epsilon_numerator(),
    epsilon_denominator=denoiser.query_key_norm_epsilon_denominator(),
    apply_rotary=false,
    function_prefix="lunaflux_joint_attention_refiner_bf16_v1",
  ) catch {
    error => raise AttentionSource(error)
  }
  let source = packed_source.source_bytes() + refiner_source.source_bytes()
  let source_digest = JointDiffusionCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let compiler_id = compiler.identity()
  let recipe = @utf8.encode(
    "lunaflux.joint-transformer-attention.v1|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|request=" +
    request.request_digest().as_hex() +
    "|packed-timestep=" +
    packed_timestep.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
    "|layers=" +
    layers.to_string() +
    ":" +
    refiners.to_string() +
    "|geometry=" +
    packed_rows.to_string() +
    ":" +
    refiner_rows.to_string() +
    ":" +
    hidden.to_string() +
    ":" +
    heads.to_string() +
    ":" +
    head_dim.to_string() +
    ":" +
    rotary.to_string() +
    "|qk-epsilon=" +
    denoiser.query_key_norm_epsilon_numerator().to_string() +
    "/" +
    denoiser.query_key_norm_epsilon_denominator().to_string() +
    "|weights=qkv-bf16[" +
    inner.to_string() +
    "," +
    hidden.to_string() +
    "]:out-bf16[" +
    hidden.to_string() +
    "," +
    inner.to_string() +
    "]:qk-norm-bf16[" +
    head_dim.to_string() +
    "]|" +
    (if boundaries.length() == 2 {
      "mask=none"
    } else {
      "documents=" + boundaries.map(row => row.to_string()).join(",")
    }) +
    "|causal=0" +
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
    "|stages=" +
    attention_stage_text(stages) +
    "|source=" +
    source_digest.as_hex(),
  )
  if source.length() > limits.maximum_source_bytes ||
    recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    numerical_contract: OrderedF32AccumulationBf16StagedAttentionV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    request_digest: request.request_digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    layer_count: layers,
    refiner_layer_count: refiners,
    packed_row_count: packed_rows,
    refiner_row_count: refiner_rows,
    hidden_size: hidden,
    head_count: heads,
    head_dimension: head_dim,
    inner_size: inner.to_int(),
    rotary_dimension: rotary,
    qk_epsilon_numerator: denoiser.query_key_norm_epsilon_numerator(),
    qk_epsilon_denominator: denoiser.query_key_norm_epsilon_denominator(),
    projection_weight_element_count: projection_weights,
    norm_weight_element_count: head_dim.to_int64(),
    stages: ReadOnlyArray::from_array(stages),
    source,
    source_digest,
    recipe,
    recipe_digest: JointDiffusionCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(recipe)),
    ),
  }
}
