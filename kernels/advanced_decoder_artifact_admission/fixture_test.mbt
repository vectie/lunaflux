///|
fn artifact_test_content(fill : Char) -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(String::make(64, fill))
}

///|
fn artifact_test_plan(
  fill? : Char = 'a',
  hash_layer_count? : Int = 0,
) -> @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan {
  let shape = try! @advanced_decoder_execution_plan.AdvancedDecoderShape::new(
    vocabulary_size=128,
    hidden_size=128,
    decoder_layer_count=2,
    attention_head_count=2,
    kv_head_count=1,
    attention_head_dimension=128,
    rope_head_dimension=64,
    query_low_rank=128,
    output_low_rank=4,
    output_group_count=1,
    maximum_position_count=1024,
  )
  let low_rank = try! @advanced_decoder_execution_plan.LowRankAttentionPlan::new(
    query_rank=128,
    output_rank=4,
    output_group_count=1,
    query_key_norm_epsilon=0.001,
  )
  let position = try! @advanced_decoder_execution_plan.PositionEncodingPlan::new(
    rope_theta=10000.0,
    compressed_rope_theta=20000.0,
    original_position_count=128,
    scaling_factor=8.0,
    beta_fast=32.0,
    beta_slow=1.0,
  )
  let hyper = try! @advanced_decoder_execution_plan.HyperConnectionPlan::new(
    width=2,
    control_epsilon=0.000001,
    normalization_epsilon=0.000001,
    sinkhorn_iterations=4,
  )
  let moe = try! @advanced_decoder_execution_plan.MoeExecutionPlan::new(
    routed_expert_count=8,
    shared_expert_count=1,
    experts_per_token=2,
    intermediate_size=32,
    routed_scaling_factor=1.5,
    normalize_selected_probabilities=true,
    score_function=Sigmoid,
    selection_method=if hash_layer_count == 0 {
      NoAuxiliaryLossTokenChoice
    } else {
      TokenHashThenNoAuxiliaryLossTokenChoice(hash_layer_count~)
    },
    activation=SwiGlu,
  )
  try! @advanced_decoder_execution_plan.AdvancedDecoderExecutionPlan::new(
    schema_version=1,
    profile_identity="test/advanced-artifact-admission/v1",
    content_digest=artifact_test_content(fill),
    shape~,
    low_rank_attention=low_rank,
    position_encoding=position,
    hyper_connection=hyper,
    moe~,
    layers=[
      try! @advanced_decoder_execution_plan.AdvancedDecoderLayerPlan::new(
        layer_index=0,
        attention=try! @advanced_decoder_execution_plan.AdvancedAttentionPlan::sliding_window_causal(
          64,
        ),
      ),
      try! @advanced_decoder_execution_plan.AdvancedDecoderLayerPlan::new(
        layer_index=1,
        attention=try! @advanced_decoder_execution_plan.AdvancedAttentionPlan::indexed_compressed(
          compression_ratio=4,
          index_head_count=2,
          index_head_dimension=4,
          index_top_k=16,
        ),
      ),
    ],
    auxiliary_prediction=try! @advanced_decoder_execution_plan.AuxiliaryPredictionPlan::multi_token_prediction(
      2,
    ),
  )
}

///|
fn artifact_test_requirements(
  fill? : Char = 'a',
  hash_layer_count? : Int = 0,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements {
  try! @advanced_decoder_kernel_requirements.derive_requirements(
    artifact_test_plan(fill~, hash_layer_count~),
    limits=try! @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementLimits::new(
      maximum_layers=8,
      maximum_auxiliary_stages=8,
      maximum_requirements=64,
    ),
  )
}

///|
fn artifact_test_target(minor? : Int = 0) -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn artifact_test_dimensions(
  grid_x? : Int = 1,
) -> @launch_contract.AotLaunchDimensions {
  try! @launch_contract.AotLaunchDimensions::new(
    grid_x~,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn artifact_test_dimensions_for(
  operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation,
) -> @launch_contract.AotLaunchDimensions {
  match operation {
    QueryLowRankAProjection(rank) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=1,
        grid_y=(rank + 255) / 256,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    KeyValueProjection(_, head_dimension, _) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=1,
        grid_y=(head_dimension + 255) / 256,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    HyperConnectionBlockControlV1(_, _, _, _) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=1,
        grid_y=1,
        grid_z=1,
        block_x=1,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    HyperConnectionPreReduceV1(_) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=1,
        grid_y=1,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    HyperConnectionPostCombineV1(_) | HyperConnectionHeadReduceV1(_, _, _) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=1,
        grid_y=1,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    QueryKeyRmsNorm(_, _, _) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=1,
        grid_y=2,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=1024,
      )
    RotaryPositionEncoding(_, _, _, _, _, _) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=1,
        grid_y=3,
        grid_z=1,
        block_x=32,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    TokenHashRouting(_, _, _, _) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=1,
        grid_y=1,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    NoAuxiliaryLossTopK(_, _) | SelectedExpertWeightFinalize(_, _, _, _) =>
      try! @launch_contract.AotLaunchDimensions::new(
        grid_x=1,
        grid_y=1,
        grid_z=1,
        block_x=1,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      )
    _ => artifact_test_dimensions()
  }
}

///|
fn artifact_test_limits(
  maximum_canonical_bytes? : Int = 262144,
) -> AdvancedDecoderArtifactAdmissionLimits {
  try! AdvancedDecoderArtifactAdmissionLimits::new(
    maximum_requirements=64,
    maximum_modules=4,
    maximum_entry_points=64,
    maximum_module_bytes=1024L,
    maximum_total_module_bytes=4096L,
    maximum_symbol_bytes=64,
    maximum_operands_per_launch=8,
    maximum_canonical_bytes~,
  )
}

///|
fn artifact_test_digest(bytes : Bytes) -> @catalog.AotArtifactDigest {
  try! @catalog.AotArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  )
}

///|
fn artifact_test_module_digest() -> @catalog.AotArtifactDigest {
  artifact_test_digest(b"advanced-aot")
}

///|
fn artifact_test_entry(
  ordinal : Int,
  digest? : @catalog.AotArtifactDigest = artifact_test_module_digest(),
) -> @catalog.AotKernelEntryPoint {
  @catalog.AotKernelEntryPoint::new(
    family=@catalog.AotKernelFamily::new(
      module_digest=digest,
      family_id=try! @catalog.AotKernelFamilyId::from_stable_int(ordinal + 1),
    ),
    entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(1),
  )
}

///|
fn artifact_test_operands(
  operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation,
) -> Array[AdvancedDecoderLaunchOperand] {
  let operands : Array[AdvancedDecoderLaunchOperand] = []
  for index, role in artifact_test_roles(operation) {
    let (byte_count, alignment) = match operation {
      QueryLowRankAProjection(rank) =>
        match index {
          0 => (20L, 4L)
          1 => (128L * 2L, 16L)
          2 => (rank.to_int64() * 128L, 16L)
          3 => (rank.to_int64() / 128L, 16L)
          _ => (rank.to_int64() * 2L, 16L)
        }
      KeyValueProjection(heads, head_dimension, _) => {
        let output = heads.to_int64() * head_dimension.to_int64()
        match index {
          0 => (20L, 4L)
          1 => (128L * 2L, 16L)
          2 => (output * 128L, 16L)
          3 => (output / 128L, 16L)
          _ => (output * 2L, 16L)
        }
      }
      LanguageModelHead if index == 0 => (20L, 4L)
      ExpertScoreTransform(_, routed_expert_count) =>
        if index == 0 {
          (20L, 4L)
        } else {
          (routed_expert_count.to_int64() * 4L, 16L)
        }
      QueryKeyRmsNorm(_, query_width, key_value_width) =>
        match index {
          0 => (20L, 4L)
          1 | 2 | 3 => (query_width.to_int64() * 2L, 16L)
          _ => (key_value_width.to_int64() * 2L, 16L)
        }
      RotaryPositionEncoding(_, _, _, _, _, _) =>
        match index {
          0 => (20L, 4L)
          1 => (4L, 4L)
          2 | 3 => (256L, 16L)
          _ => (128L, 16L)
        }
      HyperConnectionBlockControlV1(width, _, _, _) => {
        let stream = width.to_int64()
        let hidden = 128L
        match index {
          0 => (20L, 4L)
          1 => (stream * hidden * 2L, 16L)
          2 => (6L * stream * stream * hidden * 4L, 16L)
          3 => (6L * stream * 4L, 16L)
          4 => (12L, 4L)
          5 | 6 => (stream * 4L, 16L)
          _ => (stream * stream * 4L, 16L)
        }
      }
      HyperConnectionPreReduceV1(width) => {
        let stream = width.to_int64()
        let hidden = 128L
        match index {
          0 => (20L, 4L)
          1 => (stream * hidden * 2L, 16L)
          2 => (stream * 4L, 16L)
          _ => (hidden * 2L, 16L)
        }
      }
      HyperConnectionPostCombineV1(width) => {
        let stream = width.to_int64()
        let hidden = 128L
        match index {
          0 => (20L, 4L)
          1 => (hidden * 2L, 16L)
          2 => (stream * hidden * 2L, 16L)
          3 => (stream * 4L, 16L)
          4 => (stream * stream * 4L, 16L)
          _ => (stream * hidden * 2L, 16L)
        }
      }
      HyperConnectionHeadReduceV1(width, _, _) => {
        let stream = width.to_int64()
        let hidden = 128L
        match index {
          0 => (20L, 4L)
          1 => (stream * hidden * 2L, 16L)
          2 => (stream * stream * hidden * 4L, 16L)
          3 => (stream * 4L, 16L)
          4 => (4L, 4L)
          _ => (hidden * 2L, 16L)
        }
      }
      TokenHashRouting(_, vocabulary_size, experts_per_token, _) =>
        match index {
          0 => (20L, 4L)
          1 => (4L, 4L)
          2 =>
            (vocabulary_size.to_int64() * experts_per_token.to_int64() * 4L, 4L)
          _ => (experts_per_token.to_int64() * 4L, 4L)
        }
      NoAuxiliaryLossTopK(routed_expert_count, top_k) =>
        match index {
          0 => (20L, 4L)
          1 => (routed_expert_count.to_int64() * 4L, 4L)
          2 => (routed_expert_count.to_int64() * 4L, 4L)
          _ => (top_k.to_int64() * 4L, 4L)
        }
      SelectedExpertWeightFinalize(routed_expert_count, top_k, _, _) =>
        match index {
          0 => (20L, 4L)
          1 => (routed_expert_count.to_int64() * 4L, 4L)
          _ => (top_k.to_int64() * 4L, 4L)
        }
      _ => (64L, 16L)
    }
    operands.push(
      try! AdvancedDecoderLaunchOperand::new(role~, byte_count~, alignment~),
    )
  }
  operands
}

///|
fn artifact_test_launch(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  target : @catalog.DeviceTarget,
  ordinal : Int,
  digest? : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest,
  source_identity? : @spec.ModelIdentity,
  operation? : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation,
  capability? : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability,
  entry_point? : @catalog.AotKernelEntryPoint,
  module_digest? : @catalog.AotArtifactDigest,
  dimensions? : @launch_contract.AotLaunchDimensions,
  operands? : Array[AdvancedDecoderLaunchOperand],
  qualification? : AdvancedDecoderDeclaredQualificationDigest,
) -> AdvancedDecoderLaunchEvidenceInput {
  let expected = requirements.requirements()[ordinal]
  let exact_operation = operation.unwrap_or(expected.operation())
  try! AdvancedDecoderLaunchEvidenceInput::new(
    requirements_digest=digest.unwrap_or(requirements.digest()),
    source_identity=source_identity.unwrap_or(requirements.source_identity()),
    requirement_ordinal=ordinal,
    operation=exact_operation,
    capability=capability.unwrap_or(expected.capability()),
    target~,
    entry_point=entry_point.unwrap_or(artifact_test_entry(ordinal)),
    module_digest=module_digest.unwrap_or(artifact_test_module_digest()),
    dimensions=dimensions.unwrap_or(
      artifact_test_dimensions_for(exact_operation),
    ),
    operands=operands.unwrap_or(artifact_test_operands(exact_operation)),
    declared_qualification?=qualification,
  )
}

///|
fn artifact_test_launches(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  target : @catalog.DeviceTarget,
  qualification? : AdvancedDecoderDeclaredQualificationDigest,
) -> Array[AdvancedDecoderLaunchEvidenceInput] {
  let launches : Array[AdvancedDecoderLaunchEvidenceInput] = []
  for ordinal, _ in requirements.requirements() {
    launches.push(
      artifact_test_launch(requirements, target, ordinal, qualification?),
    )
  }
  launches
}

///|
fn artifact_test_entries(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
) -> Array[@artifact.KernelEntryPointInput] {
  let entries : Array[@artifact.KernelEntryPointInput] = []
  for ordinal, _ in requirements.requirements() {
    entries.push(
      @artifact.KernelEntryPointInput::new(
        entry_point=artifact_test_entry(ordinal),
        function_symbol="advanced_decoder_\{ordinal}",
      ),
    )
  }
  entries
}

///|
fn artifact_test_modules() -> Array[AdvancedDecoderModuleInput] {
  [
    AdvancedDecoderModuleInput::new(
      digest=artifact_test_module_digest(),
      module_bytes=b"advanced-aot",
    ),
  ]
}

///|
fn artifact_test_admit(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  target : @catalog.DeviceTarget,
  launches : ArrayView[AdvancedDecoderLaunchEvidenceInput],
) -> AdvancedDecoderArtifactAdmission raise AdvancedDecoderArtifactAdmissionError {
  admit(
    requirements,
    target~,
    module_inputs=artifact_test_modules(),
    entry_point_inputs=artifact_test_entries(requirements),
    launch_inputs=launches,
    limits=artifact_test_limits(),
  )
}
