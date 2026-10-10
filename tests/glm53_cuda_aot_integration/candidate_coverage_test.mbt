///|
fn glm_coverage_limits() -> @aot_candidate_coverage.AotCandidateCoverageLimits {
  try! @aot_candidate_coverage.AotCandidateCoverageLimits::new(
    maximum_requirements=64,
    maximum_candidates=64,
    maximum_encoding_bytes=65536,
  )
}

///|
fn glm_candidate_evidence(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Array[@aot_candidate_coverage.AotCandidateEvidenceInput] {
  let evidence : Array[@aot_candidate_coverage.AotCandidateEvidenceInput] = []
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      TokenEmbedding
      | DecoderRmsNorm(_, _)
      | DsaQueryARmsNorm(_, _)
      | FinalRmsNorm(_, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_foundation_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_foundation(candidate),
        )
      }
      DsaQueryAProjection(_, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_dsa_query_a_projection_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_dense_projection(candidate),
        )
      }
      DsaQueryBProjection(_, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_dsa_query_b_projection_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_dense_projection(candidate),
        )
      }
      DsaKeyValueAProjection(_, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_dsa_key_value_a_projection_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_dense_projection(candidate),
        )
      }
      KeyValueLowRankProjection(_, _, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_dsa_key_value_b_projection_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_dense_projection(candidate),
        )
      }
      DsaKeyValueAttentionAssembly(_, _, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_dsa_kv_attention_assembly_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_kv_assembly(candidate),
        )
      }
      QueryKeyRmsNorm(_) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_dsa_key_value_split_norm_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_kv_split_norm(candidate),
        )
      }
      GlmRotaryPosition(Some(_), true) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_dsa_interleaved_rotary_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_rotary(candidate),
        )
      }
      DenseSwiGlu(_, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_dense_swiglu_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_dense_swiglu(candidate),
        )
      }
      KdaProjection(_, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_kda_projection_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_kda_projection(candidate),
        )
      }
      KdaShortConvolution(_, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_kda_short_convolution_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          maximum_sequences=8,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_kda_short_convolution(
            candidate,
          ),
        )
      }
      KdaRecurrentUpdate(_, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_kda_recurrent_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          maximum_sequences=8,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_kda_recurrent(candidate),
        )
      }
      KdaDecayControl(_, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_kda_decay_control_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_kda_decay_control(candidate),
        )
      }
      KdaOutputGateProjection(_, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_kda_output_gate_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_kda_output_gate(candidate),
        )
      }
      KdaGatedRmsNorm(_, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_kda_gated_rms_norm_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_kda_gated_norm(candidate),
        )
      }
      KdaOutputProjection(_, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_kda_output_projection_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_dense_projection(candidate),
        )
      }
      MoeRouterProjection(_, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_moe_router_projection_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_moe_router(candidate),
        )
      }
      MoeSigmoidAndCorrectionBias(_) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_moe_sigmoid_correction_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_moe_sigmoid_correction(
            candidate,
          ),
        )
      }
      MoeGroupedNoAuxiliaryLossTopK(_, _, _, _, _, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_moe_grouped_routing_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_moe_grouped_routing(
            candidate,
          ),
        )
      }
      RoutedExpertFeedForward(_, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_moe_routed_expert_feed_forward_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_moe_feed_forward(candidate),
        )
      }
      SharedExpertFeedForward(_, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_moe_shared_expert_feed_forward_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_moe_feed_forward(candidate),
        )
      }
      RoutedSharedExpertCombine(_, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_moe_routed_shared_combine_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_moe_feed_forward(candidate),
        )
      }
      MoeDecoderResidualAdd(_) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_moe_decoder_residual_add_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_moe_feed_forward(candidate),
        )
      }
      HyperConnectionMix(_) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_hyper_connection_mix_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_moe_feed_forward(candidate),
        )
      }
      HyperConnectionFunction(_, _, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_hyper_connection_function_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_hyper_control(candidate),
        )
      }
      HyperConnectionSinkhorn(_, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_hyper_connection_sinkhorn_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_hyper_control(candidate),
        )
      }
      DsaIndexCompute(_, _, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_dsa_index_selection_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_dsa_index(candidate),
        )
      }
      DsaIndexPooling(_, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_dsa_key_pooling_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_dsa_pooling(candidate),
        )
      }
      DsaIndexReuse => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_dsa_index_reuse_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_dsa_index(candidate),
        )
      }
      DsaSparseAttention(_, _, _, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_dsa_sparse_attention_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_sparse_attention(candidate),
        )
      }
      AttentionOutputProjection(_, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_attention_output_projection_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_dense_projection(candidate),
        )
      }
      LanguageModelHead(_, _) => {
        let candidate = try! @glm_hybrid_cuda_aot.lower_language_model_head_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=32,
          target=target(),
          compiler=compiler(),
          limits=limits(),
        )
        evidence.push(
          try! @glm_hybrid_candidate_evidence.from_dense_projection(candidate),
        )
      }
      _ => ()
    }
  }
  evidence
}

///|
test "official GLM candidate coverage remains exact incomplete and inert" {
  let profiles = [
    @glm53_kernel_requirements.build_full_kernel_requirements(full_metadata()),
    @glm53_kernel_requirements.build_flash_kernel_requirements(flash_metadata()),
  ]
  for profile_index, profile in profiles {
    let candidates = glm_candidate_evidence(profile)
    let coverage = @aot_candidate_coverage.audit(
      source_identity=profile.source_identity(),
      requirements_digest_sha256=profile.digest().as_hex(),
      requirement_count=profile.requirements().length(),
      candidates~,
      limits=glm_coverage_limits(),
    )
    assert_eq(coverage.status(), Incomplete)
    let expected_candidates = if profile_index == 0 { 24 } else { 32 }
    assert_eq(coverage.candidate_count(), expected_candidates)
    assert_eq(coverage.non_bindable_ordinals().length(), expected_candidates)
    let expected_missing = if profile_index == 0 { 4 } else { 15 }
    assert_eq(coverage.missing_ordinals().length(), expected_missing)
    assert_eq(
      coverage.missing_ordinals().length(),
      profile.requirements().length() - expected_candidates,
    )
    try coverage.require_complete() catch {
      IncompleteCoverage(count) =>
        assert_eq(count, profile.requirements().length() - expected_candidates)
      _ => fail("GLM candidate coverage returned the wrong gap")
    } noraise {
      _ => fail("partial GLM candidate coverage was promoted")
    }
    inspect(
      coverage.digest().as_hex(),
      content=if profile_index == 0 {
        "ad14aa79800b7623194952e53b531af53b3559bd1d16c029b1bb4571a3969938"
      } else {
        "c34a6c405be1ed09ac972751b8ea61b6d9bc2f2aa0b9c37084125137bf950369"
      },
    )
  }
}
