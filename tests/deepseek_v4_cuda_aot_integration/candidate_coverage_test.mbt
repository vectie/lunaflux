///|
fn deepseek_coverage_limits() -> @aot_candidate_coverage.AotCandidateCoverageLimits {
  try! @aot_candidate_coverage.AotCandidateCoverageLimits::new(
    maximum_requirements=64,
    maximum_candidates=64,
    maximum_encoding_bytes=65536,
  )
}

///|
test "Flash-0731 candidate coverage remains exact incomplete and inert" {
  let requirements = official_deepseek_requirements(
    "deepseek-ai/DeepSeek-V4-Flash-0731", '9',
  )
  let target = official_deepseek_target()
  let compiler = official_deepseek_compiler()
  let limits = official_deepseek_limits()
  let candidates : Array[@aot_candidate_coverage.AotCandidateEvidenceInput] = []
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      QueryLowRankAProjection(_) => {
        let candidate = @advanced_decoder_cuda_aot.lower_query_a_projection_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_query_a_projection(
            candidate,
          ),
        )
      }
      KeyValueProjection(_, _, _) => {
        let candidate = @advanced_decoder_cuda_aot.lower_key_value_projection_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_key_value_projection(
            candidate,
          ),
        )
      }
      TokenEmbedding | QueryKeyRmsNorm(_, _, _) | FinalRmsNorm(_) => {
        let candidate = @advanced_decoder_cuda_aot.lower_foundation_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_foundation(candidate),
        )
      }
      LanguageModelHead => {
        let candidate = @advanced_decoder_cuda_aot.lower_language_model_head_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_projection(candidate),
        )
      }
      RotaryPositionEncoding(_, _, _, _, _, _) => {
        let candidate = @advanced_decoder_cuda_aot.lower_scaled_rotary_position_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_rotary(candidate),
        )
      }
      HyperConnectionBlockControlV1(_, _, _, _) => {
        let candidate = @advanced_decoder_cuda_aot.lower_hyper_connection_block_control_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_hyper_connection(
            candidate,
          ),
        )
      }
      HyperConnectionPreReduceV1(_) => {
        let candidate = @advanced_decoder_cuda_aot.lower_hyper_connection_pre_reduce_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_hyper_connection_pre_reduce(
            candidate,
          ),
        )
      }
      HyperConnectionPostCombineV1(_) => {
        let candidate = @advanced_decoder_cuda_aot.lower_hyper_connection_post_combine_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_hyper_connection_post_combine(
            candidate,
          ),
        )
      }
      HyperConnectionHeadReduceV1(_, _, _) => {
        let candidate = @advanced_decoder_cuda_aot.lower_hyper_connection_head_reduce_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_hyper_connection_head_reduce(
            candidate,
          ),
        )
      }
      DeterministicCompressedIndex(_) => {
        let candidate = @advanced_decoder_cuda_aot.lower_deterministic_compressed_index_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_compressed_index(
            candidate,
          ),
        )
      }
      DeterministicWindowCompressedIndexJoinV1(_, _) => {
        let candidate = @advanced_decoder_cuda_aot.lower_deterministic_window_compressed_join_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_window_compressed_join(
            candidate,
          ),
        )
      }
      NonOverlappingCompressedKeyValuePrepareV1(_, _) => {
        let candidate = @advanced_decoder_cuda_aot.lower_non_overlapping_compressed_kv_prepare_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_batches=1,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_compressed_kv_prepare(
            candidate,
          ),
        )
      }
      SelectedSparseAttentionV1(_, _) => {
        let candidate = @advanced_decoder_cuda_aot.lower_selected_sparse_attention_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          maximum_batches=1,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_sparse_attention(
            candidate,
          ),
        )
      }
      AttentionOutputInverseRotaryV1(_, _, _, _, _) => {
        let candidate = @advanced_decoder_cuda_aot.lower_attention_output_inverse_rotary_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_inverse_rotary(
            candidate,
          ),
        )
      }
      AttentionOutputAProjectionV1(_, _, _, _) => {
        let candidate = @advanced_decoder_cuda_aot.lower_attention_output_a_projection_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_output_a_projection(
            candidate,
          ),
        )
      }
      AttentionOutputBProjectionV1(_, _, _) => {
        let candidate = @advanced_decoder_cuda_aot.lower_attention_output_b_projection_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_output_b_projection(
            candidate,
          ),
        )
      }
      ExpertScoreTransform(_, _) => {
        let candidate = @advanced_decoder_cuda_aot.lower_expert_score_transform_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_routing(candidate),
        )
      }
      TokenHashRouting(_, _, _, _) => {
        let candidate = @advanced_decoder_cuda_aot.lower_token_hash_routing_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_routing(candidate),
        )
      }
      NoAuxiliaryLossTopK(_, _) => {
        let candidate = @advanced_decoder_cuda_aot.lower_no_auxiliary_loss_top_k_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_routing(candidate),
        )
      }
      SelectedExpertWeightFinalize(_, _, _, _) => {
        let candidate = @advanced_decoder_cuda_aot.lower_selected_expert_weight_finalize_cuda_aot_candidate(
          requirements,
          requirement_ordinal=ordinal,
          maximum_tokens=2,
          target~,
          compiler~,
          limits~,
        )
        candidates.push(
          try! @advanced_decoder_candidate_evidence.from_routing(candidate),
        )
      }
      _ => ()
    }
  }
  let coverage = @aot_candidate_coverage.audit(
    source_identity=requirements.source_identity(),
    requirements_digest_sha256=requirements.digest().as_hex(),
    requirement_count=requirements.requirements().length(),
    candidates~,
    limits=deepseek_coverage_limits(),
  )
  assert_eq(coverage.status(), Incomplete)
  assert_eq(coverage.candidate_count(), 22)
  assert_eq(coverage.non_bindable_ordinals(), [
    0, 1, 2, 3, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 38, 39,
  ])
  assert_eq(coverage.missing_ordinals(), [
    4, 10, 11, 12, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
  ])
  try coverage.require_complete() catch {
    IncompleteCoverage(18) => ()
    _ => fail("DeepSeek candidate coverage returned the wrong gap")
  } noraise {
    _ => fail("partial DeepSeek candidate coverage was promoted")
  }
  inspect(
    coverage.digest().as_hex(),
    content="60a0547f79ab1bce753f5f074f2ee3fcee696759c68b44068909ae4c99199819",
  )
}
