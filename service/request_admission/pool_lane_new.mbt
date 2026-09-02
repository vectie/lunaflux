///|
fn new_preparation_lane(
  tokenizer : @tokenizer.TokenizerSpec,
  inference : @inference.InferenceLimits,
  framed_limits : @framed_wire.FramedWireLimits,
  input_capacity : Int,
  output_capacity : Int,
) -> LunaRequestPreparationLane raise RequestAdmissionError {
  let tokenizer_budget = @tokenizer.LunaTokenizerStepBudget::new(1) catch {
    _ => raise Invalid(LunaPreparationPool)
  }
  let output_budget = @incremental_output.LunaIncrementalOutputStepBudget::new(
    1,
  ) catch {
    _ => raise Invalid(LunaPreparationPool)
  }
  let framed_workspace = @framed_wire.LunaFramedRequestWorkspace::new(
    framed_limits,
    @framed_wire.LunaFramedRequestStepBudget::new(1) catch {
      _ => raise Invalid(LunaPreparationPool)
    },
  )
  {
    tokenizer: @tokenizer.LunaTokenizerWorker::new(
      tokenizer,
      input_capacity~,
      output_capacity~,
      tokenizer_budget,
    ) catch {
      _ => raise Invalid(Tokenization)
    },
    token_storage: @inference.LunaTokenBufferStorage::new(
      capacity=output_capacity,
      limits=inference,
    ) catch {
      _ => raise Invalid(Input)
    },
    semantic_storage: @inference.LunaRequestSemanticStorage::new(
      inference,
      @inference.LunaRequestSemanticStepBudget::new(1) catch {
        _ => raise Invalid(LunaPreparationPool)
      },
    ) catch {
      _ => raise Invalid(LunaPreparationStorage)
    },
    output_workspace: @incremental_output.LunaIncrementalOutputWorkspace::new(
      tokenizer, inference, output_budget,
    ) catch {
      _ => raise Invalid(OutputContract)
    },
    framed_workspace,
    framed_work: Array::new(capacity=1),
    framed_view: Array::new(capacity=1),
    typed_handoff: Array::new(capacity=1),
    generation: 0UL,
    phase: LUNA_PREPARATION_IDLE,
    queued: false,
    receipt_at_millis: 0UL,
    request: None,
    deadline: None,
    hard_deadline: None,
    effective: None,
    input_capacity: 0,
    tokenizer_input_write: None,
    tokenizer_work: None,
    token_write: None,
    token_lease: None,
    semantic_write: Array::new(capacity=1),
    semantic_work: Array::new(capacity=1),
    semantic_lease: Array::new(capacity=1),
    output_work: None,
    output_lease: None,
    tokens: None,
    copy_index: 0,
    semantic_item_index: 0,
    semantic_unit_offset: 0,
    semantic_byte_length: 0,
    semantic_codepoint: 0,
    semantic_codepoint_width: 0,
    semantic_codepoint_byte_index: 0,
    framed_direct: false,
    typed_direct: false,
    typed_request_id: @inference.RequestId::new(1UL) catch {
      _ => abort("Luna typed request-id default invariant")
    },
    typed_sampling: @inference.LunaValidatedSampling::greedy(),
    typed_sampling_seed: @inference.SamplingSeed::new(1UL) catch {
      _ => abort("Luna typed sampling-seed default invariant")
    },
    typed_stream_preference: Streaming,
    framed_receipt_complete: false,
    framed_model_index: 0,
    framed_input_kind: 0,
    framed_input_count: 0,
    framed_stop_token_count: 0,
    framed_stop_string_count: 0,
    framed_cache_scope_length: 0,
    framed_request_id: 0UL,
    framed_sampling_mode: 0,
    framed_temperature: 1.0,
    framed_has_top_k: false,
    framed_top_k: 0,
    framed_has_top_p: false,
    framed_top_p: 0.0,
    framed_sampling_seed: 0UL,
    framed_streaming: false,
    framed_token_ids_only: false,
    trusted_receipt_bound: false,
    trusted_deadline_millis: 0UL,
    claim: None,
    failure: None,
    last_work_units: 0,
    total_work_units: 0UL,
  }
}
