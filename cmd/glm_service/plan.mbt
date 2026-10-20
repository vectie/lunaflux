///|
/// Deployment envelope only. Numerical and physical schedules are existing
/// immutable compiler plans; this command supplies no reference-engine logic.
priv struct ServicePlan {
  model_root : String
  kernel_root : String
  inventory : String
  module0 : String
  module1 : String
  sampling_module : String
  control_address : String
  rendezvous_address : String
  listen_address : String
  production_ready : Bool
  executable : String
  model_label : String
  runtime_label : String
  tokenizer_label : String
  credential_file : String
  rows : Int
  slots : Int
  admission_lanes : Int
  context : Int
  history_capacity : @history.ResidentHistoryCapacity
  sparse_accumulation : @precision.AttentionAccumulationLaw
  index_scoring : @precision.IndexScoringLaw
  sparse_cache : @precision.SparseCacheLaw
  execution : ExecutionPolicy
  budget : Int64
  reserve : Int64
}

///|
fn json_string(j : Json, key : String) -> String raise {
  guard j is Object(fields) && fields.get(key) is Some(String(value)) else {
    fail("missing string \{key}")
  }
  value
}

///|
fn json_int(j : Json, key : String) -> Int raise {
  guard j is Object(fields) &&
    fields.get(key) is Some(Number(value, ..)) &&
    value >= 0.0 &&
    value <= 2147483647.0 &&
    value.to_int().to_double() == value else {
    fail("missing integer \{key}")
  }
  value.to_int()
}

///|
fn json_optional_bool(j : Json, key : String) -> Bool raise {
  guard j is Object(fields) else { fail("expected service object") }
  match fields.get(key) {
    None => false
    Some(True) => true
    Some(False) => false
    _ => fail("expected boolean \{key}")
  }
}

///|
fn parse_plan(bytes : Bytes) -> ServicePlan raise {
  guard bytes.length() <= 65536 else {
    fail("service plan exceeds startup envelope")
  }
  let j = @json.parse(@utf8.decode(bytes))
  let rows = json_int(j, "rows")
  let slots = json_int(j, "slots")
  let context = json_int(j, "context")
  let admission_lanes = service_admission_lanes(j, slots)
  let history_capacity = service_history_capacity(j, slots, context)
  let p = ServicePlan::{
    model_root: json_string(j, "model_root"),
    kernel_root: json_string(j, "kernel_root"),
    inventory: json_string(j, "inventory"),
    module0: json_string(j, "module0"),
    module1: json_string(j, "module1"),
    sampling_module: json_string(j, "sampling_module"),
    control_address: json_string(j, "control_address"),
    rendezvous_address: json_string(j, "rendezvous_address"),
    listen_address: json_string(j, "listen_address"),
    production_ready: json_optional_bool(j, "production_ready"),
    executable: json_string(j, "executable"),
    model_label: json_string(j, "model_label"),
    runtime_label: json_string(j, "runtime_label"),
    tokenizer_label: json_string(j, "tokenizer_label"),
    credential_file: json_string(j, "credential_file"),
    rows,
    slots,
    admission_lanes,
    context,
    history_capacity,
    sparse_accumulation: service_sparse_accumulation(j),
    index_scoring: service_index_scoring(j),
    sparse_cache: service_sparse_cache(j),
    execution: execution_policy(j),
    budget: json_int(j, "budget_mib").to_int64() * 1048576L,
    reserve: json_int(j, "reserve_mib").to_int64() * 1048576L,
  }
  guard p.rows > 0 &&
    p.rows <= 256 &&
    p.slots > 0 &&
    p.slots <= p.rows &&
    p.context >= p.rows &&
    p.context <= 1048576 &&
    p.context % 16 == 0 &&
    p.budget <= 107374182400L &&
    p.reserve >= 1073741824L &&
    p.reserve < p.budget else {
    fail("invalid service capacity")
  }
  for
    path in [
      p.inventory,
      p.module0,
      p.module1,
      p.sampling_module,
      p.credential_file,
    ] {
    ignore(@approved_fs.ApprovedRelativeLocator::new(path))
  }
  ignore(@socket.Addr::parse(p.control_address))
  ignore(@socket.Addr::parse(p.rendezvous_address))
  ignore(@socket.Addr::parse(p.listen_address))
  p
}

///|
test "production readiness is an explicit deployment choice" {
  assert_false(json_optional_bool({}, "production_ready"))
  assert_true(
    json_optional_bool({ "production_ready": true }, "production_ready"),
  )
  assert_false(
    json_optional_bool({ "production_ready": false }, "production_ready"),
  )
  try
    ignore(
      json_optional_bool({ "production_ready": "true" }, "production_ready"),
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("nonboolean readiness was accepted")
  }
}

///|
fn read_bounded(
  root : @approved_fs.ApprovedRoot,
  name : String,
  ceiling : Int64,
) -> Bytes raise {
  let file = root.open_file(@approved_fs.ApprovedRelativeLocator::new(name))
  defer file.close()
  file.read_immutable_snapshot(maximum_bytes=ceiling)
}

///|
fn ServicePlan::history(self : Self) -> @history.PagedHistoryDomain {
  self.history_capacity.domain()
}

///|
fn ServicePlan::worker_limits(
  self : Self,
  spec : @spec.Glm53FlashSpec,
) -> @protocol.WorkerProtocolLimits raise {
  @protocol.WorkerProtocolLimits::new(
    max_prefill_rows=self.slots,
    max_decode_rows=self.slots,
    max_plan_rows=self.slots,
    max_plan_tokens=self.rows,
    max_plan_pages=self.history().table_capacity(),
    max_capabilities=self.slots,
    max_completion_slots=self.slots,
    max_sequence_tokens=self.context,
    max_token_id=spec.vocabulary_size() - 1,
  )
}

///|
fn ServicePlan::inference(
  self : Self,
  spec : @spec.Glm53FlashSpec,
  tokenizer : @tokenizer.TokenizerSpec,
) -> @inference.InferenceLimits raise {
  glm_inference_limits(
    self.context,
    spec.vocabulary_size(),
    maximum_token_bytes=tokenizer.maximum_decoded_piece_bytes(),
  )
}

///|
fn glm_inference_limits(
  context : Int,
  vocabulary : Int,
  maximum_token_bytes? : Int = 4096,
) -> @inference.InferenceLimits raise {
  let maximum_stop_bytes = 256
  @inference.InferenceLimits::new(
    max_text_bytes=1048576,
    max_input_tokens=context,
    max_new_tokens=context,
    max_context_tokens=context,
    max_token_id=vocabulary - 1,
    max_stop_token_ids=32,
    max_stop_strings=16,
    max_stop_string_bytes=maximum_stop_bytes,
    max_trace_bytes=256,
    max_cache_scope_bytes=256,
    // A failed stop-prefix match can release held bytes together with a full
    // tokenizer piece and the UTF-8 carry from the preceding token.
    max_decoded_delta_bytes=maximum_token_bytes + maximum_stop_bytes + 3,
    max_deadline_millis=3600000UL,
    max_top_k=vocabulary,
    max_temperature=100.0,
  )
}

///|
test "service decoding envelope includes declared stop prefixes and UTF8 carry" {
  let limits = glm_inference_limits(2048, 154880)
  assert_eq(
    limits.max_decoded_delta_bytes(),
    4096 + limits.max_stop_string_bytes() + 3,
  )
  assert_eq(limits.max_context_tokens(), 2048)
  assert_eq(limits.max_new_tokens(), 2048)
  let exact = glm_inference_limits(8192, 154880, maximum_token_bytes=128)
  assert_eq(exact.max_decoded_delta_bytes(), 128 + 256 + 3)
  assert_eq(exact.max_new_tokens(), 8192)
}

///|
fn ServicePlan::frame(
  self : Self,
  spec : @spec.Glm53FlashSpec,
) -> @frame.BatchedDecoderFrame raise {
  @frame.BatchedDecoderFrame::new(
    maximum_rows=self.rows,
    maximum_sequences=self.slots,
    cache_slots=self.slots,
    maximum_context=self.context,
    vocabulary_size=spec.vocabulary_size(),
    model_generation=1UL,
    paged_history=self.history(),
  )
}

///|
fn ServicePlan::stage(
  self : Self,
  spec : @spec.Glm53FlashSpec,
  ordinal : Int,
) -> @glm.GlmDecoderStage raise {
  guard ordinal is (0 | 1) && self.context <= spec.context_length() else {
    fail("invalid GLM rank or context")
  }
  let cap = self.budget - self.reserve
  @glm.GlmDecoderStage::new(
    spec,
    @placement.DecoderStageRange::new(
      layer_count=spec.layer_count(),
      first_layer=0,
      end_layer=spec.layer_count(),
    ),
    maximum_rows=self.rows,
    maximum_sequences=self.slots,
    cache_slots=self.slots,
    maximum_output_rows=self.slots,
    maximum_history=self.context,
    paged_history=self.history(),
    index_layer_norm_epsilon=1.0e-5,
    maximum_layer_state_bytes=cap,
    maximum_layer_workspace_bytes=cap,
    maximum_layer_expert_bytes=cap,
    maximum_text_workspace_bytes=cap,
    accuracy=@precision.Accuracy::new(absolute=0.125, relative=0.125),
    recurrent_evaluation=self.execution.recurrent,
    matrix_law=ShapeAwareFloat32,
    matrix_staging=self.execution.staging,
    matrix_tile_rows=self.execution.recurrent_tile_rows,
    sparse_matrix_staging=self.execution.sparse_staging,
    sparse_accumulation=self.sparse_accumulation,
    index_scoring=self.index_scoring,
    sparse_cache=self.sparse_cache,
    expert_scaled_dot_law=BlockScaledMatrixFloat32(block_columns=16),
    expert_transport=self.execution.transport,
    expert_fragment_slots=2,
    expert_scale_partitions=self.execution.scale_partitions,
    moe_reduction=self.execution.moe_reduction,
    tensor_publication=self.execution.tensor_publication,
    hc_reduction=Tree(partitions=256),
    shared_matrix_columns=64,
    tensor_group=(ordinal, 2),
    shard_vocabulary=true,
    vocabulary_publication=FullLogits,
  )
}
