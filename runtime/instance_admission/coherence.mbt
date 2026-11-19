///|
fn require_runtime_digest(
  expected : StringView,
  actual : @descriptor_file.RuntimeDescriptorDigest,
) -> Unit raise RuntimeInstanceAdmissionError {
  if expected != actual.as_hex() {
    raise reject(RuntimeDigest)
  }
}

///|
fn require_layout_geometry(
  cache_tokens_per_page : Int32,
  cache_total_pages : Int32,
  cache_layout_version : Int32,
  admitted_tokens_per_page : Int32,
  admitted_total_pages : Int32,
  admitted_layout_version : Int32,
) -> Unit raise RuntimeInstanceAdmissionError {
  if cache_tokens_per_page != admitted_tokens_per_page ||
    cache_total_pages != admitted_total_pages ||
    cache_layout_version != admitted_layout_version {
    raise reject(Layout)
  }
}

///|
fn require_tokenizer_identity(
  expected : StringView,
  actual : @tokenizer.TokenizerDigest,
) -> Unit raise RuntimeInstanceAdmissionError {
  if expected != actual.as_string() {
    raise reject(TokenizerBinding)
  }
}

///|
fn require_tokenizer_envelope(
  tokenizer_vocabulary : Int32,
  model_vocabulary : Int32,
  tokenizer_input_bytes : Int32,
  inference_text_bytes : Int32,
  tokenizer_output_tokens : Int32,
  inference_input_tokens : Int32,
  tokenizer_decoded_bytes : Int32,
  inference_delta_bytes : Int32,
) -> Unit raise RuntimeInstanceAdmissionError {
  if tokenizer_vocabulary > model_vocabulary ||
    tokenizer_input_bytes < inference_text_bytes ||
    tokenizer_output_tokens < inference_input_tokens ||
    tokenizer_decoded_bytes < inference_delta_bytes {
    raise reject(TokenizerEnvelope)
  }
}

///|
fn require_blueprint_capacity(
  resolved_slots : Int32,
  blueprint_slots : Int32,
) -> Unit raise RuntimeInstanceAdmissionError {
  if resolved_slots != blueprint_slots {
    raise reject(Capacity)
  }
}

///|
fn require_service_frame_bytes(
  service_bytes : Int32,
  framed_bytes : Int32,
) -> Unit raise RuntimeInstanceAdmissionError {
  if service_bytes != framed_bytes {
    raise reject(FramedWire)
  }
}

///|
fn require_preparation_lanes(
  configured_lanes : Int32,
  resolved_slots : Int32,
) -> Unit raise RuntimeInstanceAdmissionError {
  if configured_lanes != resolved_slots {
    raise reject(Preparation)
  }
}

///|
fn required_worker_frame_capacity(
  worker : @worker_protocol.WorkerProtocolLimits,
) -> Int32 raise RuntimeInstanceAdmissionError {
  let plan_bytes = @worker_wire.plan_frame_byte_capacity(worker) catch {
    _ => raise reject(WorkerProcess)
  }
  let completion_bytes = @worker_wire.completion_frame_byte_capacity(worker) catch {
    _ => raise reject(WorkerProcess)
  }
  Int32::max(
    @worker_wire.bootstrap_source_capacity(),
    Int32::max(plan_bytes, completion_bytes),
  )
}
