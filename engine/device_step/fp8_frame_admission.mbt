///|
fn make_fp8_frame_envelope_v3(
  authority : @fp8_release_authority.Fp8ReusablePagedEnvelopeAuthorityV3,
  generation : @worker_protocol.ModelPlanGeneration,
  limits : DeviceStepLimits,
) -> Fp8PagedFrameEnvelopeV3 raise DeviceStepError {
  let recipe = authority.recipe()
  let plan = recipe.startup().model_plan()
  let profile = recipe.profile()
  let layout = recipe.kv_layout()
  if authority.target() !=
    recipe.startup().kernel_artifacts().manifest().target() ||
    limits.max_rows() > profile.max_query_rows() ||
    limits.max_tokens() > profile.max_query_tokens() ||
    limits.max_pages() > profile.max_page_table_entries() ||
    limits.max_sequence_tokens() >
    plan.shape_constraints().max_sequence_tokens() ||
    layout.identity() != plan.identity() ||
    layout.stable_layout_version() <= 0 ||
    layout.tokens_per_page() <= 0 ||
    layout.total_page_count() <= 0 ||
    authority.digest().as_hex().is_empty() {
    raise InvalidExecutor(Fp8FrameEnvelope)
  }
  {
    profile,
    model_generation_value: generation.value(),
    kv_layout_version: layout.stable_layout_version(),
    tokens_per_page: layout.tokens_per_page(),
    total_page_count: layout.total_page_count(),
    release_digest: authority.digest().as_hex(),
  }
}

///|
fn preflight_fp8_frame_v3(
  descriptor : DeviceStepOwner,
  frame : @worker_wire.ValidatedPlanFrame,
  envelope : Fp8PagedFrameEnvelopeV3,
) -> DeviceStepSummary raise DeviceStepError {
  if descriptor.model_generation_value != envelope.model_generation_value ||
    descriptor.layout.stable_layout_version() != envelope.kv_layout_version ||
    descriptor.layout.tokens_per_page() != envelope.tokens_per_page ||
    descriptor.layout.total_page_count() != envelope.total_page_count ||
    envelope.release_digest.is_empty() {
    raise InvalidExecutor(Fp8FrameEnvelope)
  }
  let summary = descriptor.preflight_validated_frame(frame)
  if summary.row_count > envelope.profile.max_query_rows() ||
    summary.token_count > envelope.profile.max_query_tokens() ||
    summary.page_count > envelope.profile.max_page_table_entries() {
    raise InvalidExecutor(Fp8FrameAdmission)
  }
  summary
}

///|
fn authenticate_and_publish_fp8_frame_token_v3(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  frame : @worker_wire.ValidatedPlanFrame,
  staged : StagedDeviceStep,
  summary : DeviceStepSummary,
) -> Unit raise DeviceStepError {
  let wire_owner = frame.owner_identity() catch { error => raise Wire(error) }
  let wire_epoch = frame.epoch_value() catch { error => raise Wire(error) }
  guard descriptor.staged_wire_owner is Some(staged_owner) else {
    raise InvalidExecutor(Fp8FrameAdmission)
  }
  if !physical_equal(staged_owner, wire_owner) ||
    descriptor.staged_wire_epoch != wire_epoch ||
    descriptor.staged_summary != summary ||
    staged.1 != descriptor.stage_epoch {
    raise InvalidExecutor(Fp8FrameAdmission)
  }
  publish_fp8_frame_token_v3(
    executor,
    wire_owner,
    descriptor_epoch=staged.1,
    wire_epoch~,
    summary,
  )
}

///|
fn publish_fp8_frame_token_v3(
  executor : PagedGraphExecutor,
  wire_owner : @worker_wire.PlanFrameOwnerIdentity,
  descriptor_epoch~ : UInt64,
  wire_epoch~ : UInt64,
  summary : DeviceStepSummary,
) -> Unit raise DeviceStepError {
  guard executor.fp8_frame_admission is Some(admission) && !admission.armed else {
    raise InvalidExecutor(Fp8FrameAdmission)
  }
  arm_fp8_frame_admission_v3(admission, descriptor_epoch, wire_epoch, summary)
  admission.wire_owner = Some(wire_owner)
}

///|
fn arm_fp8_frame_admission_v3(
  admission : Fp8FrameAdmissionStateV3,
  descriptor_epoch : UInt64,
  wire_epoch : UInt64,
  summary : DeviceStepSummary,
) -> Unit raise DeviceStepError {
  if admission.armed ||
    descriptor_epoch == 0UL ||
    wire_epoch == 0UL ||
    summary.sequence_value == 0UL ||
    summary.row_count <= 0 ||
    summary.token_count <= 0 ||
    summary.page_count <= 0 {
    raise InvalidExecutor(Fp8FrameAdmission)
  }
  admission.descriptor_epoch = descriptor_epoch
  admission.wire_epoch = wire_epoch
  admission.sequence_value = summary.sequence_value
  admission.row_count = summary.row_count
  admission.token_count = summary.token_count
  admission.page_count = summary.page_count
  admission.armed = true
}

///|
fn clear_fp8_frame_admission_v3(executor : PagedGraphExecutor) -> Unit {
  match executor.fp8_frame_admission {
    Some(admission) => clear_fp8_frame_admission_v3_state(admission)
    None => ()
  }
}

///|
fn clear_fp8_frame_admission_v3_state(
  admission : Fp8FrameAdmissionStateV3,
) -> Unit {
  admission.armed = false
  admission.descriptor_epoch = 0UL
  admission.wire_owner = None
  admission.wire_epoch = 0UL
  admission.sequence_value = 0UL
  admission.row_count = 0
  admission.token_count = 0
  admission.page_count = 0
}

///|
fn authenticate_and_consume_fp8_frame_token_v3(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  staged : StagedPagedGraphExecution,
) -> Unit raise DeviceStepError {
  guard executor.fp8_frame_envelope is Some(_) else {
    if executor.fp8_frame_admission is Some(_) {
      raise InvalidExecutor(Fp8FrameAdmission)
    }
    return
  }
  guard descriptor.staged_wire_owner is Some(staged_owner) else {
    raise InvalidExecutor(Fp8FrameAdmission)
  }
  consume_fp8_frame_token_v3(
    executor,
    staged_epoch=staged.epoch,
    staged_owner~,
    wire_epoch=descriptor.staged_wire_epoch,
    sequence_value=descriptor.staged_summary.sequence_value,
  )
}

///|
fn consume_fp8_frame_token_v3(
  executor : PagedGraphExecutor,
  staged_epoch~ : UInt64,
  staged_owner~ : @worker_wire.PlanFrameOwnerIdentity,
  wire_epoch~ : UInt64,
  sequence_value~ : UInt64,
) -> Unit raise DeviceStepError {
  guard executor.fp8_frame_admission is Some(admission) && admission.armed else {
    raise InvalidExecutor(Fp8FrameAdmission)
  }
  guard admission.wire_owner is Some(admitted_owner) else {
    raise InvalidExecutor(Fp8FrameAdmission)
  }
  if admission.descriptor_epoch != staged_epoch ||
    !physical_equal(staged_owner, admitted_owner) ||
    wire_epoch != admission.wire_epoch ||
    sequence_value != admission.sequence_value ||
    admission.row_count <= 0 ||
    admission.token_count <= 0 ||
    admission.page_count <= 0 {
    raise InvalidExecutor(Fp8FrameAdmission)
  }
  clear_fp8_frame_admission_v3(executor)
}
