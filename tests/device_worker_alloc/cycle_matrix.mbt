///|
const CYCLE_ORDINARY_PREFILL : Int = 0

///|
const CYCLE_GREEDY_FINAL_PREFILL : Int = 1

///|
const CYCLE_STOCHASTIC_FINAL_PREFILL : Int = 2

///|
const CYCLE_GREEDY_DECODE : Int = 3

///|
const CYCLE_STOCHASTIC_DECODE : Int = 4

///|
struct PreparedCycle {
  frame : @worker_wire.ValidatedPlanFrame
}

///|
fn stochastic_parameters() -> @inference.SamplingParameters raise {
  @inference.SamplingParameters::sample(
    temperature=0.75,
    top_k=4,
    top_p=0.9,
    fixture_inference_limits(),
  )
}

///|
fn prepare_cycle_for_branch(
  sequence : UInt64,
  branch : Int,
  model_plan : @plan.ModelPlan,
  page : @page_allocator.PageId,
) -> PreparedCycle raise {
  let limits = fixture_worker_limits()
  if branch < CYCLE_ORDINARY_PREFILL || branch > CYCLE_STOCHASTIC_DECODE {
    abort("invalid allocation evidence branch")
  }
  let request_group = (sequence - 1UL) / 3UL
  let request_id = request_group + 1UL
  let request_generation = request_group
  let generation = @inference.RequestGeneration::from_wire(request_generation)
  let seed = @inference.SamplingSeed::new(request_id + 1000UL)
  let owner = @worker_protocol.SchedulePlanBuffer::new(limits)
  let mut row_position = 0
  let mut sample_index = 0
  let mut expected_input_token = 0
  if branch == CYCLE_ORDINARY_PREFILL {
    owner.append_prefill(
      request_id=@inference.RequestId::new(request_id),
      request_generation=generation,
      sequence_token_start=0,
      tokens=[0],
      pages=[page],
      capabilities=model_plan.required_capabilities(),
      completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
    )
  } else if branch == CYCLE_GREEDY_FINAL_PREFILL ||
    branch == CYCLE_STOCHASTIC_FINAL_PREFILL {
    row_position = 1
    sample_index = 0
    expected_input_token = 1
    let sampling = if branch == CYCLE_GREEDY_FINAL_PREFILL {
      @inference.SamplingParameters::greedy()
    } else {
      stochastic_parameters()
    }
    owner.append_final_prefill(
      request_id=@inference.RequestId::new(request_id),
      request_generation=generation,
      sequence_token_start=row_position,
      tokens=[1],
      pages=[page],
      capabilities=model_plan.required_capabilities(),
      sampling~,
      sampling_seed=seed,
      sample_index~,
      completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
    )
  } else {
    row_position = 2
    sample_index = 1
    let sampling = if branch == CYCLE_GREEDY_DECODE {
      @inference.SamplingParameters::greedy()
    } else {
      stochastic_parameters()
    }
    expected_input_token = 0
    owner.append_decode(
      request_id=@inference.RequestId::new(request_id),
      request_generation=generation,
      sequence_token_position=row_position,
      input_token=expected_input_token,
      pages=[page],
      capabilities=model_plan.required_capabilities(),
      sampling~,
      sampling_seed=seed,
      sample_index~,
      completion_slot=@worker_protocol.CompletionSlot::new(0, limits),
    )
  }
  let model_generation = @worker_protocol.ModelPlanGeneration::first()
  let plan = owner.submit(
    sequence=@worker_protocol.PlanSequence::from_wire(sequence),
    model_plan_generation=model_generation,
    token_budget=1,
    if sequence == 1UL {
      None
    } else {
      Some(@worker_protocol.PlanSequence::from_wire(sequence - 1UL))
    },
    model_generation,
  )
  let frame_owner = @worker_wire.PlanFrameBuffer::new(
    limits,
    fixture_inference_limits(),
  )
  { frame: frame_owner.encode_plan(plan) }
}
