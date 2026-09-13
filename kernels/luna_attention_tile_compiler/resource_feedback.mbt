///|
/// Offline device facts. A unit is an independently resident execution unit;
/// backend discovery translates its native limits into this portable value.
pub struct AttentionResourceBudget {
  priv units : Int
  priv threads_per_unit : Int
  priv registers_per_unit : Int
  priv register_allocation_grain : Int
  priv register_allocation_threads : Int
  priv shared_bytes_per_unit : Int64
  priv blocks_per_unit : Int
} derive(Debug, Eq)

///|
pub fn AttentionResourceBudget::new(
  units~ : Int,
  threads_per_unit~ : Int,
  registers_per_unit~ : Int,
  register_allocation_grain~ : Int,
  register_allocation_threads? : Int = 1,
  shared_bytes_per_unit~ : Int64,
  blocks_per_unit~ : Int,
) -> AttentionResourceBudget raise AttentionTileCompilerError {
  if units <= 0 ||
    units > 1048576 ||
    threads_per_unit <= 0 ||
    threads_per_unit > 8192 ||
    registers_per_unit <= 0 ||
    registers_per_unit > 16777216 ||
    register_allocation_grain <= 0 ||
    register_allocation_grain > registers_per_unit ||
    register_allocation_threads <= 0 ||
    register_allocation_threads > threads_per_unit ||
    shared_bytes_per_unit <= 0L ||
    shared_bytes_per_unit > 16777216L ||
    blocks_per_unit <= 0 ||
    blocks_per_unit > 128 {
    raise InvalidAttentionTileCompilation(ResourceFeedback)
  }
  {
    units,
    threads_per_unit,
    registers_per_unit,
    register_allocation_grain,
    register_allocation_threads,
    shared_bytes_per_unit,
    blocks_per_unit,
  }
}

///|
/// Exact compiled-schedule observation. Absence is unknown register pressure,
/// never a fabricated zero-register kernel. Latency records still outrank this
/// model. No profiling is performed in this package or in request execution.
pub struct AttentionRegisterObservation {
  priv schedule : @luna_attention_tile_schedule.AttentionTileScheduleDigest
  priv registers_per_thread : Int
} derive(Debug, Eq)

///|
pub fn AttentionRegisterObservation::new(
  schedule : @luna_attention_tile_schedule.AttentionTileScheduleDigest,
  registers_per_thread~ : Int,
) -> AttentionRegisterObservation raise AttentionTileCompilerError {
  if registers_per_thread <= 0 || registers_per_thread > 1024 {
    raise InvalidAttentionTileCompilation(ResourceFeedback)
  }
  { schedule, registers_per_thread, }
}

///|
pub fn AttentionTileCompilerRequest::with_resource_feedback(
  self : AttentionTileCompilerRequest,
  budget : AttentionResourceBudget,
  observations : ArrayView[AttentionRegisterObservation],
) -> AttentionTileCompilerRequest raise AttentionTileCompilerError {
  if observations.length() > 128 {
    raise InvalidAttentionTileCompilation(ResourceFeedback)
  }
  for i, observation in observations {
    for other in observations[:i] {
      if other.schedule == observation.schedule {
        raise InvalidAttentionTileCompilation(ResourceFeedback)
      }
    }
  }
  let ordered = observations.to_owned()
  ordered.sort_by((a, b) => {
    String::compare(a.schedule.as_hex(), b.schedule.as_hex())
  })
  {
    ..self,
    resource_budget: Some(budget),
    register_observations: ReadOnlyArray::from_array(ordered),
  }
}

///|
/// Complete waves, not occupancy percentage, determine useful parallelism.
/// This preserves discontinuities at N*resident_slots + 1 workgroups.
fn attention_wave_score(
  base : Int64,
  workgroups : Int64,
  resident_slots : Int64,
) -> Int64 raise AttentionTileCompilerError {
  if base <= 0L || workgroups <= 0L || resident_slots <= 0L {
    raise InvalidAttentionTileCompilation(ResourceFeedback)
  }
  let waves = 1L + (workgroups - 1L) / resident_slots
  let quotient = base / workgroups
  let remainder = base % workgroups
  if remainder > 0L && waves > 9223372036854775807L / remainder {
    raise InvalidAttentionTileCompilation(ResourceFeedback)
  }
  let product = remainder * waves
  quotient * waves +
  product / workgroups +
  (if product % workgroups == 0L { 0L } else { 1L })
}

///|
fn attention_observed_resource_score(
  request : AttentionTileCompilerRequest,
  scheduled : ScheduledAttentionTileCompilation,
  budget : AttentionResourceBudget,
) -> Int64 raise AttentionTileCompilerError {
  let schedule = scheduled.schedule()
  let (groups, _, _, shared) = schedule.execution_geometry()
  let threads = groups * request.capabilities.subgroup_width()
  let registers = attention_register_observation(request, schedule.digest())
  let by_registers = match registers {
    None => 1
    Some(observation) => {
      let allocation_threads = budget.register_allocation_threads.to_int64()
      let raw = observation.registers_per_thread.to_int64() * allocation_threads
      let grain = budget.register_allocation_grain.to_int64()
      let allocated = (raw + grain - 1L) /
        grain *
        grain *
        ((threads.to_int64() + allocation_threads - 1L) / allocation_threads)
      (budget.registers_per_unit.to_int64() / allocated).to_int()
    }
  }
  let by_shared = if shared <= 0L {
    budget.blocks_per_unit
  } else {
    (budget.shared_bytes_per_unit / shared).to_int()
  }
  let residency = Int::min(
    budget.blocks_per_unit,
    Int::min(
      by_registers,
      Int::min(by_shared, budget.threads_per_unit / threads),
    ),
  )
  if residency <= 0 {
    return 9223372036854775807L
  }
  let (queries, tile) = schedule.loops()[0].geometry()
  let (heads, head_tile) = schedule.loops()[1].geometry()
  let per_row = (1L + (queries.to_int64() - 1L) / tile.to_int64()) *
    (1L + (heads.to_int64() - 1L) / head_tile.to_int64())
  if per_row > 9223372036854775807L / request.problem.batch_rows().to_int64() {
    raise InvalidAttentionTileCompilation(ResourceFeedback)
  }
  let query_groups = per_row * request.problem.batch_rows().to_int64()
  let partitions = schedule.key_value_partitions().to_int64()
  if partitions <= 0L || query_groups > 9223372036854775807L / partitions {
    raise InvalidAttentionTileCompilation(ResourceFeedback)
  }
  let workgroups = query_groups * partitions
  attention_wave_score(
    scheduled.plan().static_cost(),
    workgroups,
    budget.units.to_int64() * residency.to_int64(),
  )
}

///|
fn attention_register_observation(
  request : AttentionTileCompilerRequest,
  schedule : @luna_attention_tile_schedule.AttentionTileScheduleDigest,
) -> AttentionRegisterObservation? {
  for observation in request.register_observations {
    if observation.schedule == schedule {
      return Some(observation)
    }
  }
  None
}
