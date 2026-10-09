///|
fn parallel_arena(space : TensorParallelMemorySpace) -> Int {
  match space {
    DescriptorSpace => 0
    RankConstantsSpace => 1
    WeightSpace => 2
    PersistentKvSpace => 3
    ActivationWorkspaceSpace => 4
  }
}

///|
fn parallel_access(
  buffers : Array[@parallel_ir.Buffer],
  indices : Map[(Int, Int64, Int64), Int],
  arena : Int,
  offset : Int64,
  bytes : Int64,
  mode : @parallel_ir.AccessMode,
) -> @parallel_ir.Access raise @parallel_ir.ParallelError {
  let key = (arena, offset, bytes)
  let index = match indices.get(key) {
    Some(index) => index
    None => {
      let index = buffers.length()
      buffers.push(
        @parallel_ir.Buffer::new(arena~, offset_bytes=offset, byte_count=bytes),
      )
      indices[key] = index
      index
    }
  }
  @parallel_ir.Access::new(buffer=index, mode~)
}

///|
/// Opt-in pure startup projection of the already admitted physical plan.
/// Existing admission/execution and its content identity remain unchanged.
/// Returned commands still require an explicitly qualified backend binding.
pub fn TensorParallelExecutionPlan::parallel_program(
  self : TensorParallelExecutionPlan,
) -> @parallel_ir.Program raise @parallel_ir.ParallelError {
  let arenas = [
    @parallel_ir.Arena::new(bytes=self.descriptor_bytes, initialized=true),
    @parallel_ir.Arena::new(
      bytes=self.rank_constants.end_offset_bytes(),
      initialized=true,
    ),
    @parallel_ir.Arena::new(bytes=self.weight_arena_bytes, initialized=true),
    @parallel_ir.Arena::new(bytes=self.kv_arena_bytes, initialized=true),
    @parallel_ir.Arena::new(
      bytes=self.activation_workspace_bytes,
      initialized=false,
    ),
  ]
  let buffers : Array[@parallel_ir.Buffer] = []
  let indices : Map[(Int, Int64, Int64), Int] = Map([])
  let steps : Array[@parallel_ir.Step] = []
  let mut collective_index = 0
  for operation in self.operations {
    let accesses : Array[@parallel_ir.Access] = []
    for operand in operation.operands {
      let mode : @parallel_ir.AccessMode = match operand.role {
        ActivationOutput(_) | Workspace => Write
        // Conservative until operand contracts describe finer KV effects.
        PersistentKvComponent(_, _) => ReadWrite
        _ => Read
      }
      let region = operand.region
      accesses.push(
        parallel_access(
          buffers,
          indices,
          parallel_arena(region.space),
          region.offset_bytes,
          region.byte_count,
          mode,
        ),
      )
    }
    let producer = steps.length()
    steps.push(
      @parallel_ir.Step::new(Compute(operation.operation_id.as_int()), accesses),
    )
    while collective_index < self.collectives.length() &&
          self.collectives[collective_index].operation_id ==
          operation.operation_id {
      let collective = self.collectives[collective_index]
      let kind : @parallel_ir.CollectiveKind = match collective.kind {
        InPlaceSumAllReduce => SumAllReduce
        InPlaceLastAxisAllGather => LastAxisAllGather
      }
      // Gather reads only the produced local shard; it writes the full result.
      // Marking its whole receive range ReadWrite would invent uninitialized reads.
      let accesses = [
        parallel_access(
          buffers,
          indices,
          parallel_arena(collective.region.space),
          collective.local_send_offset_bytes,
          collective.local_send_bytes,
          Read,
        ),
        parallel_access(
          buffers,
          indices,
          parallel_arena(collective.region.space),
          collective.region.offset_bytes,
          collective.receive_bytes,
          Write,
        ),
      ]
      steps.push(
        @parallel_ir.Step::new(
          Communicate(
            @parallel_ir.Collective::new(
              sequence=collective.sequence,
              kind~,
              element=BFloat16,
              send_bytes=collective.local_send_bytes,
              receive_bytes=collective.receive_bytes,
            ),
          ),
          accesses,
          dependencies=[producer],
        ),
      )
      collective_index += 1
    }
  }
  if collective_index != self.collectives.length() {
    raise @parallel_ir.ParallelError::Group
  }
  @parallel_ir.compile(
    rank=self.rank,
    world_size=self.world_size,
    arenas,
    buffers,
    steps,
  )
}

///|
/// Physical-plan identity/profile agreement precedes generic collective matching.
pub fn parallel_group(
  plans : ArrayView[TensorParallelExecutionPlan],
) -> @parallel_ir.Group raise @parallel_ir.ParallelError {
  if plans.is_empty() {
    raise @parallel_ir.ParallelError::Group
  }
  let first = plans[0]
  let programs : Array[@parallel_ir.Program] = []
  for rank, plan in plans {
    if plan.rank != rank ||
      plan.world_size != plans.length() ||
      plan.identity != first.identity ||
      plan.profile != first.profile ||
      plan.target != first.target ||
      plan.collectives.length() != first.collectives.length() {
      raise @parallel_ir.ParallelError::Group
    }
    for index, collective in plan.collectives {
      if collective.operation_id != first.collectives[index].operation_id {
        raise @parallel_ir.ParallelError::CollectiveOrder(index + 1)
      }
    }
    programs.push(plan.parallel_program())
  }
  @parallel_ir.Group::new(programs)
}
