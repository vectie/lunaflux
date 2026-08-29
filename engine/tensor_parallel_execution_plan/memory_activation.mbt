///|
fn activation_fact(
  facts : ArrayView[MutableActivationFact],
  reference : @plan.ActivationRef,
) -> Int {
  let index = reference.as_int()
  if index >= 0 &&
    index < facts.length() &&
    facts[index].reference.as_int() == index {
    index
  } else {
    -1
  }
}

///|
fn output_operand(
  contract : @tensor_parallel_launch_contract.TensorParallelAotLaunchContract,
  reference : @plan.ActivationRef,
) -> @tensor_parallel_launch_contract.TensorParallelOperand? {
  for operand in contract.operands() {
    match operand.role() {
      ActivationOutput(actual) if actual == reference => return Some(operand)
      _ => ()
    }
  }
  None
}

///|
fn record_activation_input(
  facts : Array[MutableActivationFact],
  reference : @plan.ActivationRef,
  byte_count : Int64,
  operation : Int,
  operand : Int,
) -> Unit raise TensorParallelExecutionPlanError {
  let index = activation_fact(facts, reference)
  if index < 0 || facts[index].producer.as_int() >= operation {
    raise rejected(ActivationRegion, operation~, operand~)
  }
  if byte_count != facts[index].byte_count {
    raise rejected(ActivationRegion, operation~, operand~)
  }
  let current = facts[index]
  facts[index] = { ..current, live_through: operation }
}

///|
fn collect_activation_facts(
  model : @plan.ModelPlan,
  contracts : ArrayView[
    @tensor_parallel_launch_contract.TensorParallelAotLaunchContract,
  ],
  rank : Int,
  world_size : Int,
) -> Array[MutableActivationFact] raise TensorParallelExecutionPlanError {
  let facts : Array[MutableActivationFact] = []
  let operations = model.operations()
  for operation_index, operation in operations {
    let outputs = operation.outputs()
    if outputs.length() != 1 || outputs[0].as_int() != operation_index {
      raise rejected(ActivationRegion, operation=operation_index)
    }
    guard output_operand(contracts[operation_index], outputs[0]) is Some(output) else {
      raise rejected(ActivationRegion, operation=operation_index)
    }
    let mut byte_count = output.byte_count()
    match contracts[operation_index].collective_after() {
      CollectiveLastAxisAllGather(full_width~, local_width~, ..) => {
        if local_width <= 0 ||
          full_width <= local_width ||
          full_width != local_width * world_size {
          raise rejected(CollectiveGeometry, operation=operation_index)
        }
        byte_count = checked_multiply(
          output.byte_count(),
          Int64::from_int(world_size),
          operation=operation_index,
        )
        ignore(
          checked_multiply(
            output.byte_count(),
            Int64::from_int(rank),
            operation=operation_index,
          ),
        )
      }
      _ => ()
    }
    facts.push({
      reference: outputs[0],
      byte_count,
      alignment: output.alignment(),
      producer: operation.id(),
      live_through: operation_index,
      slot: -1,
    })
  }
  for operation_index, contract in contracts {
    for operand_index, operand in contract.operands() {
      match operand.role() {
        ActivationInput(reference) =>
          record_activation_input(
            facts,
            reference,
            operand.byte_count(),
            operation_index,
            operand_index,
          )
        _ => ()
      }
    }
  }
  let final_index = activation_fact(facts, model.final_output())
  if final_index < 0 {
    raise rejected(ActivationRegion)
  }
  let final_fact = facts[final_index]
  facts[final_index] = { ..final_fact, live_through: operations.length() }
  facts
}

///|
fn assign_activation_slots(
  facts : Array[MutableActivationFact],
) -> Array[MutableActivationSlot] {
  let slots : Array[MutableActivationSlot] = []
  for index in 0..<facts.length() {
    let fact = facts[index]
    let mut selected = -1
    for slot_index, slot in slots {
      if slot.available_after < fact.producer.as_int() {
        selected = slot_index
        break
      }
    }
    if selected < 0 {
      selected = slots.length()
      slots.push({
        byte_count: fact.byte_count,
        alignment: fact.alignment,
        available_after: fact.live_through,
        offset_bytes: 0L,
      })
    } else {
      let slot = slots[selected]
      slots[selected] = {
        byte_count: slot.byte_count.max(fact.byte_count),
        alignment: slot.alignment.max(fact.alignment),
        available_after: fact.live_through,
        offset_bytes: 0L,
      }
    }
    facts[index] = { ..fact, slot: selected }
  }
  slots
}

///|
fn place_activations(
  facts : Array[MutableActivationFact],
  slots : Array[MutableActivationSlot],
) -> (ReadOnlyArray[TensorParallelActivationBinding], Int64) raise TensorParallelExecutionPlanError {
  let mut cursor = 0L
  for index in 0..<slots.length() {
    let slot = slots[index]
    cursor = checked_align(cursor, slot.alignment)
    slots[index] = { ..slot, offset_bytes: cursor }
    cursor = checked_add(cursor, checked_align(slot.byte_count, slot.alignment))
  }
  let bindings : Array[TensorParallelActivationBinding] = []
  for fact in facts {
    if fact.slot < 0 || fact.slot >= slots.length() {
      raise rejected(ActivationRegion, operation=fact.producer.as_int())
    }
    let slot = slots[fact.slot]
    bindings.push({
      reference: fact.reference,
      slot: fact.slot,
      produced_by: fact.producer,
      live_through_step: fact.live_through,
      region: scalar_region(
        ActivationWorkspaceSpace,
        slot.offset_bytes,
        fact.byte_count,
        fact.alignment,
        operation=fact.producer.as_int(),
      ),
    })
  }
  (ReadOnlyArray::from_array(bindings), cursor)
}
