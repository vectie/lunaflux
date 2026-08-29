///|
fn tp_apply_recipe(
  payload : FixedArray[Byte],
  destination : FixedArray[Byte],
  recipe : @tensor_parallel_plan.TensorTransferRecipe,
) -> Unit {
  for segment_index in 0L..<recipe.segment_count() {
    let segment = try! recipe.segment(segment_index)
    for byte_index in 0L..<segment.byte_length() {
      destination[(segment.destination_offset_bytes() + byte_index).to_int()] = payload[(segment.source_payload_offset_bytes() +
        byte_index).to_int()]
    }
  }
}

///|
test "all rank recipes reconstruct every full dense Llama tensor" {
  let (model, bindings) = tp_fixture()
  let plans = [tp_plan(model, bindings, 0, 2), tp_plan(model, bindings, 1, 2)]
  let payload_bytes = bindings.materialized_bytes().to_int()
  let payload = FixedArray::make(payload_bytes, b'\x00')
  for index in 0..<payload.length() {
    payload[index] = ((index * 29 + 7) % 251).to_byte()
  }
  let destinations = [
    FixedArray::make(plans[0].arena_bytes().to_int(), b'\x00'),
    FixedArray::make(plans[1].arena_bytes().to_int(), b'\x00'),
  ]
  for rank in 0..<2 {
    for reference in 0..<plans[rank].tensor_count() {
      let tensor_reference = try! @plan.TensorRef::new(reference)
      let placement = plans[rank].tensor(tensor_reference).unwrap()
      tp_apply_recipe(payload, destinations[rank], placement.transfer_recipe())
    }
  }
  for reference in 0..<bindings.length() {
    let tensor_reference = try! @plan.TensorRef::new(reference)
    let source = bindings.binding(tensor_reference).unwrap().tensor()
    let first = plans[0].tensor(tensor_reference).unwrap()
    let full_rows = first.extent().full_rows()
    let full_columns = first.extent().full_columns()
    let reconstructed = FixedArray::make(source.byte_length().to_int(), b'\x00')
    for rank in 0..<2 {
      let placement = plans[rank].tensor(tensor_reference).unwrap()
      let extent = placement.extent()
      let region = placement.region()
      if placement.placement() == Replicated && rank == 1 {
        continue
      }
      for local_row in 0L..<extent.row_count() {
        for local_column in 0L..<extent.column_count() {
          let global_row = extent.row_start() + local_row
          let global_column = extent.column_start() + local_column
          let global_element = global_row * full_columns + global_column
          let local_element = local_row * extent.column_count() + local_column
          for byte in 0L..<2L {
            reconstructed[(global_element * 2L + byte).to_int()] = destinations[rank][(region.offset_bytes() +
              local_element * 2L +
              byte).to_int()]
          }
        }
      }
    }
    assert_eq(full_rows * full_columns * 2L, source.byte_length())
    for byte in 0..<reconstructed.length() {
      assert_eq(
        reconstructed[byte],
        payload[source.data_start().to_int() + byte],
      )
    }
  }
}
