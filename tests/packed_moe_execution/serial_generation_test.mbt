///|
test "serial greedy continuation reuses frame and token storage without warm heap allocation" {
  let (_, initial, writer) = serial_wire_plan(7UL, tokens=[2])
  writer.append_final_prefill_completed(initial.prefill_row(0), 7)
  let completed = writer.submit()
  let generation = @serial_generation.SerialGreedyGeneration::new(
    initial,
    completed,
    serial_limits(),
    serial_inference_limits(),
    maximum_new_tokens=15,
    maximum_context=16,
    stop_tokens=[],
  )
  let completions = @wire.CompletionFrameBuffer::new(
    serial_limits(),
    serial_inference_limits(),
  )
  begin_allocations()
  for index in 0..<14 {
    let plan = generation.next_decode()
    let writer = completions.writer_for(plan)
    writer.append_decode_completed(plan.decode_row(0), index % 8)
    generation.accept(writer.submit())
  }
  let allocated = end_allocations()
  assert_eq(allocated, 0UL)
  assert_eq(generation.token_count(), 15)
  assert_eq(
    generation.finish_reason(),
    @serial_generation.GenerationFinish::Length,
  )
}
