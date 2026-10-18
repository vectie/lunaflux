///|
fn paged_host(
  physical_pages? : Int = 8,
) -> @batched_decoder_frame.BatchedDecoderFrame raise {
  let domain = @history.PagedHistoryDomain::new(
    rows_per_page=2,
    physical_pages~,
    maximum_context=8,
    maximum_sequences=3,
  ).unwrap()
  @batched_decoder_frame.BatchedDecoderFrame::new(
    maximum_rows=8,
    maximum_sequences=3,
    cache_slots=3,
    maximum_context=8,
    vocabulary_size=16,
    model_generation=9UL,
    paged_history=domain,
  )
}

///|
test "page lists and row-to-sequence ownership survive mixed reordered batches" {
  let host = paged_host()
  let (_, frame, writer) = plan(
    [
      (10UL, 0UL, 0, [1, 2, 3], false, true, false),
      (20UL, 0UL, 0, [4], false, true, false),
    ],
    page_rows=2,
  )
  host.stage(frame, writer)
  assert_eq(port(host, PageOffsets), [0, 2, 3])
  assert_eq(port(host, PageIds), Array::makei(3, i => frame.page_index_at(i)))
  assert_eq(
    port(host, PageGenerations),
    Array::makei(3, i => frame.page_generation_at(i)),
  )
  assert_eq(port(host, RowSequences), [0, 0, 0, 1])
  ignore(host.complete([2, 3], execution_failed=false))
  let (_, frame, writer) = plan(
    [
      (20UL, 0UL, 1, [3, 4, 5], false, true, false),
      (10UL, 0UL, 3, [2], true, false, false),
    ],
    page_rows=2,
  )
  host.stage(frame, writer)
  assert_eq(port(host, SequenceSlots), [1, 0])
  assert_eq(port(host, RowSequences), [0, 0, 0, 1])
  assert_eq(port(host, PageOffsets), [0, 2, 4])
  assert_eq(port(host, PageIds), Array::makei(4, i => frame.page_index_at(i)))
  assert_eq(
    port(host, PageGenerations),
    Array::makei(4, i => frame.page_generation_at(i)),
  )
  ignore(host.complete([6, 7], execution_failed=false))
  host.close()
}

///|
test "short page coverage and physical pool overflow reject before history commit" {
  for shortage in [true, false] {
    let host = paged_host(physical_pages=if shortage { 8 } else { 1 })
    let (_, frame, writer) = plan(
      [(10UL, 0UL, 0, [1, 2, 3], false, true, false)],
      page_rows=if shortage { 8 } else { 2 },
    )
    try host.stage(frame, writer) catch {
      @batched_decoder_frame.BatchFrameError::InvalidPages => ()
      e => raise e
    } noraise {
      _ => fail("invalid page table accepted")
    }
    writer.abort()
    let (_, frame, writer) = plan(
      [(10UL, 1UL, 0, [1], false, true, false)],
      page_rows=2,
    )
    host.stage(frame, writer)
    assert_eq(port(host, SequenceSlots), [0])
    assert_eq(port(host, SparseResets), [1])
    assert_eq(port(host, PageOffsets), [0, 1])
    ignore(host.complete([2], execution_failed=false))
    host.close()
  }
}

///|
test "page DMA never copies stale capacity and counts follow reused requests" {
  let host = paged_host()
  let (_, frame, writer) = plan(
    [(10UL, 0UL, 0, [1, 2, 3, 4, 5], false, true, false)],
    page_rows=2,
  )
  host.stage(frame, writer)
  ignore(host.complete([6], execution_failed=false))
  host.release_request(10UL, 0UL)
  let (_, frame, writer) = plan(
    [(10UL, 1UL, 0, [1], false, true, false)],
    page_rows=2,
  )
  host.stage(frame, writer)
  let bytes = FixedArray::make(host.input_capacity(PageIds), b'\xff')
  assert_eq(host.copy_input(PageIds, bytes), 4)
  for i in 4..<bytes.length() {
    assert_eq(bytes[i], b'\xff')
  }
  assert_eq(port(host, PageOffsets), [0, 1])
  ignore(host.complete([2], execution_failed=false))
  host.close()
}
