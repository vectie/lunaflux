///|
/// The protocol orders prefill CSR rows before single-token decode rows.
/// Narrowing counts therefore preserves every original token/page/output
/// index; no gather, renumbering, or KV mutation is involved.
#valtype
priv struct PagedAttentionPhaseRows {
  prefill_rows : Int
  prefill_tokens : Int
}

///|
fn attention_phase_rows(
  prefill_rows : Int,
  decode_rows : Int,
  token_count : Int,
) -> PagedAttentionPhaseRows {
  { prefill_rows, prefill_tokens: token_count - decode_rows, }
}

///|
fn publish_prefill_attention_counts(
  host : DeviceStepHostBuffers,
  summary : DeviceStepSummary,
) -> Unit {
  let phase = attention_phase_rows(
    summary.prefill_count,
    summary.decode_count,
    summary.token_count,
  )
  let start = PREFILL_ATTENTION_COUNTS_CELL
  write_i32(host.counts, start, phase.prefill_rows)
  write_i32(host.counts, start + 1, 0)
  write_i32(host.counts, start + 2, phase.prefill_rows)
  write_i32(host.counts, start + 3, phase.prefill_tokens)
  write_i32(host.counts, start + 4, summary.page_count)
}

///|
/// This fixed-prefix view precedes the variable-length output row-end suffix.
/// Every descriptor upload includes it, including zero-output frames; placing
/// it at the allocation tail would leave stale device counts on partial copies.
/// The argument remains fixed throughout graph capture and replay.
fn prefill_attention_counts_argument(
  descriptor : DeviceStepResources,
) -> @device.KernelArgument raise DeviceStepError {
  @device.KernelArgument::device_region(
    descriptor.counts,
    offset=byte_capacity(PREFILL_ATTENTION_COUNTS_CELL),
    byte_count=20L,
    alignment=4L,
  ) catch {
    error => raise ExecutorDevice(KernelArguments, error)
  }
}
