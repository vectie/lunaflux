///|
const OUTPUT_COUNTS_CELL : Int = 5

///|
const PURE_SUFFIX_COUNTS_CELL : Int = 10

///|
const OUTPUT_ROW_ENDS_CELL : Int = 15

///|
fn output_descriptor_cells(limits : DeviceStepLimits) -> Int {
  OUTPUT_ROW_ENDS_CELL + row_offset_cells(limits) + 5
}

///|
fn write_output_row(
  host : DeviceStepHostBuffers,
  row : Int,
  token_row : Int,
) -> Unit {
  if token_row < 0 {
    host.output_indices[row] = -1
  } else {
    host.output_indices[row] = host.output_rows
    host.output_rows += 1
    write_i32(
      host.counts,
      OUTPUT_ROW_ENDS_CELL + host.output_rows,
      token_row + 1,
    )
  }
}

///|
/// Keep the original matrix-versus-single-row arithmetic partition. Padding is
/// private terminal-view storage, never a scheduler or KV request row.
fn output_physical_rows(
  host : DeviceStepHostBuffers,
  summary : DeviceStepSummary,
) -> Int {
  if host.output_rows == 1 && summary.row_count > 1 {
    2
  } else {
    host.output_rows
  }
}

///|
fn publish_output_counts(
  host : DeviceStepHostBuffers,
  summary : DeviceStepSummary,
) -> Unit {
  publish_prefill_attention_counts(host, summary)
  let physical = output_physical_rows(host, summary)
  write_i32(host.counts, OUTPUT_COUNTS_CELL, physical - summary.decode_count)
  write_i32(host.counts, OUTPUT_COUNTS_CELL + 1, summary.decode_count)
  write_i32(host.counts, OUTPUT_COUNTS_CELL + 2, physical)
  write_i32(host.counts, OUTPUT_COUNTS_CELL + 3, summary.token_count)
  write_i32(host.counts, OUTPUT_COUNTS_CELL + 4, summary.page_count)
  // A stable captured argument view, not another graph or descriptor upload.
  // Mixed demand retains all rows: only the semantically dead suffix of an
  // entirely output-free frame may observe zero work.
  for cell in 0..<5 {
    write_i32(
      host.counts,
      PURE_SUFFIX_COUNTS_CELL + cell,
      if host.output_rows == 0 {
        0
      } else {
        read_descriptor_i32(host.counts, cell)
      },
    )
  }
  if physical > host.output_rows {
    write_i32(host.counts, OUTPUT_ROW_ENDS_CELL + physical, 0)
  }
}

///|
fn projected_result_row(descriptor : DeviceStepOwner, row : Int) -> Int {
  if descriptor.host.project_output_rows {
    descriptor.host.output_indices[row]
  } else {
    row
  }
}

///|
fn projected_result_count(descriptor : DeviceStepOwner, rows : Int) -> Int {
  if descriptor.host.project_output_rows {
    descriptor.host.output_rows
  } else {
    rows
  }
}

///|
fn sampling_descriptor_is_projected(resources : PagedGraphResources) -> Bool {
  match resources.descriptor {
    Some(Complete(owner)) => owner.host.project_output_rows
    _ => false
  }
}

///|
fn projected_operand_offset(binding : PagedKvOperandBinding) -> Int64 {
  match binding.role() {
    RuntimeInput(StepCounts) => byte_capacity(OUTPUT_COUNTS_CELL)
    RuntimeInput(QueryRowOffsets) => byte_capacity(OUTPUT_ROW_ENDS_CELL)
    _ => -1L
  }
}

///|
fn pure_suffix_counts_offset(operation : Int, effect_prefix : Int) -> Int64 {
  if effect_prefix > 0 && operation >= effect_prefix {
    byte_capacity(PURE_SUFFIX_COUNTS_CELL)
  } else {
    -1L
  }
}

///|
/// The matrix BF16 head has explicit gathered token-row output and accepts
/// zero-padded rows. Other numeric/ABI owners retain their original view.
fn paged_terminal_rows_are_projectable(
  blueprint : PagedKvExecutionBlueprint,
) -> Bool {
  let steps = blueprint.steps()
  guard steps.last() is Some(head) else { return false }
  head.operation_kind() == LanguageModelHead &&
  head.dimensions().shared_memory_bytes() > 0 &&
  head
  .operands()
  .any(binding => binding.role() == RuntimeInput(QueryRowOffsets)) &&
  head.operands().any(binding => binding.role() == RuntimeInput(StepCounts))
}
