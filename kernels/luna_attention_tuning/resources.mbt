///|
fn resource_integer(text : StringView) -> Int raise AttentionTuningError {
  let value = positive(text)
  guard value <= 2147483647L else { raise InvalidTable }
  value.to_int()
}

///|
/// Decode the budget before enumerating the resource-aware frontier. This
/// preparatory pass does not accept register observations or select a kernel.
pub fn parse_attention_resource_budget(
  bytes : Bytes,
  scope~ : String,
) -> @compiler.AttentionResourceBudget raise AttentionTuningError {
  guard bytes.length() <= 1048576 else { raise InvalidTable }
  let text = @utf8.decode(bytes) catch { _ => raise InvalidTable }
  guard text.split("\n").to_array()
    is ["luna-attention-resources-v1", actual_scope, budget_line, .., ""] &&
    actual_scope == "scope\t" + scope else {
    raise InvalidTable
  }
  guard budget_line.split("\t").to_array()
    is [
      "budget",
      units,
      threads,
      registers,
      grain,
      allocation_threads,
      shared,
      blocks,
    ] else {
    raise InvalidTable
  }
  @compiler.AttentionResourceBudget::new(
    units=resource_integer(units),
    threads_per_unit=resource_integer(threads),
    registers_per_unit=resource_integer(registers),
    register_allocation_grain=resource_integer(grain),
    register_allocation_threads=resource_integer(allocation_threads),
    shared_bytes_per_unit=positive(shared),
    blocks_per_unit=resource_integer(blocks),
  ) catch {
    _ => raise InvalidTable
  }
}

///|
/// Read offline resource measurements for one base frontier, device and
/// toolchain scope. Schedules include the budget-expanded compiler frontier;
/// observations never determine their own valid schedule set. No I/O occurs.
pub fn parse_attention_resource_feedback(
  bytes : Bytes,
  scope~ : String,
  schedules : ArrayView[@schedule.AttentionTileScheduleDigest],
) -> (
  @compiler.AttentionResourceBudget,
  ReadOnlyArray[@compiler.AttentionRegisterObservation],
) raise AttentionTuningError {
  let budget = parse_attention_resource_budget(bytes, scope~)
  let text = @utf8.decode(bytes) catch { _ => raise InvalidTable }
  guard text.split("\n").to_array() is [_, _, _, .. lines, ""] else {
    raise InvalidTable
  }
  let observations : Array[@compiler.AttentionRegisterObservation] = []
  let seen : Array[@schedule.AttentionTileScheduleDigest] = []
  guard !lines.is_empty() && lines.length() <= 128 else { raise InvalidTable }
  for line in lines {
    guard line.split("\t").to_array() is ["registers", digest, count] else {
      raise InvalidTable
    }
    guard schedules.filter(s => s.as_hex() == digest.to_owned()) is [schedule] &&
      !seen.contains(schedule) else {
      raise InvalidTable
    }
    observations.push(
      @compiler.AttentionRegisterObservation::new(
        schedule,
        registers_per_thread=resource_integer(count),
      ) catch {
        _ => raise InvalidTable
      },
    )
    seen.push(schedule)
  }
  (budget, ReadOnlyArray::from_array(observations))
}
