///|
fn checked_workspace_cells(
  total : Int,
  addition : Int,
) -> Int raise IncrementalOutputError {
  if addition < 0 || total > 0x7fffffff - addition {
    raise invalid(LunaIncrementalOutputWorkspace, Unsupported)
  }
  total + addition
}

///|
fn checked_workspace_product(
  left : Int,
  right : Int,
) -> Int raise IncrementalOutputError {
  if left < 0 || right < 0 || (right > 0 && left > 0x7fffffff / right) {
    raise invalid(LunaIncrementalOutputWorkspace, Unsupported)
  }
  left * right
}

///|
fn preflight_workspace_shape(
  inference : @inference.InferenceLimits,
  patterns : Int,
  bytes : Int,
) -> Int raise IncrementalOutputError {
  if patterns < 0 ||
    bytes < 0 ||
    patterns > inference.max_stop_strings() ||
    bytes > inference.max_stop_string_bytes() ||
    (patterns > 0 && bytes == 0) {
    raise invalid(LunaIncrementalOutputWorkspace, Unsupported)
  }
  let pattern_cells = checked_workspace_product(patterns, bytes)
  let pending = checked_workspace_cells(bytes, 3)
  let matcher_cells = checked_workspace_product(patterns, 3)
  let int_cells = checked_workspace_cells(pattern_cells, matcher_cells)
  let byte_cells = checked_workspace_cells(pattern_cells, pending)
    |> checked_workspace_cells(inference.max_decoded_delta_bytes())
  ignore(checked_workspace_cells(int_cells, byte_cells))
  pattern_cells
}

///|
/// Returns the exact number of `Int` backing cells allocated by a workspace.
pub fn LunaIncrementalOutputWorkspace::required_int_cells(
  inference : @inference.InferenceLimits,
) -> Int raise IncrementalOutputError {
  let patterns = inference.max_stop_strings()
  let pattern_cells = preflight_workspace_shape(
    inference,
    patterns,
    inference.max_stop_string_bytes(),
  )
  checked_workspace_cells(pattern_cells, checked_workspace_product(patterns, 3))
}

///|
/// Returns the exact number of `Byte` backing cells allocated by a workspace.
pub fn LunaIncrementalOutputWorkspace::required_byte_cells(
  inference : @inference.InferenceLimits,
) -> Int raise IncrementalOutputError {
  let bytes = inference.max_stop_string_bytes()
  let pattern_cells = preflight_workspace_shape(
    inference,
    inference.max_stop_strings(),
    bytes,
  )
  checked_workspace_cells(pattern_cells, checked_workspace_cells(bytes, 3))
  |> checked_workspace_cells(inference.max_decoded_delta_bytes())
}

///|
/// Returns the exact number of retained semantic-view capability slots.
pub fn LunaIncrementalOutputWorkspace::required_reference_cells(
  inference : @inference.InferenceLimits,
) -> Int raise IncrementalOutputError {
  ignore(
    preflight_workspace_shape(
      inference,
      inference.max_stop_strings(),
      inference.max_stop_string_bytes(),
    ),
  )
  1
}

///|
fn new_luna_incremental_output_workspace(
  tokenizer : @tokenizer.TokenizerSpec,
  inference : @inference.InferenceLimits,
  budget : LunaIncrementalOutputStepBudget,
  pattern_capacity : Int,
  stop_string_byte_capacity : Int,
) -> LunaIncrementalOutputWorkspace raise IncrementalOutputError {
  let pattern_cells = preflight_workspace_shape(
    inference, pattern_capacity, stop_string_byte_capacity,
  )
  let delta_bytes = inference.max_decoded_delta_bytes()
  if tokenizer.limits().max_token_bytes > delta_bytes - 3 {
    raise invalid(DecodeEnvelope, Unsupported)
  }
  {
    tokenizer,
    inference,
    budget,
    stop_string_byte_capacity,
    pattern_bytes: FixedArray::make(pattern_cells, b'\x00'),
    failure: FixedArray::make(pattern_cells, 0),
    pattern_lengths: FixedArray::make(pattern_capacity, 0),
    states: FixedArray::make(pattern_capacity, 0),
    next_states: FixedArray::make(pattern_capacity, 0),
    pending: FixedArray::make(stop_string_byte_capacity + 3, b'\x00'),
    scratch: FixedArray::make(delta_bytes, b'\x00'),
    setup_stops: None,
    setup_semantics: Array::new(capacity=1),
    special_policy: Preserve,
    phase: LUNA_INCREMENTAL_OUTPUT_IDLE,
    epoch: 0UL,
    stop_count: 0,
    setup_pattern: 0,
    setup_unit_offset: 0,
    setup_emit_index: 0,
    setup_emit_length: 0,
    setup_codepoint: 0,
    setup_failure_index: 0,
    setup_failure_matched: 0,
    longest_pattern: 0,
    last_work_units: 0,
    setup_error_rule: Lifecycle,
    setup_error_issue: Invalid,
    pending_count: 0,
    utf8_remaining: 0,
    utf8_codepoint: 0,
    utf8_minimum: 0,
    utf8_seen: 0,
    lifecycle: OUTPUT_ACTIVE,
  }
}

///|
/// Allocates every envelope-sized pattern, matcher, carry, and decode cell once.
pub fn LunaIncrementalOutputWorkspace::new(
  tokenizer : @tokenizer.TokenizerSpec,
  inference : @inference.InferenceLimits,
  budget : LunaIncrementalOutputStepBudget,
) -> LunaIncrementalOutputWorkspace raise IncrementalOutputError {
  new_luna_incremental_output_workspace(
    tokenizer,
    inference,
    budget,
    inference.max_stop_strings(),
    inference.max_stop_string_bytes(),
  )
}

///|
fn exact_max_stop_bytes(stops : @inference.StopConditions) -> Int {
  let values = stops.strings()
  let mut maximum = 0
  for value in values {
    let mut offset = 0
    let mut length = 0
    while offset < value.length() {
      let first = value.unsafe_get(offset).to_int()
      let codepoint = if first >= 0xd800 && first <= 0xdbff {
        let second = value.unsafe_get(offset + 1).to_int()
        offset += 2
        0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00)
      } else {
        offset += 1
        first
      }
      length += utf8_width(codepoint)
    }
    if length > maximum {
      maximum = length
    }
  }
  maximum
}
