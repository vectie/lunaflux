///|
fn BatchedDecoderFrame::find_live(self : Self, id : UInt64) -> Int {
  for slot in 0..<self.slots {
    if self.live[slot] && self.ids[slot] == id {
      return slot
    }
  }
  -1
}

///|
fn BatchedDecoderFrame::reserve_entry(
  self : Self,
  entry : Int,
  id : UInt64,
  generation : UInt64,
  position : Int,
  start : Int,
  rows : Int,
  prefill : Bool,
  output : Bool,
) -> Unit raise BatchFrameError {
  guard rows > 0 &&
    position >= 0 &&
    rows <= self.context &&
    position <= self.context - rows &&
    start == self.active_rows &&
    rows <= self.rows - start else {
    raise InvalidPosition
  }
  let old = self.find_live(id)
  let slot = if old >= 0 {
    guard self.generations[old] == generation else { raise GenerationMismatch }
    guard self.processed[old] == position else { raise InvalidPosition }
    old
  } else {
    guard prefill && position == 0 else { raise UnknownRequest }
    let available = for slot in 0..<self.slots {
      if self.live[slot] {
        continue
      }
      let taken = for previous in 0..<entry {
        if self.pending_slots[previous] == slot {
          break true
        }
      } nobreak {
        false
      }
      if !taken {
        break slot
      }
    } nobreak {
      -1
    }
    guard available >= 0 else { raise CapacityExceeded }
    available
  }
  self.pending_slots[entry] = slot
  self.pending_ids[entry] = id
  self.pending_generations[entry] = generation
  self.pending_ends[entry] = position + rows
  self.entry_outputs[entry] = if output { self.outputs } else { -1 }
  if output {
    self.output_entries[self.outputs] = entry
    store_i32(self.inputs[7], self.outputs, start + rows - 1)
    self.outputs += 1
  }
  store_i32(self.inputs[2], entry, start)
  store_i32(self.inputs[2], entry + 1, start + rows)
  store_i32(self.inputs[3], entry, slot)
  let reset = if old < 0 { 1 } else { 0 }
  store_i32(self.inputs[4], entry, reset)
  store_i32(self.inputs[5], entry, reset)
  store_i32(self.inputs[10], entry, position)
  for row in 0..<rows {
    store_i32(self.inputs[8], start + row, position + row)
    store_i32(self.inputs[9], start + row, slot)
    store_i32(self.inputs[11], start + row, entry)
  }
  self.active_rows += rows
}

///|
/// The validated wire already forbids duplicate identities and completion
/// slots. All tentative changes remain scratch until completion; a rejected
/// plan cannot consume a request slot or advance any retained history.
pub fn BatchedDecoderFrame::stage(
  self : Self,
  frame : @wire.ValidatedPlanFrame,
  writer : @wire.CompletionFrameWriter,
) -> Unit raise {
  self.require_idle()
  writer.validate_plan(frame)
  guard frame.model_generation_value() == self.generation else {
    raise BatchFrameError::ModelGenerationMismatch
  }
  let prefills = frame.prefill_count()
  let decodes = frame.decode_count()
  guard prefills + decodes > 0 &&
    prefills + decodes <= self.sequences &&
    frame.token_count() <= self.rows else {
    raise BatchFrameError::CapacityExceeded
  }
  for token in 0..<frame.token_count() {
    guard frame.token_at(token) >= 0 && frame.token_at(token) < self.vocabulary else {
      raise BatchFrameError::InvalidToken
    }
  }
  self.active_rows = 0
  self.outputs = 0
  self.active_pages = 0
  for entry in 0..<prefills {
    let row = frame.prefill_row(entry)
    self.reserve_entry(
      entry,
      row.request_id_value(),
      row.request_generation_value(),
      row.sequence_token_start(),
      row.token_start(),
      row.token_length(),
      true,
      row.produces_token(),
    )
    self.stage_pages(
      frame,
      entry,
      row.page_start(),
      row.page_length(),
      row.sequence_token_start() + row.token_length(),
    )
  }
  for index in 0..<decodes {
    let row = frame.decode_row(index)
    self.reserve_entry(
      prefills + index,
      row.request_id_value(),
      row.request_generation_value(),
      row.sequence_token_position(),
      row.token_start(),
      1,
      false,
      true,
    )
    self.stage_pages(
      frame,
      prefills + index,
      row.page_start(),
      row.page_length(),
      row.sequence_token_position() + 1,
    )
  }
  guard self.active_rows == frame.token_count() else {
    raise BatchFrameError::InvalidPosition
  }
  for token in 0..<self.active_rows {
    store_i32(self.inputs[1], token, frame.token_at(token))
  }
  store_i32(self.inputs[0], 0, prefills)
  store_i32(self.inputs[0], 1, decodes)
  store_i32(self.inputs[0], 2, prefills + decodes)
  store_i32(self.inputs[0], 3, self.active_rows)
  store_i32(self.inputs[0], 4, 0)
  store_i32(self.inputs[6], 0, self.outputs)
  self.completion.bind(frame, writer)
  self.active_sequences = prefills + decodes
  self.state = Staged
}

///|
/// Copies the scheduler-owned logical page list, including its reuse
/// generation. No private per-request KV reservation or payload access occurs.
/// A later device history owner uses this exact list for all sparse operands.
fn BatchedDecoderFrame::stage_pages(
  self : Self,
  frame : @wire.ValidatedPlanFrame,
  entry : Int,
  start : Int,
  length : Int,
  tokens : Int,
) -> Unit raise {
  store_i32(self.inputs[12], entry, self.active_pages)
  if self.history is Some(domain) {
    let (_, physical_pages, context, _) = domain.geometry()
    let required = domain.pages_for(tokens).unwrap()
    guard length >= required &&
      length <= domain.pages_for(context).unwrap() &&
      length <= domain.table_capacity() - self.active_pages else {
      raise BatchFrameError::InvalidPages
    }
    for index in 0..<length {
      let page = frame.page_index_at(start + index)
      guard page >= 0 && page < physical_pages else {
        raise BatchFrameError::InvalidPages
      }
      store_i32(self.inputs[13], self.active_pages + index, page)
      store_i32(
        self.inputs[14],
        self.active_pages + index,
        frame.page_generation_at(start + index),
      )
    }
    self.active_pages += length
  }
  store_i32(self.inputs[12], entry + 1, self.active_pages)
}
