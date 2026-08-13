///|
fn ValidatedPlanFrame::read_i32_at(
  self : ValidatedPlanFrame,
  offset : Int,
) -> Int raise WorkerWireError {
  let owner = self.owner()
  read_i32(owner.storage, offset)
}

///|
fn ValidatedPlanFrame::read_u64_at(
  self : ValidatedPlanFrame,
  offset : Int,
) -> UInt64 raise WorkerWireError {
  let owner = self.owner()
  read_u64(owner.storage, offset)
}

///|
pub fn ValidatedPlanFrame::sequence_value(
  self : ValidatedPlanFrame,
) -> UInt64 raise WorkerWireError {
  self.read_u64_at(16)
}

///|
pub fn ValidatedPlanFrame::model_generation_value(
  self : ValidatedPlanFrame,
) -> UInt64 raise WorkerWireError {
  self.read_u64_at(24)
}

///|
pub fn ValidatedPlanFrame::token_budget(
  self : ValidatedPlanFrame,
) -> Int raise WorkerWireError {
  self.read_i32_at(32)
}

///|
pub fn ValidatedPlanFrame::token_count(
  self : ValidatedPlanFrame,
) -> Int raise WorkerWireError {
  self.read_i32_at(36)
}

///|
pub fn ValidatedPlanFrame::page_count(
  self : ValidatedPlanFrame,
) -> Int raise WorkerWireError {
  self.read_i32_at(40)
}

///|
pub fn ValidatedPlanFrame::capability_count(
  self : ValidatedPlanFrame,
) -> Int raise WorkerWireError {
  self.read_i32_at(44)
}

///|
pub fn ValidatedPlanFrame::prefill_count(
  self : ValidatedPlanFrame,
) -> Int raise WorkerWireError {
  self.read_i32_at(48)
}

///|
pub fn ValidatedPlanFrame::decode_count(
  self : ValidatedPlanFrame,
) -> Int raise WorkerWireError {
  self.read_i32_at(52)
}

///|
pub fn ValidatedPlanFrame::token_at(
  self : ValidatedPlanFrame,
  index : Int,
) -> Int raise WorkerWireError {
  let count = self.token_count()
  if index < 0 || index >= count {
    raise InvalidFrame(TokenTable)
  }
  self.read_i32_at(token_table_offset() + index * 4)
}

///|
pub fn ValidatedPlanFrame::page_index_at(
  self : ValidatedPlanFrame,
  index : Int,
) -> Int raise WorkerWireError {
  let count = self.page_count()
  if index < 0 || index >= count {
    raise InvalidFrame(PageTable)
  }
  self.read_i32_at(page_table_offset(self.token_count()) + index * 8)
}

///|
pub fn ValidatedPlanFrame::page_generation_at(
  self : ValidatedPlanFrame,
  index : Int,
) -> Int raise WorkerWireError {
  let count = self.page_count()
  if index < 0 || index >= count {
    raise InvalidFrame(PageTable)
  }
  self.read_i32_at(page_table_offset(self.token_count()) + index * 8 + 4)
}

///|
pub fn ValidatedPlanFrame::capability_value_at(
  self : ValidatedPlanFrame,
  index : Int,
) -> Int raise WorkerWireError {
  let count = self.capability_count()
  if index < 0 || index >= count {
    raise InvalidFrame(CapabilityTable)
  }
  self.read_i32_at(
    capability_table_offset(self.token_count(), self.page_count()) + index * 4,
  )
}

///|
pub fn ValidatedPlanFrame::prefill_row(
  self : ValidatedPlanFrame,
  index : Int,
) -> PlanFramePrefillRow raise WorkerWireError {
  let owner = self.owner()
  if index < 0 || index >= read_i32(owner.storage, 48) {
    raise InvalidFrame(PrefillRow)
  }
  PlanFramePrefillRow(owner, self.1, index)
}

///|
pub fn ValidatedPlanFrame::decode_row(
  self : ValidatedPlanFrame,
  index : Int,
) -> PlanFrameDecodeRow raise WorkerWireError {
  let owner = self.owner()
  if index < 0 || index >= read_i32(owner.storage, 52) {
    raise InvalidFrame(DecodeRow)
  }
  PlanFrameDecodeRow(owner, self.1, index)
}

///|
fn PlanFramePrefillRow::offset(
  self : PlanFramePrefillRow,
) -> Int raise WorkerWireError {
  ignore(ValidatedPlanFrame(self.0, self.1).owner())
  prefill_table_offset(
    read_i32(self.0.storage, 36),
    read_i32(self.0.storage, 40),
    read_i32(self.0.storage, 44),
  ) +
  self.2 * prefill_row_bytes
}

///|
fn PlanFrameDecodeRow::offset(
  self : PlanFrameDecodeRow,
) -> Int raise WorkerWireError {
  ignore(ValidatedPlanFrame(self.0, self.1).owner())
  decode_table_offset(
    read_i32(self.0.storage, 36),
    read_i32(self.0.storage, 40),
    read_i32(self.0.storage, 44),
    read_i32(self.0.storage, 48),
  ) +
  self.2 * decode_row_bytes
}

///|
pub fn PlanFramePrefillRow::request_id_value(
  self : PlanFramePrefillRow,
) -> UInt64 raise WorkerWireError {
  read_u64(self.0.storage, self.offset())
}

///|
pub fn PlanFramePrefillRow::request_generation_value(
  self : PlanFramePrefillRow,
) -> UInt64 raise WorkerWireError {
  read_u64(self.0.storage, self.offset() + 8)
}

///|
pub fn PlanFramePrefillRow::sequence_token_start(
  self : PlanFramePrefillRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 16)
}

///|
pub fn PlanFramePrefillRow::token_start(
  self : PlanFramePrefillRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 20)
}

///|
pub fn PlanFramePrefillRow::token_length(
  self : PlanFramePrefillRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 24)
}

///|
pub fn PlanFramePrefillRow::page_start(
  self : PlanFramePrefillRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 28)
}

///|
pub fn PlanFramePrefillRow::page_length(
  self : PlanFramePrefillRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 32)
}

///|
pub fn PlanFramePrefillRow::capability_start(
  self : PlanFramePrefillRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 36)
}

///|
pub fn PlanFramePrefillRow::capability_length(
  self : PlanFramePrefillRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 40)
}

///|
pub fn PlanFramePrefillRow::produces_token(
  self : PlanFramePrefillRow,
) -> Bool raise WorkerWireError {
  read_u32(self.0.storage, self.offset() + 44) == 1
}

///|
pub fn PlanFramePrefillRow::sampling_mode_value(
  self : PlanFramePrefillRow,
) -> Int raise WorkerWireError {
  read_u32(self.0.storage, self.offset() + 48).reinterpret_as_int()
}

///|
pub fn PlanFramePrefillRow::sampling_seed_value(
  self : PlanFramePrefillRow,
) -> UInt64 raise WorkerWireError {
  read_u64(self.0.storage, self.offset() + 80)
}

///|
pub fn PlanFramePrefillRow::sample_index(
  self : PlanFramePrefillRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 88)
}

///|
pub fn PlanFramePrefillRow::completion_slot(
  self : PlanFramePrefillRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 92)
}

///|
pub fn PlanFrameDecodeRow::request_id_value(
  self : PlanFrameDecodeRow,
) -> UInt64 raise WorkerWireError {
  read_u64(self.0.storage, self.offset())
}

///|
pub fn PlanFrameDecodeRow::request_generation_value(
  self : PlanFrameDecodeRow,
) -> UInt64 raise WorkerWireError {
  read_u64(self.0.storage, self.offset() + 8)
}

///|
pub fn PlanFrameDecodeRow::sequence_token_position(
  self : PlanFrameDecodeRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 16)
}

///|
pub fn PlanFrameDecodeRow::token_start(
  self : PlanFrameDecodeRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 20)
}

///|
pub fn PlanFrameDecodeRow::page_start(
  self : PlanFrameDecodeRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 24)
}

///|
pub fn PlanFrameDecodeRow::page_length(
  self : PlanFrameDecodeRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 28)
}

///|
pub fn PlanFrameDecodeRow::capability_start(
  self : PlanFrameDecodeRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 32)
}

///|
pub fn PlanFrameDecodeRow::capability_length(
  self : PlanFrameDecodeRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 36)
}

///|
pub fn PlanFrameDecodeRow::sampling_mode_value(
  self : PlanFrameDecodeRow,
) -> Int raise WorkerWireError {
  read_u32(self.0.storage, self.offset() + 40).reinterpret_as_int()
}

///|
pub fn PlanFrameDecodeRow::sampling_seed_value(
  self : PlanFrameDecodeRow,
) -> UInt64 raise WorkerWireError {
  read_u64(self.0.storage, self.offset() + 72)
}

///|
pub fn PlanFrameDecodeRow::sample_index(
  self : PlanFrameDecodeRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 80)
}

///|
pub fn PlanFrameDecodeRow::completion_slot(
  self : PlanFrameDecodeRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 84)
}
