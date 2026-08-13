///|
fn read_wire_has_top_k(bytes : FixedArray[Byte], offset : Int) -> Bool {
  read_u32(bytes, offset + 12) == 1
}

///|
fn read_wire_has_top_p(bytes : FixedArray[Byte], offset : Int) -> Bool {
  read_u32(bytes, offset + 20) == 1
}

///|
pub fn PlanFramePrefillRow::sampling_temperature(
  self : PlanFramePrefillRow,
) -> Double raise WorkerWireError {
  read_u64(self.0.storage, self.offset() + 52).reinterpret_as_double()
}

///|
pub fn PlanFramePrefillRow::sampling_has_top_k(
  self : PlanFramePrefillRow,
) -> Bool raise WorkerWireError {
  read_wire_has_top_k(self.0.storage, self.offset() + 48)
}

///|
pub fn PlanFramePrefillRow::sampling_top_k_value(
  self : PlanFramePrefillRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 64)
}

///|
pub fn PlanFramePrefillRow::sampling_has_top_p(
  self : PlanFramePrefillRow,
) -> Bool raise WorkerWireError {
  read_wire_has_top_p(self.0.storage, self.offset() + 48)
}

///|
pub fn PlanFramePrefillRow::sampling_top_p_value(
  self : PlanFramePrefillRow,
) -> Double raise WorkerWireError {
  read_u64(self.0.storage, self.offset() + 72).reinterpret_as_double()
}

///|
pub fn PlanFrameDecodeRow::sampling_temperature(
  self : PlanFrameDecodeRow,
) -> Double raise WorkerWireError {
  read_u64(self.0.storage, self.offset() + 44).reinterpret_as_double()
}

///|
pub fn PlanFrameDecodeRow::sampling_has_top_k(
  self : PlanFrameDecodeRow,
) -> Bool raise WorkerWireError {
  read_wire_has_top_k(self.0.storage, self.offset() + 40)
}

///|
pub fn PlanFrameDecodeRow::sampling_top_k_value(
  self : PlanFrameDecodeRow,
) -> Int raise WorkerWireError {
  read_i32(self.0.storage, self.offset() + 56)
}

///|
pub fn PlanFrameDecodeRow::sampling_has_top_p(
  self : PlanFrameDecodeRow,
) -> Bool raise WorkerWireError {
  read_wire_has_top_p(self.0.storage, self.offset() + 40)
}

///|
pub fn PlanFrameDecodeRow::sampling_top_p_value(
  self : PlanFrameDecodeRow,
) -> Double raise WorkerWireError {
  read_u64(self.0.storage, self.offset() + 64).reinterpret_as_double()
}
