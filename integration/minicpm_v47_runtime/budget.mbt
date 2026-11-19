///|
/// Explicit allocation payloads only: driver/module/queue overhead is separate.
pub(all) struct Budget {
  weight_bytes : Int64
  storage_bytes : Int64
  output_bytes : Int64
  total_allocation_bytes : Int64
}

///|
pub fn Budget::for_text(
  program : @text.Program,
  weight_bytes~ : Int64,
) -> Self raise {
  let storage = program.storage_budget()
  let mutable_bytes = storage.owned_bytes + storage.output_workspace_bytes
  guard weight_bytes >= 0L &&
    weight_bytes <= 9223372036854775807L - mutable_bytes else {
    fail("invalid resident allocation budget")
  }
  {
    weight_bytes,
    storage_bytes: storage.owned_bytes,
    output_bytes: storage.output_workspace_bytes,
    total_allocation_bytes: weight_bytes + mutable_bytes,
  }
}
