///|
pub struct WeightEntry {
  priv name : String
  priv bytes : Int64
  priv weight : @checkpoint.PackedCheckpointWeight
}

///|
pub fn WeightEntry::name(self : Self) -> String {
  self.name
}

///|
pub fn WeightEntry::byte_count(self : Self) -> Int64 {
  self.bytes
}

///|
/// Startup-only dense BF16 bindings; no repacking or duplicate device bank.
pub struct WeightPlan {
  priv checkpoint : @checkpoint.PackedCheckpoint
  priv entries : ReadOnlyArray[WeightEntry]
  priv bytes : Int64
}

///|
fn selected_requirements(
  spec : @model.Spec,
  program : Program,
) -> Array[@weights.Requirement] raise {
  let (raw, encoded, _) = program.row_capacities()
  guard (encoded < raw) == (spec.downsample() is SixteenX) else {
    fail("vision weights and program compression differ")
  }
  let available : Map[String, @weights.Requirement] = Map([])
  for requirement in @weights.requirements(spec) {
    available[requirement.name] = requirement
  }
  program
  .weight_names()
  .map(name => {
    guard available.get(name) is Some(requirement) else {
      fail("unknown vision tensor " + name)
    }
    requirement
  })[:].to_owned()
}

///|
pub fn WeightPlan::new(
  spec : @model.Spec,
  program : Program,
  checkpoint : @checkpoint.PackedCheckpoint,
) -> Self raise {
  let entries = []
  let mut bytes = 0L
  for requirement in selected_requirements(spec, program) {
    let shape = requirement.shape
    let columns = shape[shape.length() - 1]
    let rows = shape[:shape.length() - 1].fold(init=1L, (a, b) => a * b)
    let representation = @format.Representation::new(
      format=Bf16,
      rows=rows.to_int(),
      columns=columns.to_int(),
    )
    let weight = checkpoint.bind_dense_tensor(
      representation~,
      payload_name=requirement.name,
      source_shape=shape[:],
    )
    entries.push({
      name: requirement.name,
      bytes: weight.buffer_bytes(),
      weight,
    })
    bytes += weight.buffer_bytes()
  }
  { checkpoint, entries: ReadOnlyArray::from_array(entries), bytes, }
}

///|
pub fn WeightPlan::entries(self : Self) -> ReadOnlyArray[WeightEntry] {
  self.entries
}

///|
pub fn WeightPlan::byte_count(self : Self) -> Int64 {
  self.bytes
}

///|
/// One bounded transfer session and scratch for all selected tensors. This is
/// only startup I/O; offsets always refer to the final destination allocation.
pub fn WeightPlan::transfer(
  self : Self,
  root : @approved_fs.ApprovedRoot,
  scratch~ : FixedArray[Byte],
  limits~ : @streaming.StreamingSafetensorsLimits,
  consume~ : (Int32, Int64, FixedArray[Byte], Int32) -> Bool,
) -> Unit raise {
  guard scratch.length() > 0 else { fail("empty vision weight scratch") }
  let weights = self.entries.map(entry => entry.weight)
  self.checkpoint.with_transfer_session(root, limits, checkpoint => {
    checkpoint.transfer_weights(root, weights[:], scratch~, limits~, consume~)
  })
}

///|
priv struct WeightRegion {
  allocation : @device.Allocation
  bytes : Int64
}

///|
/// Borrowed immutable allocations; close frames/execution before their owner.
pub struct Weights {
  priv regions : Map[String, WeightRegion]
}

///|
/// Borrow an existing GPU weight owner. Shapes come from the selected model,
/// never from the allocation size or a caller-supplied untyped descriptor.
pub fn Weights::borrow(
  spec : @model.Spec,
  program : Program,
  context : @device.Context,
  lookup : (String) -> @device.Allocation raise,
) -> Self raise {
  let regions : Map[String, WeightRegion] = Map([])
  for requirement in selected_requirements(spec, program) {
    let bytes = requirement.shape.fold(init=2L, (a, b) => a * b)
    let allocation = lookup(requirement.name)
    context.validate_allocation_region(
      allocation,
      offset=0L,
      byte_count=bytes,
      alignment=2L,
    )
    regions[requirement.name] = { allocation, bytes, }
  }
  { regions, }
}
