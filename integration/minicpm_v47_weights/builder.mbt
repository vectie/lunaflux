///|
pub struct AllocationPlan {
  priv label : String
  priv sources : ReadOnlyArray[String]
  priv bytes : Int64
  priv weights : ReadOnlyArray[@checkpoint.PackedCheckpointWeight]
  priv destinations : ReadOnlyArray[(Int32, Int64)]
}

///|
pub fn AllocationPlan::label(self : Self) -> String {
  self.label
}

///|
pub fn AllocationPlan::sources(self : Self) -> ReadOnlyArray[String] {
  self.sources
}

///|
pub fn AllocationPlan::byte_count(self : Self) -> Int64 {
  self.bytes
}

///|
priv struct Builder {
  checkpoint : @checkpoint.PackedCheckpoint
  operators : @operators.DecoderOperators
  entries : Array[AllocationPlan]
}

///|
fn Builder::tensor(
  self : Self,
  name : String,
  shape : ArrayView[Int64],
) -> Int32 raise {
  guard !shape.is_empty() else { fail("empty BF16 tensor shape") }
  let columns = shape[shape.length() - 1]
  let rows = for dimension in shape[:shape.length() - 1]; product = 1L {
    guard dimension > 0L && product <= 2147483647L / dimension else {
      fail("BF16 tensor extent exceeds matrix representation")
    }
    continue product * dimension
  } nobreak {
    product
  }
  guard columns > 0L && columns <= 2147483647L else {
    fail("invalid BF16 tensor columns")
  }
  let representation = @format.Representation::new(
    format=Bf16,
    rows=rows.to_int(),
    columns=columns.to_int(),
  )
  let weight = self.checkpoint.bind_dense_tensor(
    representation~,
    payload_name=name,
    source_shape=shape,
  )
  let ordinal = self.entries.length()
  self.entries.push({
    label: name,
    sources: [name],
    bytes: weight.buffer_bytes(),
    weights: [weight],
    destinations: [(0, 0L)],
  })
  ordinal
}

///|
fn Builder::matrix(
  self : Self,
  name : String,
  rows : Int32,
  columns : Int32,
) -> Int32 raise {
  self.tensor(name, [rows.to_int64(), columns.to_int64()])
}

///|
fn Builder::vector(self : Self, name : String, width : Int32) -> Int32 raise {
  self.tensor(name, [width.to_int64()])
}

///|
fn Builder::experts(self : Self, layer : Int32) -> Int32 raise {
  let plan = self.operators.routed_experts()
  let slices = @moe.bind_expert_weights(self.checkpoint, plan, layer~)
  let destinations = slices.mapi((i, _) => {
    (
      0,
      (i / 3).to_int64() * plan.expert_stride() + plan.weight_offsets()[i % 3],
    )
  })
  let prefix = "model.language_model.layers.\{layer}.mlp.experts."
  let ordinal = self.entries.length()
  self.entries.push({
    label: "layer.\{layer}.routed_bank",
    sources: [prefix + "gate_up_proj", prefix + "down_proj"],
    bytes: plan.bank_bytes(),
    weights: ReadOnlyArray::from_array(slices),
    destinations: ReadOnlyArray::from_array(destinations),
  })
  ordinal
}
