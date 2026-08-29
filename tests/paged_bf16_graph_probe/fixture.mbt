///|
suberror GraphProbeFailure {
  GraphProbeFailure(String)
}

///|
fn require_graph(
  condition : Bool,
  message : String,
) -> Unit raise GraphProbeFailure {
  if !condition {
    raise GraphProbeFailure(message)
  }
}

///|
fn fixture_bytes(
  kind : Int,
  byte_count : Int,
) -> FixedArray[Byte] raise GraphProbeFailure {
  let result = FixedArray::make(byte_count, b'\x00')
  require_graph(
    fill_graph_fixture(kind, result),
    "CPU fixture construction failed for kind \{kind}",
  )
  result
}

///|
fn upload_fixture(
  context : @device.Context,
  allocation : @device.Allocation,
  kind : Int,
  byte_count : Int,
  destination_offset : Int64,
) -> Unit raise {
  let bytes = fixture_bytes(kind, byte_count)
  context.copy_from_fixed_host(
    allocation,
    bytes,
    0L,
    destination_offset,
    byte_count.to_int64(),
  )
}

///|
fn validate_graph_outputs(
  outputs : FixedArray[Bytes],
) -> (Int, Double) raise GraphProbeFailure {
  validate_graph_reference(outputs)
}
