///|
/// A distinct numerical program, selected at startup and identically consumed
/// by AOT export and execution. External comparisons never authorize rewrites.
fn service_sparse_accumulation(
  j : Json,
) -> @precision.AttentionAccumulationLaw raise {
  guard j is Object(fields) else { fail("expected service object") }
  match fields.get("sparse_attention_accumulation") {
    None | Some(String("ordered-f32")) => OrderedFloat32
    Some(String("tiled-f32")) => TiledFloat32
    _ => fail("unsupported sparse attention numerical program")
  }
}

///|
test "sparse arithmetic is explicit and unknown policies reject" {
  assert_true(service_sparse_accumulation({}) is OrderedFloat32)
  assert_true(
    service_sparse_accumulation({ "sparse_attention_accumulation": "tiled-f32" })
    is TiledFloat32,
  )
  try
    service_sparse_accumulation({ "sparse_attention_accumulation": "automatic" })
  catch {
    _ => ()
  } noraise {
    _ => fail("unknown policy accepted")
  }
}
