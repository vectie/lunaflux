///|
fn service_index_scoring(j : Json) -> @precision.IndexScoringLaw raise {
  guard j is Object(fields) else { fail("expected service object") }
  match fields.get("index_scoring") {
    None | Some(String("ordered-bf16")) => OrderedBFloat16
    Some(String("row-scaled-e4m3-matrix")) => RowScaledE4M3Matrix
    _ => fail("unsupported index_scoring")
  }
}

///|
fn service_sparse_cache(j : Json) -> @precision.SparseCacheLaw raise {
  guard j is Object(fields) else { fail("service object") }
  match fields.get("sparse_cache") {
    None | Some(String("expanded-bf16")) => ExpandedBFloat16
    Some(String("latent-bf16")) => LatentBFloat16
    _ => fail("unsupported sparse_cache")
  }
}

///|
fn ServicePlan::sparse_description(self : Self) -> String {
  let index = match self.index_scoring {
    OrderedBFloat16 => "ordered-bf16"
    RowScaledE4M3Matrix => "row-scaled-e4m3-matrix"
  }
  let cache = match self.sparse_cache {
    ExpandedBFloat16 => "expanded-bf16"
    LatentBFloat16 => "latent-bf16"
  }
  "index_scoring=\{index} sparse_cache=\{cache}"
}

///|
test "latent cache is explicit and unknown storage never falls back" {
  assert_true(service_sparse_cache({}) is ExpandedBFloat16)
  assert_true(
    service_sparse_cache({ "sparse_cache": "latent-bf16" }) is LatentBFloat16,
  )
  try service_sparse_cache({ "sparse_cache": "automatic" }) catch {
    _ => return
  } noraise {
    _ => fail("accepted unknown cache")
  }
}

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
  assert_true(service_index_scoring({}) is OrderedBFloat16)
  assert_true(
    service_index_scoring({ "index_scoring": "row-scaled-e4m3-matrix" })
    is RowScaledE4M3Matrix,
  )
  try service_index_scoring({ "index_scoring": "automatic" }) catch {
    _ => ()
  } noraise {
    _ => fail("unknown index scoring accepted")
  }
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
