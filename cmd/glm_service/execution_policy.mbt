///|
/// Startup selection only: these are existing numerical/physical IR values,
/// not an alternative compiler or a benchmark-driven runtime policy.
priv struct ExecutionPolicy {
  recurrent : @precision.RecurrentEvaluationLaw
  staging : @matrix.MatrixStaging
  transport : @matrix.ExpertTransport
  scale_partitions : Int
}

///|
fn execution_policy(j : Json) -> ExecutionPolicy raise {
  guard j is Object(fields) else { fail("expected service object") }
  let recurrent : @precision.RecurrentEvaluationLaw = match
    fields.get("recurrent_evaluation") {
    None | Some(String("ordered-token-f32")) => OrderedTokenFloat32
    Some(String("chunk-triangular-f32")) => ChunkTriangularFloat32
    _ => fail("unsupported recurrent_evaluation")
  }
  let staging : @matrix.MatrixStaging = match fields.get("projection_staging") {
    None | Some(String("synchronous")) => Synchronous
    Some(String("buffered-k32")) => Buffered(inner_tile=32)
    _ => fail("unsupported projection_staging")
  }
  let transport : @matrix.ExpertTransport = match
    fields.get("expert_transport") {
    None | Some(String("packed-rows")) => PackedRows
    Some(String("bulk-tiles-128x2x48")) =>
      PackedBulkTiles(columns=128, consumer_teams=2, workers=48)
    _ => fail("unsupported expert_transport")
  }
  let scale_partitions = match fields.get("expert_scale_partitions") {
    None | Some(Number(1.0, ..)) => 1
    Some(Number(32.0, ..)) => 32
    _ => fail("unsupported expert_scale_partitions")
  }
  { recurrent, staging, transport, scale_partitions, }
}

///|
fn ExecutionPolicy::description(self : Self) -> String {
  let recurrent = match self.recurrent {
    OrderedTokenFloat32 => "ordered-token-f32"
    ChunkTriangularFloat32 => "chunk-triangular-f32"
  }
  let staging = match self.staging {
    Synchronous => "synchronous"
    Buffered(inner_tile~) => "buffered-k\{inner_tile}"
  }
  "recurrent=\{recurrent} projection_staging=\{staging} expert_transport=\{Repr(self.transport)} expert_scale_partitions=\{self.scale_partitions}"
}

///|
test "HTTP execution defaults preserve the validated numerical and physical plan" {
  let p = execution_policy({})
  assert_true(p.recurrent is OrderedTokenFloat32)
  assert_true(p.staging is Synchronous)
  assert_true(p.transport is PackedRows)
  assert_eq(p.scale_partitions, 1)
}

///|
test "HTTP compiler alternatives compose independently without implicit selection" {
  let scales = execution_policy({ "expert_scale_partitions": 32 })
  assert_eq(scales.scale_partitions, 32)
  assert_true(scales.transport is PackedRows)
  assert_true(scales.staging is Synchronous)
  assert_true(scales.recurrent is OrderedTokenFloat32)
  let bulk = execution_policy({ "expert_transport": "bulk-tiles-128x2x48" })
  assert_true(
    bulk.transport is PackedBulkTiles(columns=128, consumer_teams=2, workers=48),
  )
  assert_eq(bulk.scale_partitions, 1)
  let all = execution_policy({
    "expert_scale_partitions": 32,
    "expert_transport": "bulk-tiles-128x2x48",
    "projection_staging": "buffered-k32",
    "recurrent_evaluation": "chunk-triangular-f32",
  })
  assert_true(all.recurrent is ChunkTriangularFloat32)
  assert_true(all.staging is Buffered(inner_tile=32))
  assert_true(all.description().contains("expert_scale_partitions=32"))
}

///|
test "unknown execution choices cannot silently use the serving baseline" {
  let cases : Array[Json] = [
    { "expert_scale_partitions": 0 },
    { "expert_scale_partitions": 31.5 },
    { "expert_scale_partitions": "32" },
    { "expert_transport": "automatic" },
    { "projection_staging": "buffered-k128" },
    { "recurrent_evaluation": "automatic" },
  ]
  for value in cases {
    try execution_policy(value) catch {
      _ => ()
    } noraise {
      _ => fail("unsupported execution choice accepted")
    }
  }
}
