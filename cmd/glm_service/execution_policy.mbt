///|
/// Startup selection only: these are existing numerical/physical IR values,
/// not an alternative compiler or a benchmark-driven runtime policy.
priv struct ExecutionPolicy {
  recurrent : @precision.RecurrentEvaluationLaw
  staging : @matrix.MatrixStaging
  transport : @matrix.ExpertTransport
  scale_partitions : Int
  moe_reduction : @precision.MoeReductionPlacement
  tensor_publication : @precision.ProjectionPublication
  exact_capacity_replay : Bool
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
  let moe_reduction : @precision.MoeReductionPlacement = match
    fields.get("moe_reduction") {
    None | Some(String("expert-partials-f32")) => ExpertPartialsFloat32
    Some(String("local-combined-bf16")) => LocalCombinedBFloat16
    _ => fail("unsupported moe_reduction")
  }
  let tensor_publication : @precision.ProjectionPublication = match
    fields.get("projection_reduction") {
    None | Some(String("f32-before-round")) => ReduceFloat32BeforeRound
    Some(String("bf16-after-round")) => ReduceBFloat16AfterRound
    _ => fail("unsupported projection_reduction")
  }
  let exact_capacity_replay = match fields.get("collective_replay") {
    None | Some(String("decode-only")) => false
    Some(String("exact-capacity")) => true
    _ => fail("unsupported collective_replay")
  }
  {
    recurrent,
    staging,
    transport,
    scale_partitions,
    moe_reduction,
    tensor_publication,
    exact_capacity_replay,
  }
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
  "recurrent=\{recurrent} projection_staging=\{staging} expert_transport=\{Repr(self.transport)} expert_scale_partitions=\{self.scale_partitions} moe_reduction=\{Repr(self.moe_reduction)} projection_reduction=\{Repr(self.tensor_publication)} exact_capacity_replay=\{self.exact_capacity_replay}"
}

///|
fn ExecutionPolicy::replay(
  self : Self,
  capacity : Int,
) -> @parallel.ReplayPolicy {
  if self.exact_capacity_replay && capacity > 1 {
    CaptureExactCollectives(capacity)
  } else {
    CaptureDecodeCollectives
  }
}

///|
test "exact prefill replay is explicit and does not change numerical choices" {
  assert_true(execution_policy({}).replay(64) is CaptureDecodeCollectives)
  let p = execution_policy({ "collective_replay": "exact-capacity" })
  assert_true(p.replay(64) is CaptureExactCollectives(64))
  assert_true(p.replay(1) is CaptureDecodeCollectives)
  assert_true(p.tensor_publication is ReduceFloat32BeforeRound)
  assert_true(p.description().contains("exact_capacity_replay=true"))
}

///|
test "HTTP execution defaults preserve the validated numerical and physical plan" {
  let p = execution_policy({})
  assert_true(p.recurrent is OrderedTokenFloat32)
  assert_true(p.staging is Synchronous)
  assert_true(p.transport is PackedRows)
  assert_eq(p.scale_partitions, 1)
  assert_eq(p.moe_reduction, ExpertPartialsFloat32)
  assert_eq(p.tensor_publication, ReduceFloat32BeforeRound)
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
    "moe_reduction": "local-combined-bf16",
    "projection_reduction": "bf16-after-round",
  })
  assert_true(all.recurrent is ChunkTriangularFloat32)
  assert_eq(all.moe_reduction, LocalCombinedBFloat16)
  assert_eq(all.tensor_publication, ReduceBFloat16AfterRound)
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
    { "moe_reduction": "automatic" },
    { "projection_reduction": "automatic" },
    { "collective_replay": "automatic" },
  ]
  for value in cases {
    try execution_policy(value) catch {
      _ => ()
    } noraise {
      _ => fail("unsupported execution choice accepted")
    }
  }
}
