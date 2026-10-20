///|
/// Startup selection only: these are existing numerical/physical IR values,
/// not an alternative compiler or a benchmark-driven runtime policy.
priv struct ExecutionPolicy {
  recurrent : @precision.RecurrentEvaluationLaw
  staging : @matrix.MatrixStaging
  sparse_staging : @matrix.MatrixStaging
  recurrent_tile_rows : Int
  recurrent_reuse_minimum_inner : Int
  transport : @matrix.ExpertTransport
  scale_partitions : Int
  moe_reduction : @precision.MoeReductionPlacement
  tensor_publication : @precision.ProjectionPublication
  exact_capacity_replay : Bool
  collective_window_sites : Int
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
    Some(String("synchronous-k64")) => SynchronousTiles(inner_tile=64)
    _ => fail("unsupported projection_staging")
  }
  let recurrent_tile_rows = match fields.get("recurrent_projection_tile_rows") {
    None | Some(Number(16.0, ..)) => 16
    Some(Number(32.0, ..)) if staging is (Synchronous | SynchronousTiles(_)) =>
      32
    _ => fail("unsupported recurrent_projection_tile_rows")
  }
  let transport : @matrix.ExpertTransport = match
    fields.get("expert_transport") {
    None | Some(String("packed-rows")) => PackedRows
    Some(String("bulk-tiles-128x2x48")) =>
      PackedBulkTiles(columns=128, consumer_teams=2, workers=48)
    _ => fail("unsupported expert_transport")
  }
  let recurrent_reuse_minimum_inner = match
    fields.get("recurrent_reuse_minimum_inner") {
    None => 0
    Some(Number(n, ..)) if n >= 0.0 &&
      n <= 1073741760.0 &&
      n.to_int().to_double() == n => n.to_int()
    _ => fail("invalid recurrent reuse envelope")
  }
  let sparse_staging : @matrix.MatrixStaging = match
    fields.get("sparse_projection_staging") {
    None => staging
    Some(
      {
        "kind": "workset-bounded-synchronous",
        "inner_tile": Number(k, ..),
        "maximum_weight_bytes": Number(bytes, ..),
        "maximum_tiles": Number(tiles, ..),
        ..
      }
    ) => {
      guard (k == 32.0 || k == 64.0 || k == 128.0) &&
        bytes >= 1.0 &&
        bytes <= 2147483647.0 &&
        bytes.to_int().to_double() == bytes &&
        tiles >= 1.0 &&
        tiles <= 2147483647.0 &&
        tiles.to_int().to_double() == tiles else {
        fail("invalid sparse staging resource envelope")
      }
      WorksetBoundedTiles(
        inner_tile=k.to_int(),
        maximum_weight_bytes=bytes.to_int().to_int64(),
        maximum_tiles=tiles.to_int(),
      )
    }
    _ => fail("unsupported sparse_projection_staging")
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
  let collective_window_sites = match fields.get("collective_window_sites") {
    None => 0
    Some(Number(n, ..)) if n >= 1.0 &&
      n <= 2147483647.0 &&
      n.to_int().to_double() == n &&
      exact_capacity_replay => n.to_int()
    _ =>
      fail(
        "collective_window_sites requires exact replay and a positive integer",
      )
  }
  {
    recurrent,
    staging,
    sparse_staging,
    recurrent_tile_rows,
    recurrent_reuse_minimum_inner,
    transport,
    scale_partitions,
    moe_reduction,
    tensor_publication,
    exact_capacity_replay,
    collective_window_sites,
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
    SynchronousTiles(inner_tile~) => "synchronous-k\{inner_tile}"
    WorksetBoundedTiles(inner_tile~, maximum_weight_bytes~, maximum_tiles~) =>
      "bounded-synchronous-k\{inner_tile}-bytes\{maximum_weight_bytes}-tiles\{maximum_tiles}"
    Buffered(inner_tile~) => "buffered-k\{inner_tile}"
  }
  "recurrent=\{recurrent} projection_staging=\{staging} recurrent_projection_tile_rows=\{self.recurrent_tile_rows} recurrent_reuse_minimum_inner=\{self.recurrent_reuse_minimum_inner} sparse_projection_staging=\{Repr(self.sparse_staging)} expert_transport=\{Repr(self.transport)} expert_scale_partitions=\{self.scale_partitions} moe_reduction=\{Repr(self.moe_reduction)} projection_reduction=\{Repr(self.tensor_publication)} exact_capacity_replay=\{self.exact_capacity_replay} collective_window_sites=\{self.collective_window_sites}"
}

///|
fn ExecutionPolicy::replay(
  self : Self,
  capacity : Int,
) -> @parallel.ReplayPolicy {
  if self.exact_capacity_replay && capacity > 1 {
    if self.collective_window_sites > 0 {
      CaptureWindowedCollectives(capacity, self.collective_window_sites)
    } else {
      CaptureExactCollectives(capacity)
    }
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
  assert_eq(p.sparse_staging, p.staging)
  assert_true(p.transport is PackedRows)
  assert_eq(p.scale_partitions, 1)
  assert_eq(p.moe_reduction, ExpertPartialsFloat32)
  assert_eq(p.tensor_publication, ReduceFloat32BeforeRound)
}

///|
test "sparse projection workset selection is independent of fused recurrent scratch" {
  let p = execution_policy({
    "sparse_projection_staging": {
      "kind": "workset-bounded-synchronous",
      "inner_tile": 128,
      "maximum_weight_bytes": 16777216,
      "maximum_tiles": 192,
    },
  })
  assert_true(p.staging is Synchronous)
  assert_true(
    p.sparse_staging
    is WorksetBoundedTiles(
      inner_tile=128,
      maximum_weight_bytes=16777216L,
      maximum_tiles=192
    ),
  )
  assert_true(
    p.description().contains("sparse_projection_staging=WorksetBoundedTiles"),
  )
  for bytes in [0.0, -1.0, 1.5] {
    try
      execution_policy({
        "sparse_projection_staging": {
          "kind": "workset-bounded-synchronous",
          "inner_tile": 128,
          "maximum_weight_bytes": bytes,
          "maximum_tiles": 192,
        },
      })
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid physical workset")
    }
  }
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
    { "collective_window_sites": 16 },
    { "collective_replay": "exact-capacity", "collective_window_sites": 0 },
    { "collective_replay": "exact-capacity", "collective_window_sites": 1.5 },
  ]
  for value in cases {
    try execution_policy(value) catch {
      _ => ()
    } noraise {
      _ => fail("unsupported execution choice accepted")
    }
  }
}

///|
test "bounded graph horizon is explicit and independent of arithmetic" {
  let p = execution_policy({
    "collective_replay": "exact-capacity",
    "collective_window_sites": 16,
  })
  assert_true(p.replay(64) is CaptureWindowedCollectives(64, 16))
  assert_true(p.replay(1) is CaptureDecodeCollectives)
  assert_true(p.tensor_publication is ReduceFloat32BeforeRound)
}
