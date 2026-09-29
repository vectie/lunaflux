///|
priv enum PointwiseProgram {
  Embedding(@physical_tile_ir.VectorOwnership)
  Residual(@physical_tile_ir.VectorOwnership)
  Normalize(@elementwise_physical_ir.NormalizationProgram)
  NormalizeHeads(@elementwise_physical_ir.NormalizationProgram)
  Rotary(@elementwise_physical_ir.RotaryProgram)
  WriteKv(@elementwise_physical_ir.KvWriteProgram)
} derive(Eq, Debug)

///|
fn pointwise_program(
  parameters : PointwiseParameters,
) -> PointwiseProgram raise PointwiseAotError {
  match parameters.family {
    EmbeddingLookup | ResidualAdd => {
      guard @physical_tile_ir.VectorOwnership::new(
          vectors=parameters.width,
          owners=256,
        )
        is Some(ownership) else {
        raise Unsupported(Shape)
      }
      if parameters.family == EmbeddingLookup {
        Embedding(ownership)
      } else {
        Residual(ownership)
      }
    }
    RmsNorm | QkRmsNorm => {
      let heads = parameters.family == QkRmsNorm
      guard @elementwise_physical_ir.NormalizationProgram::new(
          width=if heads { parameters.head_dimension } else { parameters.width },
          owners=if heads { 32 } else { 256 },
          subgroup_lanes=32,
          residual=false,
        )
        is Some(norm) else {
        raise Unsupported(Shape)
      }
      if heads {
        NormalizeHeads(norm)
      } else {
        Normalize(norm)
      }
    }
    PositionedRotary => {
      guard @elementwise_physical_ir.RotaryProgram::new(
          width=parameters.width,
          query_heads=parameters.query_heads,
          key_heads=parameters.kv_heads,
          head_dimension=parameters.head_dimension,
          owners=32,
        )
        is Some(rotary) else {
        raise Unsupported(Shape)
      }
      Rotary(rotary)
    }
    KvCacheWrite => {
      guard @elementwise_physical_ir.KvWriteProgram::new(
          query_heads=parameters.query_heads,
          kv_heads=parameters.kv_heads,
          head_dimension=parameters.head_dimension,
          page_stride_bytes=parameters.page_stride_bytes,
          element_bytes=2,
          owners=256,
        )
        is Some(kv) else {
        raise Unsupported(Shape)
      }
      WriteKv(kv)
    }
  }
}

///|
fn PointwiseProgram::work_per_token(
  self : PointwiseProgram,
  parameters : PointwiseParameters,
) -> Int {
  match self {
    Normalize(_) => 1
    NormalizeHeads(_) => parameters.query_heads + 2 * parameters.kv_heads
    Rotary(rotary) => rotary.work_per_token()
    WriteKv(kv) => {
      let (_, width, _) = kv.geometry()
      width
    }
    Embedding(ownership) | Residual(ownership) => {
      let (width, _, _) = ownership.geometry()
      width
    }
  }
}

///|
fn normalization_reduction_source(
  program : @elementwise_physical_ir.NormalizationProgram,
) -> String {
  let (_, _, lanes) = program.geometry()
  let partials = program.partial_count()
  let output = StringBuilder()
  for effect in program.reduction() {
    output.write_string(
      match effect {
        ReduceSubgroup =>
          "  for (int offset = \{lanes / 2}; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, offset);\n"
        PublishPartials => "  if (lane == 0) warp_sums[warp] = sum;\n"
        SynchronizeOwners => "  __syncthreads();\n"
        ReducePartials =>
          "  float total = lane < \{partials} ? warp_sums[lane] : 0.0f;\n" +
          "  if (warp == 0) {\n" +
          "    for (int offset = \{lanes / 2}; offset > 0; offset >>= 1) total += __shfl_down_sync(0xffffffffU, total, offset);\n"
        PublishTotal => "    if (lane == 0) warp_sums[0] = total;\n  }\n"
        ReduceOwnerTree(_) => abort("pointwise norm uses subgroup reduction")
      },
    )
  }
  output.to_string()
}

///|
fn qk_normalization_reduction_source(
  program : @elementwise_physical_ir.NormalizationProgram,
) -> String {
  let (_, _, lanes) = program.geometry()
  let source = StringBuilder()
  for effect in program.reduction() {
    source.write_string(
      match effect {
        ReduceSubgroup =>
          "  for (int offset = \{lanes / 2}; offset > 0; offset >>= 1)\n" +
          "    sum += __shfl_down_sync(0xffffffffU, sum, offset);\n"
        _ => abort("per-head pointwise normalization requires one subgroup")
      },
    )
  }
  source.to_string()
}
