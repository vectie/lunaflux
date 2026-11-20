///|
const PROJECTION_BF16_BYTES : Int64 = 2L

///|
const PROJECTION_STEP_COUNTS_BYTES : Int64 = 20L

///|
const PROJECTION_VALUE_ALIGNMENT : Int64 = 16L

///|
priv struct ProjectionParameters {
  family : ProjectionAotFamily
  input_width : Int32
  query_width : Int32
  output_width : Int32
  intermediate_width : Int32
}

///|
/// CUDA's private shape adapter into the backend-neutral strategy layer.
fn projection_strategy_shape(
  parameters : ProjectionParameters,
) -> @luna_projection_strategy.ProjectionShape {
  let family = match parameters.family {
    QkvProjection => @luna_projection_strategy.Qkv
    DenseProjection => Dense
    GatedMlp => GatedMlp
    LanguageModelHead => LanguageModelHead
  }
  @luna_projection_strategy.ProjectionShape::new(
    family,
    input_width=parameters.input_width,
    output_width=parameters.output_width,
    intermediate_width=parameters.intermediate_width,
  ) catch {
    _ => abort("validated projection shape rejected by strategy")
  }
}

///|
/// CUDA capabilities are lowering facts, not workload-selection policy.
fn projection_cuda_capabilities(
  parameters : ProjectionParameters,
) -> @luna_projection_strategy.ProjectionCapabilities {
  let matrix_tile = if parameters.family == GatedMlp {
    if parameters.input_width % 256 == 0 &&
      parameters.output_width % 256 == 0 &&
      parameters.intermediate_width % 256 == 0 &&
      parameters.intermediate_width <= 3840 {
      Some(
        @luna_projection_strategy.ProjectionTile::new(
          rows=16,
          columns=16,
          reduction=16,
        ) catch {
          _ => abort("static CUDA MLP tile rejected")
        },
      )
    } else {
      None
    }
  } else if parameters.input_width % 16 == 0 &&
    parameters.output_width % 16 == 0 &&
    (
      parameters.family != QkvProjection ||
      (
        parameters.query_width > 0 &&
        parameters.query_width % 16 == 0 &&
        parameters.intermediate_width > 0 &&
        parameters.intermediate_width % 16 == 0
      )
    ) {
    Some(
      @luna_projection_strategy.ProjectionTile::new(
        rows=16,
        columns=16,
        reduction=16,
      ) catch {
        _ => abort("static CUDA matrix tile rejected")
      },
    )
  } else {
    None
  }
  @luna_projection_strategy.ProjectionCapabilities::new(
    subgroup_width=32,
    matrix_tile~,
  ) catch {
    _ => abort("static CUDA projection capability rejected")
  }
}

///|
fn projection_cuda_backend_id() -> @luna_projection_strategy.ProjectionBackendId {
  @luna_projection_strategy.ProjectionBackendId::new("cuda-aot-v1") catch {
    _ => abort("static CUDA backend identity rejected")
  }
}

///|
fn projection_cuda_target_id(
  target : @catalog.DeviceTarget,
) -> @luna_projection_strategy.ProjectionTargetId {
  @luna_projection_strategy.ProjectionTargetId::new(
    "sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string(),
  ) catch {
    _ => abort("validated CUDA target identity rejected")
  }
}

///|
fn projection_build_runtime_dispatch(
  parameters : ProjectionParameters,
  maximum_rows : Int32,
  target : @catalog.DeviceTarget,
  records : ArrayView[@luna_projection_strategy.ProjectionAutotuneRecord],
) -> @luna_projection_strategy.ProjectionRuntimeDispatch raise ProjectionAotError {
  @luna_projection_strategy.build_projection_runtime_dispatch(
    records,
    backend=projection_cuda_backend_id(),
    target=projection_cuda_target_id(target),
    numeric_type=ProjectionBFloat16,
    shape=projection_strategy_shape(parameters),
    maximum_rows~,
    capabilities=projection_cuda_capabilities(parameters),
  ) catch {
    _ => raise Invalid(Strategy)
  }
}

///|
fn projection_validate_runtime_dispatch(
  runtime : @luna_projection_strategy.ProjectionRuntimeDispatch,
  parameters : ProjectionParameters,
  maximum_rows : Int32,
  target : @catalog.DeviceTarget,
) -> @luna_projection_strategy.ProjectionDispatchPlan raise ProjectionAotError {
  if runtime.backend() != projection_cuda_backend_id() ||
    runtime.target() != projection_cuda_target_id(target) ||
    runtime.numeric_type() != ProjectionBFloat16 ||
    runtime.shape() != projection_strategy_shape(parameters) ||
    runtime.maximum_rows() < maximum_rows {
    raise Mismatch(Strategy)
  }
  let dispatch = runtime.dispatch_plan(maximum_rows) catch {
    _ => raise Mismatch(Strategy)
  }
  let capabilities = projection_cuda_capabilities(parameters)
  let subgroup = capabilities.subgroup_width()
  let decode = dispatch.decode()
  let prefill = dispatch.prefill()
  let expected_decode_tile = @luna_projection_strategy.ProjectionTile::new(
    rows=1,
    columns=subgroup,
    reduction=subgroup,
  ) catch {
    _ => raise Invalid(Strategy)
  }
  if decode.tile() != expected_decode_tile {
    raise Mismatch(Strategy)
  }
  // A single-row profile has no admitted prefill bucket. The neutral runtime
  // therefore returns its decode selection for both fields. Validate that
  // exact alias instead of requiring a prefill-only kernel that can never run.
  let prefill_supported = if maximum_rows == 1 {
    prefill == decode
  } else {
    match prefill.kernel_class() {
      MatrixTile => capabilities.matrix_tile() == Some(prefill.tile())
      ScalarTile =>
        prefill.tile() ==
        (@luna_projection_strategy.ProjectionTile::new(
          rows=1,
          columns=subgroup,
          reduction=1,
        ) catch {
          _ => raise Invalid(Strategy)
        })
      SubgroupGemv => false
    }
  }
  let pair_supported = if maximum_rows == 1 {
    prefill == decode
  } else {
    match (decode.kernel_class(), prefill.kernel_class()) {
      (SubgroupGemv, MatrixTile) | (ScalarTile, ScalarTile) => true
      _ => false
    }
  }
  if !prefill_supported || !pair_supported {
    raise Mismatch(Strategy)
  }
  dispatch
}
