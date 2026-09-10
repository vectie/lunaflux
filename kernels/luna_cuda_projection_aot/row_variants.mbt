///|
/// Compile measured per-bucket strategies into the same immutable AOT module.
/// Strategy equality does not imply executable equality: partial evaluation
/// of a bounded row map may remove complete producer and accumulator tiles.
/// These variants share allocations but accept only their declared row domain.
fn projection_row_variant_sources(
  parameters : ProjectionParameters,
  profile : @launch_contract.PagedKernelProfile,
  runtime : @luna_projection_strategy.ProjectionRuntimeDispatch,
  primary : @luna_projection_strategy.ProjectionDispatchPlan,
  symbol : String,
  max_registers~ : Int,
) -> (String, ReadOnlyArray[@artifact.KernelRowVariant]) raise ProjectionAotError {
  let sources = StringBuilder()
  let variants : Array[@artifact.KernelRowVariant] = []
  let primary_compilation = projection_row_variant_compilation(
    parameters,
    primary,
    profile.max_query_rows(),
    profile.max_query_tokens(),
  )
  for
    bound in [
      8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
      131072, 262144, 524288, 1048576,
    ] {
    if bound > profile.max_query_tokens() {
      break
    }
    let selected = runtime.select(bound) catch {
      _ => raise Invalid(CompilerPolicy)
    }
    let strategy = selected.strategy()
    if strategy.kernel_class() != MatrixTile ||
      strategy.kernel_class() != primary.prefill().kernel_class() {
      continue
    }
    // A row-domain specialization is not a new autotune recommendation. Keep
    // the installed primary distribution unless a measured winner overrides it.
    let dispatch = if selected.source() == OfflineAutotune {
      runtime.dispatch_plan(bound) catch {
        _ => raise Invalid(CompilerPolicy)
      }
    } else {
      primary
    }
    let compilation = projection_row_variant_compilation(
      parameters,
      dispatch,
      Int::min(profile.max_query_rows(), bound),
      bound,
    )
    let measured_distribution = selected.source() == OfflineAutotune &&
      (
        strategy.output_tiles_per_workgroup() !=
        primary.prefill().output_tiles_per_workgroup() ||
        strategy.input_residency() != primary.prefill().input_residency()
      )
    let bounded_schedule = compilation.schedule()
    let primary_schedule = primary_compilation.schedule()
    let specialized_pipeline = bounded_schedule.matrix_pipeline() !=
      primary_schedule.matrix_pipeline() ||
      bounded_schedule.sibling_pipeline() != primary_schedule.sibling_pipeline() ||
      bounded_schedule.intermediate_pipeline() !=
      primary_schedule.intermediate_pipeline()
    if !measured_distribution && !specialized_pipeline {
      continue
    }
    // Keep operand allocations at the complete profile envelope, but derive
    // the launch from this bounded executable's domain. In particular a head
    // variant still accepts token rows, not an invented output-row domain.
    let variant_profile = @launch_contract.PagedKernelProfile::new(
      id=profile.id(),
      max_query_rows=Int::min(profile.max_query_rows(), bound),
      max_query_tokens=bound,
      max_page_table_entries=profile.max_page_table_entries(),
    ) catch {
      _ => raise Invalid(CompilerPolicy)
    }
    let dimensions = projection_dimensions(
      variant_profile, parameters, dispatch,
    )
    let variant_symbol = symbol + "_rows" + bound.to_string()
    let source = projection_source(
      variant_symbol,
      parameters,
      bound,
      Int::min(profile.max_query_rows(), bound),
      dispatch,
      compilation,
      false,
      dimensions.shared_memory_bytes() > 0,
      max_registers~,
    )
    projection_append_source_fragment(
      sources,
      @utf8.decode(source) catch {
        _ => raise Invalid(CompilerPolicy)
      },
    )
    variants.push(
      @artifact.KernelRowVariant::new(
        maximum_token_rows=bound,
        function_symbol=variant_symbol,
        dimensions~,
        bounded_work_grid_x=projection_bounded_work_grid_x(
          compilation,
          parameters,
          bound,
          Int::min(profile.max_query_rows(), bound),
          dimensions,
        ),
        companion_dimensions=projection_companion_dimensions(
          parameters, dispatch, variant_profile, compilation,
        ),
      ),
    )
  }
  (sources.to_string(), ReadOnlyArray::from_array(variants))
}

///|
fn projection_row_variant_compilation(
  parameters : ProjectionParameters,
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
  maximum_rows : Int,
  maximum_tokens : Int,
) -> @luna_projection_tile_compiler.ScheduledProjectionTileCompilation raise ProjectionAotError {
  @luna_projection_tile_compiler.compile_projection_tile(
    @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
      dispatch~,
      maximum_rows~,
      maximum_tokens~,
      demand=if parameters.family == LanguageModelHead {
        QueryRowEnds
      } else {
        AllTokenRows
      },
      single_row_numerics=if projection_tensor_core_eligible(
          parameters, dispatch,
        ) ||
        projection_tensor_core_mlp_eligible(parameters, dispatch) {
        AllowDeterministicDotTree
      } else {
        PreserveDotOrder
      },
    ) catch {
      _ => raise Invalid(CompilerPolicy)
    },
  )
}
