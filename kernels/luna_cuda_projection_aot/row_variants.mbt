///|
/// Compile measured per-bucket strategies into the same immutable AOT module.
/// Identical fallback strategies need no duplicate code. These variants share
/// the full allocation layout but accept only their declared token-row domain.
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
    if selected.source() != OfflineAutotune ||
      strategy.kernel_class() != MatrixTile ||
      strategy.kernel_class() != primary.prefill().kernel_class() ||
      (
        strategy.output_tiles_per_workgroup() ==
        primary.prefill().output_tiles_per_workgroup() &&
        strategy.input_residency() == primary.prefill().input_residency()
      ) {
      continue
    }
    let dispatch = runtime.dispatch_plan(bound) catch {
      _ => raise Invalid(CompilerPolicy)
    }
    let request = @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
      dispatch~,
      maximum_rows=Int::min(profile.max_query_rows(), bound),
      maximum_tokens=bound,
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
    }
    let compilation = @luna_projection_tile_compiler.compile_projection_tile(
      request,
    )
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
        companion_dimensions=projection_companion_dimensions(
          parameters, dispatch, variant_profile, compilation,
        ),
      ),
    )
  }
  (sources.to_string(), ReadOnlyArray::from_array(variants))
}
