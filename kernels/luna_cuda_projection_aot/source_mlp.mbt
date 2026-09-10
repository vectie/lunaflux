///|
fn projection_source(
  symbol : String,
  parameters : ProjectionParameters,
  maximum_tokens : Int,
  maximum_rows : Int,
  dispatch : @luna_projection_strategy.ProjectionDispatchPlan,
  tile_compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  embed_greedy_sampling : Bool,
  use_shared_mlp : Bool,
  max_registers~ : Int,
) -> Bytes raise ProjectionAotError {
  match parameters.family {
    QkvProjection =>
      if projection_tensor_core_eligible(parameters, dispatch) {
        projection_tensor_core_direct_qkv_source(
          symbol,
          parameters,
          maximum_tokens,
          tile_compilation,
          groups=tile_compilation.schedule().output_tiles_per_workgroup(),
        )
      } else {
        projection_scalar_map_source(
          symbol, parameters, maximum_tokens, maximum_rows, tile_compilation,
        )
      }
    DenseProjection =>
      if projection_tensor_core_eligible(parameters, dispatch) {
        projection_tensor_core_direct_dense_source(
          symbol, parameters, maximum_tokens, maximum_rows, tile_compilation,
        )
      } else {
        projection_scalar_map_source(
          symbol, parameters, maximum_tokens, maximum_rows, tile_compilation,
        )
      }
    LanguageModelHead => {
      let base = if projection_tensor_core_eligible(parameters, dispatch) {
        projection_tensor_core_direct_dense_source(
          symbol, parameters, maximum_tokens, maximum_rows, tile_compilation,
        )
      } else {
        projection_scalar_map_source(
          symbol, parameters, maximum_tokens, maximum_rows, tile_compilation,
        )
      }
      if !embed_greedy_sampling {
        return base
      }
      let source = @utf8.decode(base) catch { _ => raise Invalid(Source) }
      let reducer = @luna_cuda_sampling_aot.embedded_greedy_sampling_cuda_source(
        parameters.output_width,
        maximum_rows,
      ) catch {
        _ => raise Invalid(Source)
      }
      @utf8.encode(source + reducer)
    }
    GatedMlp =>
      if projection_tensor_core_mlp_eligible(parameters, dispatch) &&
        tile_compilation.schedule().reuse_intermediate_tile() {
        projection_matrix_mlp_source(
          symbol,
          parameters,
          maximum_tokens,
          tile_compilation,
          max_registers~,
        )
      } else if use_shared_mlp {
        projection_scalar_mlp_source(symbol, parameters, maximum_tokens, true)
      } else {
        projection_scalar_mlp_source(symbol, parameters, maximum_tokens, false)
      }
  }
}
