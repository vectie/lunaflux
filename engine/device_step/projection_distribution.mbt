///|
/// Decode the private CUDA matrix lowering's work distribution at startup.
/// Model plans and the compiler strategy retain backend-neutral tile counts.
fn paged_projection_workgroups(
  dimensions : @launch_contract.AotLaunchDimensions,
) -> Int raise DeviceStepError {
  let threads = dimensions.block_x()
  if threads < 32 || threads > 512 || (threads & (threads - 1)) != 0 {
    raise InvalidExecutor(GraphBucket)
  }
  threads / 32
}

///|
fn prepare_projection_bucket_launches(
  blueprint : PagedKvExecutionBlueprint,
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  launches : FixedArray[PagedBucketStepLaunch],
) -> Unit raise DeviceStepError {
  let mlp_occurrences : Array[Int] = []
  for index, prepared in steps {
    let step = paged_blueprint_step(blueprint, prepared.operation_id)
    let source = step.dimensions()
    match step.operation_kind() {
      QkvProjection | OutputProjection => {
        let output_width = paged_projection_output_width(blueprint, step)
        launches[index] = if source.block_x() >= 32 &&
          source.block_x() <= 512 &&
          (source.block_x() & (source.block_x() - 1)) == 0 &&
          source.shared_memory_bytes() == source.block_x() * 32 {
          ProjectionTensorCoreGridX(source, output_width)
        } else {
          ProjectionScalarGridX(source, output_width)
        }
      }
      LanguageModelHead => ()
      GatedMlp => {
        let workspace = step
          .operands()
          .any(operand => operand.role() == Workspace)
        if workspace {
          let mut occurrence = 0
          for operation in mlp_occurrences {
            if operation == prepared.operation_id.as_int() {
              occurrence += 1
            }
          }
          if occurrence > 1 {
            raise executor_operation_error(prepared.operation_id, GraphBucket)
          }
          mlp_occurrences.push(prepared.operation_id.as_int())
          launches[index] = if occurrence == 0 {
            ProjectionMlpUpGridX(
              source,
              paged_projection_intermediate_width(blueprint, step),
            )
          } else {
            ProjectionMlpDownGridX(
              step.companion_dimensions.unwrap_or(source),
              paged_projection_output_width(blueprint, step),
            )
          }
        } else if source.shared_memory_bytes() > 0 {
          launches[index] = QueryTokenCappedGridX(source)
        } else {
          launches[index] = ProjectionScalarGridX(
            source,
            paged_projection_output_width(blueprint, step),
          )
        }
      }
      _ => ()
    }
  }
}

///|
fn paged_ceil_div(value : Int, divisor : Int) -> Int raise DeviceStepError {
  if value <= 0 || divisor <= 0 || value > 2147483647 - (divisor - 1) {
    raise InvalidExecutor(GraphBucket)
  }
  (value + divisor - 1) / divisor
}

///|
fn paged_projection_bucket_grid_x(
  rule : PagedBucketStepLaunch,
  query_token_bound : Int,
) -> Int raise DeviceStepError {
  if query_token_bound <= 0 {
    raise InvalidExecutor(GraphBucket)
  }
  match rule {
    FixedBucketLaunch => raise InvalidExecutor(GraphBucket)
    QueryTokenCappedGridX(source) =>
      Int::min(source.grid_x(), query_token_bound)
    QueryTileCappedGridX(source, query_tile) =>
      Int::min(source.grid_x(), paged_ceil_div(query_token_bound, query_tile))
    ProjectionTensorCoreGridX(source, output_width) => {
      let groups = paged_projection_workgroups(source)
      let grid = if query_token_bound == 1 {
        paged_ceil_div(output_width, groups)
      } else {
        paged_ceil_div(output_width, groups * 16) *
        paged_ceil_div(query_token_bound, 16)
      }
      Int::min(source.grid_x(), grid)
    }
    ProjectionScalarGridX(source, output_width) => {
      if query_token_bound > 2147483647 / output_width {
        raise InvalidExecutor(GraphBucket)
      }
      Int::min(
        source.grid_x(),
        paged_ceil_div(query_token_bound * output_width, source.block_x()),
      )
    }
    ProjectionMlpUpGridX(source, intermediate_width) => {
      let groups = paged_projection_workgroups(source)
      let grid = if query_token_bound == 1 {
        paged_ceil_div(intermediate_width, groups)
      } else {
        paged_ceil_div(query_token_bound, 16) *
        paged_ceil_div(intermediate_width, groups * 16)
      }
      Int::min(source.grid_x(), grid)
    }
    ProjectionMlpDownGridX(source, output_width) => {
      let groups = paged_projection_workgroups(source)
      let grid = if query_token_bound == 1 {
        paged_ceil_div(output_width, groups)
      } else {
        paged_ceil_div(query_token_bound, 16) *
        paged_ceil_div(output_width, Int::min(4, groups) * 16)
      }
      Int::min(source.grid_x(), grid)
    }
  }
}
