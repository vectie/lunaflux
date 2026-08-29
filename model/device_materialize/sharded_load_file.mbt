///|
fn device_sharded_weight_preparation(
  loaded : ShardedResourcePreparation[@device.Allocation],
) -> DeviceShardedWeightPreparation {
  match loaded {
    ShardedResourceReady(ready) =>
      ShardedWeightsReady({
        allocation: ready.resource,
        rank_plan: ready.rank_plan,
      })
    ShardedResourceCleanupRequired(failed) =>
      ShardedWeightsCleanupRequired({
        allocation: failed.resource,
        rank_plan: failed.rank_plan,
        cause: failed.cause,
        stage: failed.stage,
        cleanup_failure: failed.cleanup_failure,
        open: true,
      })
  }
}

///|
fn[T] finish_sharded_weight_file_load(
  outcome : Result[ShardedResourcePreparation[T], DeviceWeightError],
  close_file : () -> Unit raise DeviceWeightError,
  close_resource : (T) -> Unit raise @device.DeviceError,
) -> ShardedResourcePreparation[T] raise DeviceWeightError {
  match outcome {
    Err(primary) => {
      close_file() catch {
        _ => ()
      }
      raise primary
    }
    Ok(loaded) =>
      try close_file() catch {
        close_failure =>
          match loaded {
            ShardedResourceCleanupRequired(_) => loaded
            ShardedResourceReady(ready) =>
              sharded_resource_failure(
                ready.resource,
                ready.rank_plan,
                close_failure,
                SourceClose,
                close_resource,
              )
          }
      } noraise {
        _ => loaded
      }
  }
}

///|
fn with_sharded_weight_file_load(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  body : (@approved_fs.ApprovedFile) -> ShardedResourcePreparation[
    @device.Allocation,
  ] raise DeviceWeightError,
) -> DeviceShardedWeightPreparation raise DeviceWeightError {
  let file = open_weight_file(root, locator)
  let outcome : Result[
    ShardedResourcePreparation[@device.Allocation],
    DeviceWeightError,
  ] = Ok(body(file)) catch { error => Err(error) }
  device_sharded_weight_preparation(
    finish_sharded_weight_file_load(outcome, () => close_weight_file(file), allocation => {
      allocation.close()
    }),
  )
}

///|
/// Authenticates a complete approved safetensors descendant and its exact
/// dense-Llama rank plan without opening a device allocation. Topology and
/// arena limits are inputs; source-bearing transfer recipes are derived only
/// from the freshly authenticated file bindings.
pub fn inspect_sharded_file(
  root : @approved_fs.ApprovedRoot,
  model : @spec.LlamaModelMetadata,
  semantic_plan : @plan.ModelPlan,
  locator : @approved_fs.ApprovedRelativeLocator,
  rank~ : Int,
  world_size~ : Int,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
  limits~ : DeviceWeightFileLimits,
) -> DeviceShardedWeightFileInspection raise DeviceWeightError {
  with_weight_file(root, locator, file => {
    let initial_stamp = weight_file_stamp(file)
    let prepared = prepare_sharded_streaming_load(
      model,
      semantic_plan,
      rank,
      world_size,
      alignment_bytes,
      max_arena_bytes,
      initial_stamp,
      initial_stamp.byte_count(),
      limits,
      (destination, destination_offset, position, byte_count) => {
        read_weight_range(
          file, destination, destination_offset, position, byte_count,
        )
      },
      () => weight_file_stamp(file),
    )
    {
      model,
      semantic_plan,
      locator,
      limits,
      max_arena_bytes,
      payload_offset_bytes: prepared.payload_offset_bytes,
      rank_plan: prepared.rank_plan,
    }
  })
}

///|
/// Reauthenticates this exact source-bound rank inspection without exposing
/// model-family metadata, locators, parser limits, or transfer recipes to the
/// engine owner. The returned inspection is freshly derived from the current
/// approved file and the immutable private planning evidence.
pub fn DeviceShardedWeightFileInspection::reauthenticate(
  self : DeviceShardedWeightFileInspection,
  root : @approved_fs.ApprovedRoot,
) -> DeviceShardedWeightFileInspection raise DeviceWeightError {
  let topology = self.rank_plan.topology()
  inspect_sharded_file(
    root,
    self.model,
    self.semantic_plan,
    self.locator,
    rank=topology.rank(),
    world_size=topology.world_size(),
    alignment_bytes=self.rank_plan.alignment_bytes(),
    max_arena_bytes=self.max_arena_bytes,
    limits=self.limits,
  )
}

///|
/// Reauthenticates the complete inspected file before allocating exactly one
/// local rank arena, then hashes it again while streaming only authenticated
/// scalar shard ranges into their final compact destinations.
pub fn load_inspected_sharded_file(
  context : @device.Context,
  root : @approved_fs.ApprovedRoot,
  inspection : DeviceShardedWeightFileInspection,
) -> DeviceShardedWeightPreparation raise DeviceWeightError {
  with_sharded_weight_file_load(root, inspection.locator, file => {
    let initial_stamp = weight_file_stamp(file)
    let file_bytes = initial_stamp.byte_count()
    let topology = inspection.rank_plan.topology()
    let prepared = prepare_sharded_streaming_load(
      inspection.model,
      inspection.semantic_plan,
      topology.rank(),
      topology.world_size(),
      inspection.rank_plan.alignment_bytes(),
      inspection.max_arena_bytes,
      initial_stamp,
      file_bytes,
      inspection.limits,
      (destination, destination_offset, position, byte_count) => {
        read_weight_range(
          file, destination, destination_offset, position, byte_count,
        )
      },
      () => weight_file_stamp(file),
    )
    if prepared.payload_offset_bytes != inspection.payload_offset_bytes ||
      prepared.rank_plan != inspection.rank_plan {
      raise FileChangedDuringLoad
    }
    load_prepared_sharded_streaming_with_resource(
      prepared,
      file_bytes,
      inspection.content_digest().as_hex(),
      inspection.limits.transfer_chunk_bytes(),
      (destination, destination_offset, position, byte_count) => {
        read_weight_range(
          file, destination, destination_offset, position, byte_count,
        )
      },
      () => weight_file_stamp(file),
      arena_bytes => context.allocate(arena_bytes),
      (allocation, source, source_offset, destination_offset, byte_count) => {
        context.copy_from_fixed_host(
          allocation,
          source,
          source_offset.to_int64(),
          destination_offset,
          byte_count,
        )
      },
      allocation => allocation.close(),
    )
  })
}
