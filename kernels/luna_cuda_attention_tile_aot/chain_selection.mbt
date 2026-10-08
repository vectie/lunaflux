///|
/// A complete partial-plus-merge timing, never the sum of isolated medians.
/// Scope names the device, toolchain/flags, exact workload and launch/cache
/// context. The exporter supplies it; this pure value performs no I/O.
pub struct PartitionedAttentionChainObservation {
  priv frontier : CudaAttentionTileAotDigest
  priv compilation : CudaAttentionTileAotDigest
  priv scope : String
  priv cost : @fusion_regions.Cost
}

///|
pub fn PartitionedAttentionChainObservation::new(
  frontier~ : CudaAttentionTileAotDigest,
  compilation~ : CudaAttentionTileAotDigest,
  scope~ : String,
  elapsed_ns~ : Int64,
  samples~ : Int,
) -> PartitionedAttentionChainObservation raise CudaAttentionTileAotError {
  if scope.is_empty() {
    raise InvalidCudaAttentionTileAot(InvalidRequest)
  }
  let cost = @fusion_regions.Cost::measured(nanoseconds=elapsed_ns, samples~) catch {
    _ => raise InvalidCudaAttentionTileAot(InvalidRequest)
  }
  { frontier, compilation, scope, cost, }
}

///|
/// Select a measured complete executable chain through the common
/// continuation-aware frontier. No unmeasured or foreign observation fallback.
/// The caller's legal candidate domain supplies the numerical permission;
/// measurement is not permission to relax arithmetic or resource constraints.
pub fn CudaPartitionedAttentionTileAotFrontier::select_measured_chain(
  self : CudaPartitionedAttentionTileAotFrontier,
  scope~ : String,
  observations : ArrayView[PartitionedAttentionChainObservation],
  maximum_workspace_bytes~ : Int64,
  maximum_shared_memory_bytes~ : Int64,
) -> CudaPartitionedAttentionTileAotCompilation raise CudaAttentionTileAotError {
  if scope.is_empty() ||
    observations.is_empty() ||
    observations.length() > self.variants.length() {
    raise InvalidCudaAttentionTileAot(InvalidRequest)
  }
  let graph = @fusion_regions.Graph::new([
    @fusion_regions.Node::new([], observable=true, effect=Pure),
  ]) catch {
    _ => raise InvalidCudaAttentionTileAot(InvalidRequest)
  }
  let partition = @fusion_regions.Partition::new(graph, [
    @fusion_regions.Region::new(graph, start=0, count=1, [0]) catch {
      _ => raise InvalidCudaAttentionTileAot(InvalidRequest)
    },
  ]) catch {
    _ => raise InvalidCudaAttentionTileAot(InvalidRequest)
  }
  let alternatives : Array[@fusion_regions.PartitionAlternative] = []
  for index, observation in observations {
    if observation.frontier != self.digest ||
      observation.scope != scope ||
      observations[:index].any(prior => {
        prior.compilation == observation.compilation
      }) {
      raise InvalidCudaAttentionTileAot(InvalidRequest)
    }
    let matches = self.variants
      .mapi((id, variant) => (id, variant.compilation))
      .filter(pair => pair.1.digest() == observation.compilation)
    guard matches is [(id, compilation)] else {
      raise InvalidCudaAttentionTileAot(InvalidRequest)
    }
    alternatives.push(
      @fusion_regions.PartitionAlternative::new(
        id~,
        partition,
        observation.cost,
        workspace_bytes=compilation.source().workspace().total_bytes(),
        // Peak workgroup-local storage, not a register-count latency proxy.
        local_bytes=compilation.lowering().launch().3,
      ) catch {
        _ => raise InvalidCudaAttentionTileAot(InvalidRequest)
      },
    )
  }
  let selected = @fusion_regions.select_partition(
    alternatives,
    maximum_workspace_bytes~,
    maximum_local_bytes=maximum_shared_memory_bytes,
  ) catch {
    _ => raise InvalidCudaAttentionTileAot(InvalidRequest)
  }
  self.variants[selected].compilation
}
