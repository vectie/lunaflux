///|
// All calls are private and precede finalize. Length-prefix strings and encode
// scalars as unsigned LE64; never use Debug rendering as a canonical encoding.
fn put_int(hash : @canonical_sha256.CanonicalSha256, value : Int64) -> Unit {
  ignore(hash.update_u64_le(value.reinterpret_as_uint64()))
}

///|
fn put_text(hash : @canonical_sha256.CanonicalSha256, value : String) -> Unit {
  let bytes = @utf8.encode(value)
  put_int(hash, bytes.length().to_int64())
  ignore(hash.update_bytes(bytes))
}

///|
fn digest_plan(
  model : @plan.ModelPlan,
  topology : @network_topology.TwoNodeTopology,
  ranks : ArrayView[@tensor_parallel_plan.TensorParallelRankPlan],
  budgets : ArrayView[RankMemoryBudget],
  kv_layout : @device_layout.DeviceKvLayout,
) -> TwoNodePlanDigest raise TwoNodePlanError {
  let hash = @canonical_sha256.CanonicalSha256::new()
  put_text(hash, "lunaflux.two-node-plan.v1")
  put_text(hash, model.identity().content().as_hex())
  put_text(hash, model.identity().plan().as_hex())
  put_int(hash, kv_layout.tokens_per_page().to_int64())
  put_int(hash, kv_layout.total_page_count().to_int64())
  put_int(hash, kv_layout.stable_layout_version().to_int64())
  put_int(hash, kv_layout.segment_alignment_bytes())
  put_int(hash, kv_layout.page_logical_bytes())
  put_int(hash, kv_layout.page_stride_bytes())
  put_int(hash, kv_layout.component_stride_bytes())
  put_int(hash, kv_layout.layer_stride_bytes())
  put_int(hash, kv_layout.arena_bytes())
  put_int(hash, 2L)
  for rank, plan in ranks {
    guard topology.rank(rank) is Some(node) else {
      raise Rejected(rule=RankOrder, rank~)
    }
    let target = node.target()
    let budget = budgets[rank]
    put_text(hash, node.node().label())
    put_text(hash, node.rdma_interface())
    for
      value in [
        rank.to_int64(),
        node.device_ordinal().to_int64(),
        target.compute_major().to_int64(),
        target.compute_minor().to_int64(),
        if target.supports_bf16() {
          1L
        } else {
          0L
        },
        if target.supports_cublas_lt() {
          1L
        } else {
          0L
        },
        node.physical_memory_bytes(),
        budget.system_reserved_bytes,
        budget.runtime_ceiling_bytes,
        budget.activation_workspace_bytes,
        budget.collective_bytes,
        budget.kv_bytes,
        plan.alignment_bytes(),
        plan.arena_bytes(),
        plan.materialized_bytes(),
        plan.tensor_count().to_int64(),
      ] {
      put_int(hash, value)
    }
    for index in 0..<plan.tensor_count() {
      let reference = @plan.TensorRef::new(index) catch {
        _ => raise Rejected(rule=TensorPartition, rank~)
      }
      guard plan.tensor(reference) is Some(tensor) else {
        raise Rejected(rule=TensorPartition, rank~)
      }
      let extent = tensor.extent()
      let region = tensor.region()
      let recipe = tensor.transfer_recipe()
      let first = recipe.segment(0L) catch {
        _ => raise Rejected(rule=TensorPartition, rank~)
      }
      let last = recipe.segment(recipe.segment_count() - 1L) catch {
        _ => raise Rejected(rule=TensorPartition, rank~)
      }
      for
        value in [
          index.to_int64(),
          match tensor.placement() {
            Replicated => 0L
            ShardRows => 1L
            ShardColumns => 2L
          },
          extent.full_rows(),
          extent.full_columns(),
          extent.row_start(),
          extent.row_count(),
          extent.column_start(),
          extent.column_count(),
          extent.byte_length(),
          region.offset_bytes(),
          region.byte_length(),
          region.alignment_bytes(),
          recipe.segment_count(),
          recipe.segment_byte_length(),
          first.source_payload_offset_bytes(),
          first.destination_offset_bytes(),
          last.source_payload_offset_bytes(),
          last.destination_offset_bytes(),
        ] {
        put_int(hash, value)
      }
    }
    put_int(hash, plan.operation_count().to_int64())
    for semantic in model.operations() {
      guard plan.operation(semantic.id()) is Some(operation) else {
        raise Rejected(rule=OperationMismatch, rank~)
      }
      // Semantic kind and capability are already bound by the model digest;
      // rank-specific physical dimensions remain explicit here.
      for
        value in [
          operation.operation().as_int(),
          operation.input_width(),
          operation.output_width(),
          operation.inner_width(),
          operation.query_head_count(),
          operation.kv_head_count(),
          operation.head_dimension(),
        ] {
        put_int(hash, value.to_int64())
      }
    }
    put_int(hash, plan.collective_count().to_int64())
    for index in 0..<plan.collective_count() {
      guard plan.collective(index) is Some(site) else {
        raise Rejected(rule=CollectiveMismatch, rank~)
      }
      for
        value in [
          site.sequence().as_int(),
          site.operation().as_int(),
          match site.kind() {
            SumAllReduce => 0
            LastAxisAllGather => 1
          },
          site.full_width(),
          site.local_width(),
        ] {
        put_int(hash, value.to_int64())
      }
    }
  }
  TwoNodePlanDigest(hash.finalize_hex())
}
