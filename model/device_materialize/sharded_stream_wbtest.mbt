///|
test "sharded stream copies row column and replicated ranges directly" {
  let file = wb_sharded_file()
  let dense_layout = plan_layout(
    wb_sharded_bindings(file, wb_sharded_model(file)),
    alignment_bytes=64L,
    max_arena_bytes=4096L,
  )
  for rank in 0..<2 {
    let prepared = wb_prepare_sharded(file, rank)
    assert_eq(prepared.rank_plan, wb_sharded_rank_plan(file, rank))
    let arena = FixedArray::make(
      prepared.rank_plan.arena_bytes().to_int(),
      b'\x00',
    )
    let opened = Ref(0L)
    let result = load_prepared_sharded_streaming_with_resource(
      prepared,
      file.length().to_int64(),
      wb_sharded_model(file).content_digest().as_hex(),
      17,
      (destination, destination_offset, position, count) => {
        wb_fill(file, destination, destination_offset, position, count)
      },
      () => wb_stamp(file),
      arena_bytes => {
        opened.val = arena_bytes
        7
      },
      (_resource, source, source_offset, destination_offset, count) => {
        wb_copy_sharded(arena, source, source_offset, destination_offset, count)
      },
      _resource => (),
    )
    guard result is ShardedResourceReady(ready) else {
      fail("successful shard retained cleanup authority")
    }
    assert_eq(opened.val, prepared.rank_plan.arena_bytes())
    assert_true(opened.val < dense_layout.arena_bytes())
    assert_eq(ready.rank_plan, prepared.rank_plan)
    for reference in 0..<prepared.rank_plan.tensor_count() {
      let tensor_reference = try! @plan.TensorRef::new(reference)
      let placement = prepared.rank_plan.tensor(tensor_reference).unwrap()
      let recipe = placement.transfer_recipe()
      for segment_index in 0L..<recipe.segment_count() {
        let segment = try! recipe.segment(segment_index)
        for byte in 0L..<segment.byte_length() {
          assert_eq(
            arena[(segment.destination_offset_bytes() + byte).to_int()],
            file[(prepared.payload_offset_bytes +
            segment.source_payload_offset_bytes() +
            byte).to_int()],
          )
        }
      }
    }
  }
}

///|
fn wb_materialize_sharded_inspection(
  file : Bytes,
  inspection : DeviceShardedWeightFileInspection,
) -> @spec.ModelIdentity raise {
  let topology = inspection.rank_plan.topology()
  let prepared = prepare_sharded_streaming_load(
    inspection.model,
    inspection.semantic_plan,
    topology.rank(),
    topology.world_size(),
    inspection.rank_plan.alignment_bytes(),
    inspection.max_arena_bytes,
    wb_stamp(file),
    file.length().to_int64(),
    inspection.limits,
    (destination, destination_offset, position, count) => {
      wb_fill(file, destination, destination_offset, position, count)
    },
    () => wb_stamp(file),
  )
  assert_eq(prepared.rank_plan, inspection.rank_plan)
  let arena = FixedArray::make(
    prepared.rank_plan.arena_bytes().to_int(),
    b'\x00',
  )
  let outcome = load_prepared_sharded_streaming_with_resource(
    prepared,
    file.length().to_int64(),
    inspection.content_digest().as_hex(),
    inspection.limits.transfer_chunk_bytes(),
    (destination, destination_offset, position, count) => {
      wb_fill(file, destination, destination_offset, position, count)
    },
    () => wb_stamp(file),
    _arena_bytes => 1,
    (_resource, source, source_offset, destination_offset, count) => {
      wb_copy_sharded(arena, source, source_offset, destination_offset, count)
    },
    _resource => (),
  )
  guard outcome is ShardedResourceReady(ready) else {
    fail("authenticated sharded inspection did not materialize")
  }
  ready.rank_plan.identity()
}

///|
async test "public sharded inspection and materialization preserve dense and paged plan identity" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-sharded-inspection")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let file = wb_sharded_file()
    @fs.write_file(
      "\{directory}/weights.safetensors",
      file,
      create_mode=CreateNew,
    )
    let model = wb_sharded_model(file)
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let semantic_plans = [@llama.build(model), @llama.build_paged(model)]
    for semantic_plan in semantic_plans {
      let inspection = inspect_sharded_file(
        root,
        model,
        semantic_plan,
        wb_weight_locator("weights.safetensors"),
        rank=1,
        world_size=2,
        alignment_bytes=64L,
        max_arena_bytes=4096L,
        limits=wb_stream_limits(transfer_chunk_bytes=17),
      )
      assert_eq(inspection.content_digest(), model.content_digest())
      assert_eq(inspection.rank_plan().identity(), semantic_plan.identity())
      let refreshed = inspection.reauthenticate(root)
      assert_eq(refreshed.content_digest(), inspection.content_digest())
      assert_eq(refreshed.rank_plan(), inspection.rank_plan())
      assert_eq(
        wb_materialize_sharded_inspection(file, refreshed),
        semantic_plan.identity(),
      )
    }
    root.close()
  }
}

///|
async test "sharded inspection rejects foreign content and incompatible same-content graph" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-sharded-plan-identity")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let file = wb_sharded_file()
    @fs.write_file(
      "\{directory}/weights.safetensors",
      file,
      create_mode=CreateNew,
    )
    let model = wb_sharded_model(file)
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let inspect = semantic_plan => {
      inspect_sharded_file(
        root,
        model,
        semantic_plan,
        wb_weight_locator("weights.safetensors"),
        rank=1,
        world_size=2,
        alignment_bytes=64L,
        max_arena_bytes=4096L,
        limits=wb_stream_limits(transfer_chunk_bytes=17),
      )
    }
    let foreign_model = @spec.LlamaModelMetadata::from_verified_content(
      model.spec(),
      @spec.ContentDigest::from_sha256(
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
      ),
    )
    try ignore(inspect(@llama.build(foreign_model))) catch {
      InvalidLlamaWeights(ModelContentMismatch) => ()
      _ => fail("unexpected foreign-content sharded inspection failure")
    } noraise {
      _ => fail("foreign model content was admitted for sharded inspection")
    }
    let incompatible_model = @spec.LlamaModelMetadata::from_verified_content(
      @spec.LlamaModelSpec::new(
        vocabulary_size=8,
        hidden_size=4,
        intermediate_size=8,
        layer_count=1,
        attention_head_count=2,
        kv_head_count=2,
        context_length=16,
        rope_theta=10000.0,
        weight_dtype=BFloat16,
        kv_dtype=BFloat16,
      ),
      model.content_digest(),
    )
    try ignore(inspect(@llama.build(incompatible_model))) catch {
      InvalidLlamaWeights(ModelPlanMismatch) => ()
      _ => fail("unexpected incompatible-graph sharded inspection failure")
    } noraise {
      _ => fail("same-content incompatible graph was admitted")
    }
    root.close()
  }
}

///|
fn wb_assert_all_rank_shard_reconstruction(file : Bytes) -> Unit raise {
  let prepared = [wb_prepare_sharded(file, 0), wb_prepare_sharded(file, 1)]
  let arenas = [
    FixedArray::make(prepared[0].rank_plan.arena_bytes().to_int(), b'\x00'),
    FixedArray::make(prepared[1].rank_plan.arena_bytes().to_int(), b'\x00'),
  ]
  for rank in 0..<2 {
    ignore(
      load_prepared_sharded_streaming_with_resource(
        prepared[rank],
        file.length().to_int64(),
        wb_sharded_model(file).content_digest().as_hex(),
        17,
        (destination, destination_offset, position, count) => {
          wb_fill(file, destination, destination_offset, position, count)
        },
        () => wb_stamp(file),
        _arena_bytes => rank,
        (_resource, source, source_offset, destination_offset, count) => {
          wb_copy_sharded(
            arenas[rank],
            source,
            source_offset,
            destination_offset,
            count,
          )
        },
        _resource => (),
      ),
    )
  }
  let bindings = wb_sharded_bindings(file, wb_sharded_model(file))
  for reference in 0..<bindings.length() {
    let tensor_reference = try! @plan.TensorRef::new(reference)
    let source = bindings.binding(tensor_reference).unwrap().tensor()
    let first = prepared[0].rank_plan.tensor(tensor_reference).unwrap()
    let reconstructed = FixedArray::make(source.byte_length().to_int(), b'\x00')
    for rank in 0..<2 {
      let placement = prepared[rank].rank_plan.tensor(tensor_reference).unwrap()
      let extent = placement.extent()
      if placement.placement() == Replicated && rank == 1 {
        continue
      }
      for row in 0L..<extent.row_count() {
        for column in 0L..<extent.column_count() {
          let global = (extent.row_start() + row) * extent.full_columns() +
            extent.column_start() +
            column
          let shard = row * extent.column_count() + column
          for byte in 0L..<2L {
            reconstructed[(global * 2L + byte).to_int()] = arenas[rank][(placement
              .region()
              .offset_bytes() +
              shard * 2L +
              byte).to_int()]
          }
        }
      }
    }
    assert_eq(
      first.extent().full_rows() * first.extent().full_columns() * 2L,
      source.byte_length(),
    )
    for byte in 0..<reconstructed.length() {
      assert_eq(
        reconstructed[byte],
        file[(prepared[0].payload_offset_bytes +
        source.data_start() +
        byte.to_int64()).to_int()],
      )
    }
  }
}

///|
test "all rank shard arenas reconstruct complete tensors" {
  wb_assert_all_rank_shard_reconstruction(wb_sharded_file())
}

///|
test "physically reordered payload tensors stream and reconstruct by source" {
  let file = wb_sharded_reordered_file()
  let prepared = wb_prepare_sharded(file, 0)
  assert_eq(prepared.source_order[0], 11)
  assert_eq(prepared.source_order[prepared.source_order.length() - 1], 0)
  wb_assert_all_rank_shard_reconstruction(file)
}
