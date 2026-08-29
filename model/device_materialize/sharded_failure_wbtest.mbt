///|
test "sharded mutation and read failures close local allocation" {
  let file = wb_sharded_file()
  let prepared = wb_prepare_sharded(file, 0)
  let changed_array = file.to_array()
  let last = changed_array.length() - 1
  changed_array[last] = (changed_array[last].to_int() ^ 1).to_byte()
  let changed = Bytes::from_array(changed_array)
  let mutation_closes = Ref(0)
  try
    load_prepared_sharded_streaming_with_resource(
      prepared,
      changed.length().to_int64(),
      wb_sharded_model(file).content_digest().as_hex(),
      17,
      (destination, destination_offset, position, count) => {
        wb_fill(changed, destination, destination_offset, position, count)
      },
      () => wb_stamp(changed),
      _arena_bytes => 3,
      (_resource, _source, _source_offset, _destination, _count) => (),
      _resource => mutation_closes.val += 1,
    )
  catch {
    FileChangedDuringLoad => ()
    _ => fail("unexpected mutation error")
  } noraise {
    _ => fail("mutated shard source became ready")
  }
  assert_eq(mutation_closes.val, 1)
  let read_closes = Ref(0)
  try
    load_prepared_sharded_streaming_with_resource(
      prepared,
      file.length().to_int64(),
      wb_sharded_model(file).content_digest().as_hex(),
      17,
      (_destination, _destination_offset, _position, _count) => {
        raise FileUnavailable
      },
      () => wb_stamp(file),
      _arena_bytes => 5,
      (_resource, _source, _source_offset, _destination, _count) => (),
      _resource => read_closes.val += 1,
    )
  catch {
    FileUnavailable => ()
    _ => fail("unexpected read error")
  } noraise {
    _ => fail("failed shard read became ready")
  }
  assert_eq(read_closes.val, 1)
}

///|
test "sharded copy failure retains cleanup authority when close fails" {
  let file = wb_sharded_file()
  let prepared = wb_prepare_sharded(file, 1)
  let copies = Ref(0)
  let result = load_prepared_sharded_streaming_with_resource(
    prepared,
    file.length().to_int64(),
    wb_sharded_model(file).content_digest().as_hex(),
    17,
    (destination, destination_offset, position, count) => {
      wb_fill(file, destination, destination_offset, position, count)
    },
    () => wb_stamp(file),
    _arena_bytes => 11,
    (_resource, _source, _source_offset, _destination, _count) => {
      copies.val += 1
      raise DriverFailure(CopyToDevice)
    },
    _resource => raise Busy(AllocationClose),
  )
  guard result is ShardedResourceCleanupRequired(failed) else {
    fail("double shard failure discarded cleanup authority")
  }
  assert_eq(copies.val, 1)
  assert_eq(failed.resource, 11)
  assert_eq(failed.cause, DeviceFailed(DriverFailure(CopyToDevice)))
  assert_eq(failed.stage, Transfer)
  assert_eq(failed.cleanup_failure, Busy(AllocationClose))
  assert_eq(failed.rank_plan, prepared.rank_plan)
}

///|
test "sharded preflight authenticates complete file before allocation" {
  let file = wb_sharded_file()
  let model = wb_sharded_model(file)
  let changed_array = file.to_array()
  let last = changed_array.length() - 1
  changed_array[last] = (changed_array[last].to_int() ^ 1).to_byte()
  let changed = Bytes::from_array(changed_array)
  try
    prepare_sharded_streaming_load(
      model,
      @llama.build(model),
      0,
      2,
      64L,
      4096L,
      wb_stamp(changed),
      changed.length().to_int64(),
      wb_stream_limits(),
      (destination, destination_offset, position, count) => {
        wb_fill(changed, destination, destination_offset, position, count)
      },
      () => wb_stamp(changed),
    )
  catch {
    ContentDigestMismatch => ()
    _ => fail("unexpected preflight digest error")
  } noraise {
    _ => fail("changed complete file passed shard preflight")
  }
}

///|
test "sharded source close failure preserves retryable allocation ownership" {
  let file = wb_sharded_file()
  let rank_plan = wb_sharded_rank_plan(file, 0)
  let ready : ShardedResourcePreparation[Int] = ShardedResourceReady({
    resource: 13,
    rank_plan,
  })
  let closes = Ref(0)
  let result = finish_sharded_weight_file_load(
    Ok(ready),
    () => raise FileCloseFailed,
    _resource => {
      closes.val += 1
      raise Busy(AllocationClose)
    },
  )
  guard result is ShardedResourceCleanupRequired(failed) else {
    fail("dual source close failure lost shard allocation")
  }
  assert_eq(closes.val, 1)
  assert_eq(failed.cause, FileCloseFailed)
  assert_eq(failed.stage, SourceClose)
  assert_eq(failed.cleanup_failure, Busy(AllocationClose))
  assert_eq(failed.rank_plan, rank_plan)
}

///|
test "sharded source close preserves existing cleanup authority" {
  let file = wb_sharded_file()
  let retained : ShardedResourcePreparation[Int] = ShardedResourceCleanupRequired({
      resource: 17,
      rank_plan: wb_sharded_rank_plan(file, 1),
      cause: DeviceFailed(DriverFailure(CopyToDevice)),
      stage: Transfer,
      cleanup_failure: Busy(AllocationClose),
    },
  )
  let closes = Ref(0)
  let result = finish_sharded_weight_file_load(
    Ok(retained),
    () => raise FileCloseFailed,
    _resource => closes.val += 1,
  )
  guard result is ShardedResourceCleanupRequired(failed) else {
    fail("existing shard cleanup authority was discarded")
  }
  assert_eq(closes.val, 0)
  assert_eq(failed.resource, 17)
  assert_eq(failed.stage, Transfer)
}
