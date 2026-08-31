///|
fn broad_serving_combined_frames(
  first : FixedArray[Byte],
  second : FixedArray[Byte],
) -> FixedArray[Byte] {
  let combined = FixedArray::make(first.length() + second.length(), b'\x00')
  for index in 0..<first.length() {
    combined[index] = first[index]
  }
  for index in 0..<second.length() {
    combined[first.length() + index] = second[index]
  }
  combined
}

///|
fn broad_serving_three_frames(
  first : FixedArray[Byte],
  second : FixedArray[Byte],
  third : FixedArray[Byte],
) -> FixedArray[Byte] {
  broad_serving_combined_frames(
    broad_serving_combined_frames(first, second),
    third,
  )
}

///|
async fn connect_broad_serving_client(
  owner : RuntimeInstanceOwner,
  timeout_millis : Int,
) -> @socket.Tcp {
  connect_spawned_serving_client(owner, timeout_millis) catch {
    _ => raise broad_serving_error(BroadServingConnect)
  }
}

///|
async fn accept_broad_serving_client(
  owner : RuntimeInstanceOwner,
  turn_limit : Int,
) -> Unit {
  accept_spawned_serving_client(owner, turn_limit) catch {
    RuntimeSpawnedServingValidationFailed(SpawnedServingHealth) =>
      raise broad_serving_error(BroadServingHealth)
    RuntimeSpawnedServingValidationFailed(SpawnedServingReadiness) =>
      raise broad_serving_error(BroadServingReadiness)
    _ => raise broad_serving_error(BroadServingConnect)
  }
}

///|
async fn write_broad_serving_frame(
  client : @socket.Tcp,
  frame : FixedArray[Byte],
  timeout_millis : Int,
) -> Unit {
  write_spawned_serving_frame(client, frame, timeout_millis) catch {
    _ => raise broad_serving_error(BroadServingIngress)
  }
}

///|
async fn read_broad_serving_event(
  owner : RuntimeInstanceOwner,
  client : @socket.Tcp,
  storage : FixedArray[Byte],
  decoder : @framed_wire.EventFrameBuffer,
  timeout_millis : Int,
  failure_rule : RuntimeSpawnedBroadServingValidationRule,
) -> @inference.StreamEvent {
  let mut cursor = 0
  let mut declared = 16
  for _ in 0..<timeout_millis {
    if cursor >= declared {
      return decoder.load(storage, declared).event() catch {
          _ => raise broad_serving_error(failure_rule)
        }
    }
    let remaining = declared - cursor
    let received = @async.with_timeout_opt(1, () => {
      client.read(storage, offset=cursor, max_len=remaining)
    }) catch {
      _ => raise broad_serving_error(failure_rule)
    }
    let count = match received {
      Some(value) => value
      None => {
        ignore(owner.progress())
        if owner.health() != RuntimeHealthy ||
          !owner.spawned_serving_ingress_live() {
          raise broad_serving_error(failure_rule)
        }
        @async.pause()
        continue
      }
    }
    if count <= 0 || count > remaining {
      raise broad_serving_error(failure_rule)
    }
    cursor += count
    if cursor == 16 {
      declared = spawned_serving_read_u32(storage, 12)
      if declared < 16 || declared > storage.length() {
        raise broad_serving_error(failure_rule)
      }
    }
  }
  raise broad_serving_error(failure_rule)
}

///|
async fn retire_broad_serving_client(
  owner : RuntimeInstanceOwner,
  turn_limit : Int,
  expected_accepts : UInt64,
  expected_disconnects : UInt64,
  expected_rejections : UInt64,
) -> Unit {
  for turn in 0..<turn_limit {
    ignore(owner.progress())
    if owner.health() != RuntimeHealthy {
      raise broad_serving_error(BroadServingHealth)
    }
    if !owner.spawned_serving_ingress_live() {
      raise broad_serving_error(BroadServingReadiness)
    }
    let sample = capture_broad_serving_metrics(owner)
    if owner.spawned_serving_ingress_listening() &&
      sample.network_accepts == expected_accepts &&
      sample.network_disconnects == expected_disconnects &&
      sample.network_rejections == expected_rejections &&
      sample.queue_depth == 0 &&
      sample.active_requests == 0 &&
      sample.kv_pages_used == 0 {
      return
    }
    if turn % 64 == 63 {
      @async.pause()
    }
  }
  raise broad_serving_error(BroadServingMetrics)
}
