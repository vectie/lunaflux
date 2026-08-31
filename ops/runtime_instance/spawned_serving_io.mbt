///|
async fn connect_spawned_serving_client(
  owner : RuntimeInstanceOwner,
  timeout_millis : Int,
) -> @socket.Tcp {
  if !owner.spawned_serving_ingress_shape_valid() {
    raise spawned_serving_error(SpawnedServingOwner)
  }
  let addr = owner.singular_ingress_addr() catch {
    _ => raise spawned_serving_error(SpawnedServingOwner)
  }
  let connected = @async.with_timeout_opt(timeout_millis, () => {
    @socket.Tcp::connect(addr)
  }) catch {
    _ => raise spawned_serving_error(SpawnedServingConnect)
  }
  match connected {
    Some(client) => client
    None => raise spawned_serving_error(SpawnedServingConnect)
  }
}

///|
async fn write_spawned_serving_frame(
  client : @socket.Tcp,
  frame : FixedArray[Byte],
  timeout_millis : Int,
) -> Unit {
  let bytes = Bytes::makei(frame.length(), index => frame[index])
  let mut cursor = 0
  while cursor < bytes.length() {
    let remaining = bytes.length() - cursor
    let written = @async.with_timeout_opt(timeout_millis, () => {
      client.write_once(bytes, offset=cursor, len=remaining)
    }) catch {
      _ => raise spawned_serving_error(SpawnedServingIngress)
    }
    let count = match written {
      Some(value) => value
      None => raise spawned_serving_error(SpawnedServingIngress)
    }
    if count <= 0 || count > remaining {
      raise spawned_serving_error(SpawnedServingIngress)
    }
    cursor += count
  }
}

///|
fn spawned_serving_read_u32(source : FixedArray[Byte], offset : Int) -> Int {
  source[offset].to_int() |
  (source[offset + 1].to_int() << 8) |
  (source[offset + 2].to_int() << 16) |
  (source[offset + 3].to_int() << 24)
}

///|
async fn read_spawned_serving_event(
  owner : RuntimeInstanceOwner,
  client : @socket.Tcp,
  storage : FixedArray[Byte],
  decoder : @framed_wire.EventFrameBuffer,
  timeout_millis : Int,
) -> @inference.StreamEvent {
  let mut cursor = 0
  let mut declared = 16
  for _ in 0..<timeout_millis {
    if cursor >= declared {
      return decoder.load(storage, declared).event() catch {
          _ => raise spawned_serving_error(SpawnedServingEvent)
        }
    }
    let remaining = declared - cursor
    let received = @async.with_timeout_opt(1, () => {
      client.read(storage, offset=cursor, max_len=remaining)
    }) catch {
      _ => raise spawned_serving_error(SpawnedServingEvent)
    }
    let count = match received {
      Some(value) => value
      None => {
        ignore(owner.progress())
        if owner.health() != RuntimeHealthy {
          raise spawned_serving_error(SpawnedServingHealth)
        }
        if !owner.spawned_serving_ingress_active() {
          raise spawned_serving_error(SpawnedServingReadiness)
        }
        @async.pause()
        continue
      }
    }
    if count <= 0 || count > remaining {
      raise spawned_serving_error(SpawnedServingEvent)
    }
    cursor += count
    if cursor == 16 {
      declared = spawned_serving_read_u32(storage, 12)
      if declared < 16 || declared > storage.length() {
        raise spawned_serving_error(SpawnedServingEvent)
      }
    }
  }
  raise spawned_serving_error(SpawnedServingEvent)
}
