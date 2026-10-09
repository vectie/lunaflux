///|
pub(all) suberror RemoteCoordinatorError {
  Binding
  Unclosed
}

///|
/// Owns both authenticated sessions around a generation-scoped body. Either
/// failed session cancels that body and invalidates the whole group. Keep an
/// API loop that must survive replacement outside this cancellation scope.
pub async fn[T] with_group(
  group : @remote_group.RemoteGroup,
  rank_zero : @remote_tls.AuthenticatedRemoteChannel,
  rank_one : @remote_tls.AuthenticatedRemoteChannel,
  body : async () -> T,
) -> T {
  defer {
    rank_zero.close()
    rank_one.close()
  }
  errdefer group.transport_lost()
  let startup = group.startup()
  let zero = startup.rank(0).unwrap()
  let one = startup.rank(1).unwrap()
  let bytes = FixedArray::make(
    @remote_startup.remote_startup_frame_bytes(),
    b'\x00',
  )
  ignore(zero.copy_to(bytes))
  rank_zero.verify_binding(Bytes::makei(bytes.length(), i => bytes[i]))
  ignore(one.copy_to(bytes))
  rank_one.verify_binding(Bytes::makei(bytes.length(), i => bytes[i]))
  @remote_session.with_session(
    rank_zero,
    group.mailbox(0),
    Coordinator,
    generation=zero.group_generation_value(),
    lease_millis=zero.lease_millis(),
    heartbeat=_ => (),
    () => {
      @remote_session.with_session(
        rank_one,
        group.mailbox(1),
        Coordinator,
        generation=one.group_generation_value(),
        lease_millis=one.lease_millis(),
        heartbeat=_ => (),
        () => {
          let result = body()
          if group.lifecycle() != Closed {
            raise Unclosed
          }
          result
        },
      )
    },
  )
}
