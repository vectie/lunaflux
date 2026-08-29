///|
/// Waits without consuming bytes for the active command read to become
/// readable, bounded by both the caller interval and the frame deadline.
pub fn RankChildControl::wait_receive(
  self : RankChildControl,
  maximum_wait_millis~ : Int,
) -> Bool raise RankChildControlError {
  self.require_usable()
  if !self.receive_active || self.receive_held || self.inbound_active {
    return self.reject(ReceiveBusy)
  }
  self.channel.wait_pending_readable(maximum_wait_millis~) catch {
    error => return self.map_transport(error)
  }
}

///|
/// Waits without publishing bytes for the active response write to become
/// writable, bounded by both the caller interval and the frame deadline.
pub fn RankChildControl::wait_send(
  self : RankChildControl,
  maximum_wait_millis~ : Int,
) -> Bool raise RankChildControlError {
  self.require_usable()
  if !self.send_active {
    return self.reject(SendBusy)
  }
  self.channel.wait_pending_writable(maximum_wait_millis~) catch {
    error => return self.map_transport(error)
  }
}

///|
/// Performs a bounded readability wait while Configure is retained and no
/// subsequent receive may yet begin. It consumes no command bytes.
pub fn RankChildControl::wait_peer_readable(
  self : RankChildControl,
  maximum_wait_millis~ : Int,
) -> Bool raise RankChildControlError {
  self.require_usable()
  if self.receive_active || self.receive_held || self.send_active {
    return self.reject(Lifecycle)
  }
  self.channel.wait_readable(maximum_wait_millis~) catch {
    error => return self.map_transport(error)
  }
}
