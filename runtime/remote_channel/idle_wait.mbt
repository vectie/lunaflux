///|
/// Suspend only at an empty, indefinitely idle frame boundary. Consumes at most
/// the first prefix byte into the channel's existing private buffer; publication
/// and the bounded transfer deadline remain owned by the next fresh-clock poll.
/// This avoids manufacturing/cancelling a task group for every idle poll.
pub async fn RemoteChannel::wait_idle_receive(
  self : RemoteChannel,
  allow_idle_eof~ : Bool = false,
) -> Bool {
  if !self.open ||
    !self.receiving ||
    !self.receive_waiting ||
    self.receive_cursor != 0 ||
    self.receive_length != 4 {
    return false
  }
  guard !self.reading else { raise InvalidState }
  self.reading = true
  defer {
    self.reading = false
    if @async.is_being_cancelled() {
      self.close()
    }
  }
  let count = self.connection.read(self.input, offset=0, max_len=1) catch {
    error => {
      self.close()
      if @async.is_being_cancelled() {
        raise error
      }
      raise Disconnected
    }
  }
  if count == 0 && allow_idle_eof {
    self.close()
    return true
  }
  if count != 1 {
    self.fail(Disconnected)
  }
  self.receive_cursor = 1
  true
}
