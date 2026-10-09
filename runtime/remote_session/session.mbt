///|
pub(all) enum RemoteSessionRole {
  Coordinator
  Worker
}

///|
pub(all) suberror RemoteSessionError {
  Configuration
  Protocol
  LeaseExpired
}

///|
priv struct Session {
  channel : @remote_tls.AuthenticatedRemoteChannel
  mailbox : @remote_mailbox.RemoteMailbox
  role : RemoteSessionRole
  generation : UInt64
  interval : UInt64
  clock : @monotonic_clock.ReusableMonotonicRead
  heartbeat : (UInt64) -> Unit raise
  mut lease : @remote_lease.RemoteLease
  mut sent_sequence : UInt64
  mut received_sequence : UInt64
  mut acknowledged_sequence : UInt64
  mut next_heartbeat : UInt64
  mut finishing : Bool
  mut goodbye_sent : Bool
  mut peer_finished : Bool
}

///|
fn Session::poll(self : Session) -> UInt64 raise {
  if !self.clock.poll() {
    raise LeaseExpired
  }
  let now = self.clock.value()
  self.lease = self.lease.poll(now)
  if self.lease.status() != Active {
    raise LeaseExpired
  }
  now
}

///|
fn sequence(bytes : FixedArray[Byte]) -> UInt64 {
  let mut value = 0UL
  for i in 0..<8 {
    value = value | (bytes[i + 1].to_uint64() << (8 * i))
  }
  value
}

///|
async fn Session::read_loop(self : Session) -> Unit {
  let bytes = FixedArray::make(self.mailbox.capacity() + 1, b'\x00')
  let payload = FixedArray::make(self.mailbox.capacity(), b'\x00')
  for ;; {
    let length = self.channel.receive(bytes)
    let now = self.poll()
    match bytes[0] {
      b'\x03' => {
        if length != 1 {
          raise Protocol
        }
        self.peer_finished = true
        return
      }
      b'\x00' => {
        if length < 2 {
          raise Protocol
        }
        for i in 1..<length {
          payload[i - 1] = bytes[i]
        }
        while !self.mailbox.can_deliver() {
          ignore(self.poll())
          if self.mailbox.is_failed() {
            raise Protocol
          }
          @async.sleep(1)
        }
        self.mailbox.deliver(payload, length - 1)
        // Retain a terminal response until its synchronous owner consumes it.
        // Do not let an immediately following clean TLS EOF erase Closed.
        while !self.mailbox.can_deliver() {
          ignore(self.poll())
          if self.mailbox.is_failed() {
            raise Protocol
          }
          @async.sleep(1)
        }
      }
      b'\x01' | b'\x02' => {
        if length != 9 {
          raise Protocol
        }
        match self.role {
          Worker => if bytes[0] != b'\x01' { raise Protocol }
          Coordinator => if bytes[0] != b'\x02' { raise Protocol }
        }
        let value = sequence(bytes)
        if self.received_sequence == 0xffffffffffffffffUL ||
          value != self.received_sequence + 1UL {
          raise Protocol
        }
        match self.role {
          Coordinator => if value > self.sent_sequence { raise Protocol }
          Worker => ()
        }
        self.lease = self.lease.renew(
          generation=self.generation,
          sequence=value,
          local_now_millis=now,
        )
        if self.lease.status() != Active {
          raise LeaseExpired
        }
        // For a worker this is the exact executor watchdog renewal, before ack.
        (self.heartbeat)(value)
        self.received_sequence = value
      }
      _ => raise Protocol
    }
  }
}

///|
async fn Session::write_loop(self : Session) -> Unit {
  let payload = FixedArray::make(self.mailbox.capacity(), b'\x00')
  let bytes = FixedArray::make(self.mailbox.capacity() + 1, b'\x00')
  for ;; {
    let now = self.poll()
    if self.finishing {
      bytes[0] = b'\x03'
      self.channel.send(bytes, 1)
      self.goodbye_sent = true
      return
    }
    if self.peer_finished {
      @async.sleep(1)
      continue
    }
    let heartbeat = match self.role {
      Coordinator =>
        if now >= self.next_heartbeat {
          if self.sent_sequence == 0xffffffffffffffffUL ||
            now > 0xffffffffffffffffUL - self.interval {
            raise Protocol
          }
          self.sent_sequence += 1UL
          self.next_heartbeat = now + self.interval
          bytes[0] = b'\x01'
          Some(self.sent_sequence)
        } else {
          None
        }
      Worker =>
        if self.acknowledged_sequence < self.received_sequence {
          self.acknowledged_sequence += 1UL
          bytes[0] = b'\x02'
          Some(self.acknowledged_sequence)
        } else {
          None
        }
    }
    match heartbeat {
      Some(value) => {
        for i in 0..<8 {
          bytes[i + 1] = (value >> (8 * i)).to_byte()
        }
        self.channel.send(bytes, 9)
      }
      None =>
        match self.mailbox.outgoing(payload) {
          Some(length) => {
            bytes[0] = b'\x00'
            for i in 0..<length {
              bytes[i + 1] = payload[i]
            }
            self.channel.send(bytes, length + 1)
            self.mailbox.sent()
          }
          None => @async.sleep(1)
        }
    }
  }
}

///|
/// Runs authenticated duplex I/O and independent idle heartbeat tasks around
/// the existing synchronous protocol driver. Errors/cancellation close both
/// directions and poison the mailbox. No reconnect or lease resurrection.
pub async fn[T] with_session(
  channel : @remote_tls.AuthenticatedRemoteChannel,
  mailbox : @remote_mailbox.RemoteMailbox,
  role : RemoteSessionRole,
  generation~ : UInt64,
  lease_millis~ : UInt64,
  heartbeat~ : (UInt64) -> Unit raise,
  body : async () -> T,
) -> T {
  defer {
    channel.close()
    mailbox.fail()
  }
  if lease_millis < 30UL ||
    lease_millis > 300000UL ||
    mailbox.capacity() < 9 ||
    channel.capacity() != mailbox.capacity() + 1 {
    raise Configuration
  }
  let clock = @monotonic_clock.MonotonicClock::system().prepare_poller()
  if !clock.poll() {
    raise LeaseExpired
  }
  let now = clock.value()
  let session : Session = {
    channel,
    mailbox,
    role,
    generation,
    interval: lease_millis / 3UL,
    clock,
    heartbeat,
    lease: @remote_lease.RemoteLease::start(
      generation~,
      duration_millis=lease_millis,
      local_now_millis=now,
    ),
    sent_sequence: 0UL,
    received_sequence: 0UL,
    acknowledged_sequence: 0UL,
    next_heartbeat: now,
    finishing: false,
    goodbye_sent: false,
    peer_finished: false,
  }
  @async.with_task_group() <| tasks => {
    tasks.spawn_bg(() => session.read_loop())
    tasks.spawn_bg(() => session.write_loop())
    let result = body()
    session.finishing = true
    while !session.goodbye_sent || !session.peer_finished {
      ignore(session.poll())
      @async.sleep(1)
    }
    tasks.return_immediately(result)
    result
  }
}
