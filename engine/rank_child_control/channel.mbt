///|
priv enum RankChildChannel {
  Local(@process.InheritedChannel)
  Remote(@remote_mailbox.RemoteMailbox, Ref[FixedArray[Byte]?])
}

///|
fn RankChildChannel::begin_read_frame_or_eof(
  self : RankChildChannel,
  destination : FixedArray[Byte],
  staging~ : FixedArray[Byte],
) -> Unit raise @process.ProcessError {
  match self {
    Local(channel) => channel.begin_read_frame_or_eof(destination, staging~)
    Remote(mailbox, retained) => {
      mailbox.begin_read() catch {
        _ => raise @process.ProcessError::InvalidState(Read)
      }
      retained.val = Some(destination)
    }
  }
}

///|
fn RankChildChannel::progress_read_frame_or_eof(
  self : RankChildChannel,
) -> Int? raise @process.ProcessError {
  match self {
    Local(channel) => channel.progress_read_frame_or_eof()
    Remote(mailbox, retained) => {
      guard retained.val is Some(destination) else {
        raise @process.ProcessError::InvalidState(Read)
      }
      mailbox.poll_read(destination) catch {
        _ => raise @process.ProcessError::ChannelClosed(Read)
      }
    }
  }
}

///|
fn RankChildChannel::begin_write_frame(
  self : RankChildChannel,
  bytes : FixedArray[Byte],
  offset~ : Int,
  byte_count~ : Int,
) -> Unit raise @process.ProcessError {
  match self {
    Local(channel) => channel.begin_write_frame(bytes, offset~, byte_count~)
    Remote(mailbox, _) =>
      mailbox.begin_write(bytes, offset, byte_count) catch {
        _ => raise @process.ProcessError::ChannelClosed(Write)
      }
  }
}

///|
fn RankChildChannel::progress_write_frame(
  self : RankChildChannel,
) -> Bool raise @process.ProcessError {
  match self {
    Local(channel) => channel.progress_write_frame()
    Remote(mailbox, _) =>
      mailbox.poll_write() catch {
        _ => raise @process.ProcessError::ChannelClosed(Write)
      }
  }
}

///|
fn RankChildChannel::wait_pending_readable(
  self : RankChildChannel,
  maximum_wait_millis~ : Int,
) -> Bool raise @process.ProcessError {
  match self {
    Local(channel) => channel.wait_pending_readable(maximum_wait_millis~)
    Remote(_, _) => false
  }
}

///|
fn RankChildChannel::wait_pending_writable(
  self : RankChildChannel,
  maximum_wait_millis~ : Int,
) -> Bool raise @process.ProcessError {
  match self {
    Local(channel) => channel.wait_pending_writable(maximum_wait_millis~)
    Remote(_, _) => false
  }
}

///|
fn RankChildChannel::wait_readable(
  self : RankChildChannel,
  maximum_wait_millis~ : Int,
) -> Bool raise @process.ProcessError {
  match self {
    Local(channel) => channel.wait_readable(maximum_wait_millis~)
    Remote(_, _) => false
  }
}
