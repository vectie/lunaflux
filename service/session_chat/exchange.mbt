///|
async fn Service::receive(
  self : Self,
  connection : @socket.Tcp,
  work : @http1.LunaHttp1Work,
) -> @http1.LunaHttp1View {
  while true {
    match work.state() {
      LunaHttp1Ready => return work.take_view()
      LunaHttp1Failed => raise work.failure()
      LunaHttp1ReceivingHead | LunaHttp1ReceivingBody => {
        let count = connection.read(self.input)
        if count == 0 {
          work.finish_input()
          continue
        }
        let mut offset = 0
        while offset < count {
          let consumed = work.offer(
            self.input,
            source_offset=offset,
            length=count - offset,
          )
          offset += consumed
          if consumed == 0 {
            ignore(work.progress())
          }
          if work.state() is LunaHttp1Failed {
            raise work.failure()
          }
          if work.state() is LunaHttp1Ready && offset != count {
            raise SessionChatError::UnsupportedRequest
          }
        }
      }
      _ => {
        ignore(work.progress())
        @async.pause()
      }
    }
  } nobreak {
    raise SessionChatError::InvalidLifecycle
  }
}

///|
async fn Service::send_event(
  self : Self,
  connection : @socket.Tcp,
  view : @event.LunaEventView,
  streaming~ : Bool,
) -> Unit {
  let work = self.presentation.begin(
    view,
    LunaOpenAIChatCompletions,
    response_sequence=self.sequence,
    streaming~,
  )
  defer (work.abort() catch { _ => () })
  while work.state() is LunaOpenAICompatPending {
    ignore(work.progress())
  }
  if work.state() is LunaOpenAICompatFailed {
    raise work.failure()
  }
  let output = work.take_view()
  defer output.release()
  // The shared renderer retains buffered nonterminal text and emits SSE
  // chunks immediately for streaming requests. Await each write while its
  // event/output views are still live, preserving bounded backpressure.
  if output.length() == 0 {
    return
  }
  if !self.response_started {
    let head = if streaming {
      self.headers.begin_event_stream()
    } else {
      self.headers.begin_json(
        content_length=output.length(),
        failed=output.failed_response(),
      )
    }
    while head.state() is LunaHttp1ResponseBuilding {
      ignore(head.progress())
    }
    let header = head.take_view()
    defer header.release()
    let length = header.copy_chunk(
      self.wire,
      destination_offset=0,
      source_offset=0,
      length=header.length(),
    )
    self.response_started = true
    connection.write(Bytes::from_array(self.wire[:length].to_owned()))
  }
  let mut offset = 0
  while offset < output.length() {
    let count = (output.length() - offset)
      .min(self.wire.length())
      .min(output.maximum_copy_bytes())
    ignore(
      output.copy_chunk_to(
        self.wire,
        destination_offset=0,
        source_offset=offset,
        length=count,
      ),
    )
    connection.write(Bytes::from_array(self.wire[:count].to_owned()))
    offset += count
  }
}

///|
async fn Service::exchange(self : Self, connection : @socket.Tcp) -> Unit {
  let http = self.http.begin()
  defer (http.abort() catch { _ => () })
  let incoming = self.receive(connection, http)
  defer (if incoming.is_live() { incoming.release() })
  guard incoming.route() is LunaHttp1ChatCompletions else {
    raise SessionChatError::UnsupportedRequest
  }
  defer (if self.events.has_event() { self.events.discard() })
  let streaming : Ref[Bool?] = Ref(None)
  @async.with_task_group() <| group => {
    // After the complete request body, EOF or further bytes retire this
    // single-request connection and cancel model work before admitting another.
    group.spawn_bg(() => {
      ignore(connection.read_some(max_len=1))
      group.return_immediately(())
    })
    (self.handle)(
      incoming,
      @inference.RequestId::new(self.sequence),
      self.events,
      selected => {
        guard streaming.val is None else {
          raise SessionChatError::InvalidLifecycle
        }
        streaming.val = Some(selected)
      },
      event => {
        guard streaming.val is Some(selected) else {
          raise SessionChatError::InvalidLifecycle
        }
        self.send_event(connection, event, streaming=selected)
      },
    )
    group.return_immediately(())
  }
}

///|
async fn admit_text(
  workspace : @compat.LunaOpenAIInboundWorkspace,
  incoming : @http1.LunaHttp1View,
  request_id : @inference.RequestId,
  body : async (@inference.LunaTextRequestHandoffLease, Bool) -> Unit,
) -> Unit {
  let work = workspace.begin(incoming, request_id)
  defer (work.abort() catch { _ => () })
  while work.state() is LunaOpenAIInboundPending {
    ignore(work.progress())
    @async.pause()
  }
  if work.state() is LunaOpenAIInboundFailed {
    raise work.failure()
  }
  let inbound = work.take_view()
  defer inbound.release()
  body(inbound.handoff(), inbound.streaming_response())
}
