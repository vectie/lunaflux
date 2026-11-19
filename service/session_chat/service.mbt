///|
pub(all) suberror SessionChatError {
  UnsupportedRequest
  InvalidLifecycle
} derive(Debug)

///|
/// One active request, reusing the existing bounded HTTP/OpenAI parsers and
/// canonical event renderer. Backend callbacks own model execution and cleanup.
pub struct Service {
  priv input : FixedArray[Byte]
  priv wire : FixedArray[Byte]
  priv http : @http1.LunaHttp1Workspace
  priv presentation : @compat.LunaOpenAICompatWorkspace
  priv events : @event.LunaEventOwner
  priv headers : @http1.LunaHttp1ResponseWorkspace
  priv handle : async (
    @http1.LunaHttp1View,
    @inference.RequestId,
    @event.LunaEventOwner,
    (Bool) -> Unit raise,
    async (@event.LunaEventView) -> Unit,
  ) -> Unit
  priv ready : () -> Bool
  priv capacity_error : Bytes
  priv validation_error : Bytes
  priv authentication_error : Bytes
  priv service_error : Bytes
  priv timeout_ms : Int32
  priv mut sequence : UInt64
  priv mut busy : Bool
  priv mut closed : Bool
  priv mut response_started : Bool
}

///|
fn error_bytes(
  workspace : @http1.LunaHttp1ResponseWorkspace,
  kind : @http1.LunaHttp1ErrorResponseKind,
) -> Bytes raise {
  let work = workspace.begin_error(kind)
  while work.state() is LunaHttp1ResponseBuilding {
    ignore(work.progress())
  }
  let view = work.take_view()
  defer view.release()
  let bytes = FixedArray::make(view.length(), b'\x00')
  ignore(
    view.copy_chunk(
      bytes,
      destination_offset=0,
      source_offset=0,
      length=bytes.length(),
    ),
  )
  Bytes::from_array(bytes[:].to_owned())
}

///|
/// The caller owns the auth policy and model session. The supplied inbound and
/// presentation workspaces are exclusively lent until this service is closed.
pub fn Service::new(
  limits : @inference.InferenceLimits,
  policy : @auth.LunaApiAuthPolicy,
  inbound : @compat.LunaOpenAIInboundWorkspace,
  presentation : @compat.LunaOpenAICompatWorkspace,
  timeout_ms~ : Int32,
  generate~ : async (
    @inference.LunaTextRequestHandoffLease,
    @event.LunaEventOwner,
    async (@event.LunaEventView) -> Unit,
  ) -> Unit,
  ready~ : () -> Bool,
) -> Self raise {
  Service::with_handler(
    limits,
    policy,
    presentation,
    timeout_ms~,
    handle=(incoming, request_id, events, select_response, emit) => {
      admit_text(inbound, incoming, request_id, (handoff, streaming) => {
        select_response(streaming)
        generate(handoff, events, emit)
      })
    },
    ready~,
  )
}

///|
/// Authenticated single-request HTTP exchange with model-owned admission.
/// The handler owns parsing/preparation and any scoped request handoff; it must
/// cancel model work before returning from cancellation. It borrows the view.
/// After admission, call select_response(streaming) once before emitting events.
pub fn Service::with_handler(
  limits : @inference.InferenceLimits,
  policy : @auth.LunaApiAuthPolicy,
  presentation : @compat.LunaOpenAICompatWorkspace,
  timeout_ms~ : Int32,
  maximum_body_bytes~ : Int32 = 1048576,
  handle~ : async (
    @http1.LunaHttp1View,
    @inference.RequestId,
    @event.LunaEventOwner,
    (Bool) -> Unit raise,
    async (@event.LunaEventView) -> Unit,
  ) -> Unit,
  ready~ : () -> Bool,
) -> Self raise {
  guard timeout_ms > 0 && timeout_ms <= 300000 else {
    raise SessionChatError::InvalidLifecycle
  }
  let headers = @http1.LunaHttp1ResponseWorkspace::new(
    @http1.LunaHttp1ResponseStepBudget::new(4096),
  )
  {
    input: FixedArray::make(16384, b'\x00'),
    wire: FixedArray::make(16384, b'\x00'),
    http: @http1.LunaHttp1Workspace::new(
      @http1.LunaHttp1Limits::new(
        max_head_bytes=16384,
        max_body_bytes=maximum_body_bytes,
      ),
      @http1.LunaHttp1StepBudget::new(4096),
      policy,
    ),
    presentation,
    events: @event.LunaEventOwner::new(limits),
    headers,
    capacity_error: error_bytes(headers, LunaHttp1CapacityError),
    validation_error: error_bytes(headers, LunaHttp1ValidationError),
    authentication_error: error_bytes(headers, LunaHttp1AuthenticationError),
    service_error: error_bytes(headers, LunaHttp1ServiceError),
    handle,
    ready,
    timeout_ms,
    sequence: 0UL,
    busy: false,
    closed: false,
    response_started: false,
  }
}

///|
pub fn Service::close(self : Self) -> Unit {
  self.closed = true
  self.http.close_authentication()
}

///|
pub fn Service::is_ready(self : Self) -> Bool {
  !self.closed && !self.busy && (self.ready)()
}

///|
/// Stop/cancel serve before closing the borrowed model session. Each accepted
/// TCP connection carries one request; concurrent arrivals get the cached 429.
pub async fn Service::serve(self : Self, listener : @socket.TcpServer) -> Unit {
  guard !self.closed else { raise SessionChatError::InvalidLifecycle }
  listener.run_forever(max_connections=8, allow_failure=true) <| ((
    connection,
    _,
  ) => {
    defer connection.close()
    if self.busy {
      connection.write(self.capacity_error)
      return
    }
    if self.closed || !(self.ready)() {
      connection.write(self.service_error)
      return
    }
    self.busy = true
    self.response_started = false
    defer {
      self.busy = false
    }
    if self.sequence == 0xffffffffffffffffUL {
      connection.write(self.service_error)
      return
    }
    self.sequence += 1UL
    @async.with_timeout(self.timeout_ms, () => self.exchange(connection)) catch {
      error => {
        if @async.is_being_cancelled() {
          raise error
        }
        if self.response_started {
          return
        }
        let response = match error {
          @http1.LunaHttp1Error::LunaHttp1Failed(
            rule=LunaHttp1Authorization,
            ..
          ) => self.authentication_error
          @http1.LunaHttp1Error::LunaHttp1Failed(..)
          | @compat.LunaOpenAICompatError::LunaOpenAICompatInvalid(_)
          | SessionChatError::UnsupportedRequest => self.validation_error
          _ => self.service_error
        }
        connection.write(response)
      }
    }
  })
}
