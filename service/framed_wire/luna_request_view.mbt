///|
fn LunaFramedRequestView::require_workspace(
  self : LunaFramedRequestView,
) -> LunaFramedRequestWorkspace raise FramedWireError {
  if self.epoch == 0UL ||
    self.workspace.epoch != self.epoch ||
    self.workspace.phase != LUNA_FRAMED_REQUEST_VIEW_LEASED {
    raise LunaFramedRequestStale
  }
  self.workspace
}

///|
pub fn LunaFramedRequestView::release(
  self : LunaFramedRequestView,
) -> Unit raise FramedWireError {
  let workspace = self.require_workspace()
  workspace.phase = LUNA_FRAMED_REQUEST_IDLE
  workspace.failure = None
  workspace.last_work_units = 0
}

///|
pub fn LunaFramedRequestView::length(
  self : LunaFramedRequestView,
) -> Int raise FramedWireError {
  self.require_workspace().declared_length
}

///|
pub fn LunaFramedRequestView::inference_limits(
  self : LunaFramedRequestView,
) -> @inference.InferenceLimits raise FramedWireError {
  self.require_workspace().limits.inference_limits()
}

///|
pub fn LunaFramedRequestView::protocol_version_wire(
  self : LunaFramedRequestView,
) -> Int raise FramedWireError {
  let workspace = self.require_workspace()
  read_i32(workspace.storage, 24)
}

///|
pub fn LunaFramedRequestView::request_id_value(
  self : LunaFramedRequestView,
) -> UInt64 raise FramedWireError {
  let workspace = self.require_workspace()
  read_u64(workspace.storage, 104)
}

///|
pub fn LunaFramedRequestView::input_kind(
  self : LunaFramedRequestView,
) -> LunaFramedRequestInputKind raise FramedWireError {
  if (self.require_workspace().flags & 1) != 0 {
    LunaFramedRequestText
  } else {
    LunaFramedRequestTokenIds
  }
}

///|
pub fn LunaFramedRequestView::input_length(
  self : LunaFramedRequestView,
) -> Int raise FramedWireError {
  self.require_workspace().input_count
}

///|
pub fn LunaFramedRequestView::input_byte_at(
  self : LunaFramedRequestView,
  index : Int,
) -> Byte raise FramedWireError {
  let workspace = self.require_workspace()
  if (workspace.flags & 1) == 0 || index < 0 || index >= workspace.input_count {
    raise request_invalid(Input)
  }
  workspace.storage[workspace.input_offset + index]
}

///|
pub fn LunaFramedRequestView::input_token_at(
  self : LunaFramedRequestView,
  index : Int,
) -> Int raise FramedWireError {
  let workspace = self.require_workspace()
  if (workspace.flags & 1) != 0 || index < 0 || index >= workspace.input_count {
    raise request_invalid(Input)
  }
  read_i32(workspace.storage, workspace.input_offset + index * 4)
}

///|
pub fn LunaFramedRequestView::max_new_tokens(
  self : LunaFramedRequestView,
) -> Int raise FramedWireError {
  let workspace = self.require_workspace()
  read_i32(workspace.storage, 52)
}

///|
pub fn LunaFramedRequestView::context_ceiling(
  self : LunaFramedRequestView,
) -> Int raise FramedWireError {
  let workspace = self.require_workspace()
  read_i32(workspace.storage, 56)
}

///|
pub fn LunaFramedRequestView::sampling_mode(
  self : LunaFramedRequestView,
) -> @inference.SamplingMode raise FramedWireError {
  let workspace = self.require_workspace()
  if (workspace.flags & 2) == 0 {
    Greedy
  } else {
    Sample
  }
}

///|
pub fn LunaFramedRequestView::sampling_temperature(
  self : LunaFramedRequestView,
) -> Double raise FramedWireError {
  let workspace = self.require_workspace()
  read_u64(workspace.storage, 72).reinterpret_as_double()
}

///|
pub fn LunaFramedRequestView::has_top_k(
  self : LunaFramedRequestView,
) -> Bool raise FramedWireError {
  (self.require_workspace().flags & 4) != 0
}

///|
pub fn LunaFramedRequestView::top_k(
  self : LunaFramedRequestView,
) -> Int raise FramedWireError {
  let workspace = self.require_workspace()
  read_i32(workspace.storage, 64)
}

///|
pub fn LunaFramedRequestView::has_top_p(
  self : LunaFramedRequestView,
) -> Bool raise FramedWireError {
  (self.require_workspace().flags & 8) != 0
}

///|
pub fn LunaFramedRequestView::top_p(
  self : LunaFramedRequestView,
) -> Double raise FramedWireError {
  let workspace = self.require_workspace()
  read_u64(workspace.storage, 80).reinterpret_as_double()
}

///|
pub fn LunaFramedRequestView::sampling_seed_value(
  self : LunaFramedRequestView,
) -> UInt64 raise FramedWireError {
  let workspace = self.require_workspace()
  read_u64(workspace.storage, 88)
}

///|
pub fn LunaFramedRequestView::stream_preference(
  self : LunaFramedRequestView,
) -> @inference.StreamPreference raise FramedWireError {
  let flags = self.require_workspace().flags
  if (flags & 64) != 0 {
    TokenIdsStreaming
  } else if (flags & 16) != 0 {
    Streaming
  } else {
    Buffered
  }
}

///|
pub fn LunaFramedRequestView::deadline_millis(
  self : LunaFramedRequestView,
) -> UInt64 raise FramedWireError {
  let workspace = self.require_workspace()
  read_u64(workspace.storage, 96)
}

///|
pub fn LunaFramedRequestView::cache_permission(
  self : LunaFramedRequestView,
) -> @inference.CachePermission raise FramedWireError {
  let workspace = self.require_workspace()
  match read_i32(workspace.storage, 240) {
    0 => Disabled
    1 => ReadOnly
    2 => ReadWrite
    _ => abort("Luna framed request permission invariant")
  }
}

///|
pub fn LunaFramedRequestView::content_digest_byte_at(
  self : LunaFramedRequestView,
  index : Int,
) -> Byte raise FramedWireError {
  let workspace = self.require_workspace()
  if index < 0 || index >= 64 {
    raise request_invalid(Identity)
  }
  workspace.storage[112 + index]
}

///|
pub fn LunaFramedRequestView::plan_digest_byte_at(
  self : LunaFramedRequestView,
  index : Int,
) -> Byte raise FramedWireError {
  let workspace = self.require_workspace()
  if index < 0 || index >= 64 {
    raise request_invalid(Identity)
  }
  workspace.storage[176 + index]
}

///|
pub fn LunaFramedRequestView::stop_token_count(
  self : LunaFramedRequestView,
) -> Int raise FramedWireError {
  self.require_workspace().stop_token_count
}

///|
pub fn LunaFramedRequestView::stop_token_at(
  self : LunaFramedRequestView,
  index : Int,
) -> Int raise FramedWireError {
  let workspace = self.require_workspace()
  if index < 0 || index >= workspace.stop_token_count {
    raise request_invalid(StopTable)
  }
  read_i32(workspace.storage, workspace.stop_token_offset + index * 4)
}

///|
pub fn LunaFramedRequestView::stop_string_count(
  self : LunaFramedRequestView,
) -> Int raise FramedWireError {
  self.require_workspace().stop_string_count
}

///|
pub fn LunaFramedRequestView::stop_string_length(
  self : LunaFramedRequestView,
  index : Int,
) -> Int raise FramedWireError {
  let workspace = self.require_workspace()
  if index < 0 || index >= workspace.stop_string_count {
    raise request_invalid(StopTable)
  }
  workspace.stop_string_lengths[index]
}

///|
pub fn LunaFramedRequestView::stop_string_byte_at(
  self : LunaFramedRequestView,
  string_index : Int,
  byte_index : Int,
) -> Byte raise FramedWireError {
  let workspace = self.require_workspace()
  if string_index < 0 ||
    string_index >= workspace.stop_string_count ||
    byte_index < 0 ||
    byte_index >= workspace.stop_string_lengths[string_index] {
    raise request_invalid(StopTable)
  }
  workspace.storage[workspace.stop_string_offsets[string_index] + byte_index]
}

///|
pub fn LunaFramedRequestView::cache_scope_length(
  self : LunaFramedRequestView,
) -> Int raise FramedWireError {
  self.require_workspace().scope_length
}

///|
pub fn LunaFramedRequestView::cache_scope_byte_at(
  self : LunaFramedRequestView,
  index : Int,
) -> Byte raise FramedWireError {
  let workspace = self.require_workspace()
  if index < 0 || index >= workspace.scope_length {
    raise request_invalid(Payload)
  }
  workspace.storage[workspace.scope_offset + index]
}

///|
pub fn LunaFramedRequestView::trace_length(
  self : LunaFramedRequestView,
) -> Int raise FramedWireError {
  self.require_workspace().trace_length
}

///|
pub fn LunaFramedRequestView::trace_byte_at(
  self : LunaFramedRequestView,
  index : Int,
) -> Byte raise FramedWireError {
  let workspace = self.require_workspace()
  if index < 0 || index >= workspace.trace_length {
    raise request_invalid(Payload)
  }
  workspace.storage[workspace.trace_offset + index]
}

///|
fn materialize_luna_framed_request(
  view : LunaFramedRequestView,
) -> @inference.GenerateRequest raise FramedWireError {
  let workspace = view.require_workspace()
  let inference = workspace.limits.inference_limits()
  let input = if (workspace.flags & 1) != 0 {
    @inference.Input::from_utf8(
      read_bytes(
        workspace.storage,
        workspace.input_offset,
        workspace.input_count,
      ),
      inference,
    ) catch {
      _ => raise request_invalid(Input)
    }
  } else {
    let tokens : Array[Int] = []
    for index in 0..<workspace.input_count {
      tokens.push(
        read_i32(workspace.storage, workspace.input_offset + index * 4),
      )
    }
    @inference.Input::from_token_ids(tokens, inference) catch {
      _ => raise request_invalid(Input)
    }
  }
  let stop_tokens : Array[Int] = []
  for index in 0..<workspace.stop_token_count {
    stop_tokens.push(
      read_i32(workspace.storage, workspace.stop_token_offset + index * 4),
    )
  }
  let stop_strings : Array[Bytes] = []
  for index in 0..<workspace.stop_string_count {
    stop_strings.push(
      read_bytes(
        workspace.storage,
        workspace.stop_string_offsets[index],
        workspace.stop_string_lengths[index],
      ),
    )
  }
  let stops = @inference.StopConditions::new(
    stop_tokens, stop_strings, inference,
  ) catch {
    _ => raise request_invalid(StopTable)
  }
  // Decoded identity is an untrusted claim until exact admission against a
  // rebuilt ModelPlan identity.
  let content = @spec.ContentDigest::from_sha256(
    read_digest(workspace.storage, 112, RequestFrame),
  ) catch {
    _ => raise request_invalid(Identity)
  }
  let plan = @spec.PlanDigest::from_sha256(
    read_digest(workspace.storage, 176, RequestFrame),
  ) catch {
    _ => raise request_invalid(Identity)
  }
  let scope = @inference.CacheScope::from_ascii(
    read_bytes(
      workspace.storage,
      workspace.scope_offset,
      workspace.scope_length,
    ),
    inference,
  ) catch {
    _ => raise request_invalid(Payload)
  }
  let trace = if workspace.trace_length > 0 {
    Some(
      @inference.TraceCorrelation::from_ascii(
        read_bytes(
          workspace.storage,
          workspace.trace_offset,
          workspace.trace_length,
        ),
        inference,
      ) catch {
        _ => raise request_invalid(Payload)
      },
    )
  } else {
    None
  }
  let model_identity = @spec.ModelIdentity::new(content, plan) catch {
    _ => raise request_invalid(Identity)
  }
  @inference.GenerateRequest::new(
    protocol_version=try! @inference.ProtocolVersion::from_wire(
      view.protocol_version_wire(),
    ),
    request_id=try! @inference.RequestId::new(view.request_id_value()),
    model_identity~,
    input~,
    max_new_tokens=view.max_new_tokens(),
    context_ceiling=view.context_ceiling(),
    sampling=if view.sampling_mode() == Greedy {
      @inference.SamplingParameters::greedy()
    } else {
      try! @inference.SamplingParameters::sample(
        temperature=view.sampling_temperature(),
        top_k?=if view.has_top_k() { Some(view.top_k()) } else { None },
        top_p?=if view.has_top_p() { Some(view.top_p()) } else { None },
        inference,
      )
    },
    sampling_seed=try! @inference.SamplingSeed::new(view.sampling_seed_value()),
    stops~,
    stream_preference=view.stream_preference(),
    deadline=try! @inference.DeadlineBudget::from_milliseconds(
      view.deadline_millis(),
      inference,
    ),
    cache=@inference.CachePolicy::new(scope, view.cache_permission()),
    trace~,
    limits=inference,
  ) catch {
    _ => raise request_invalid(CanonicalEncoding)
  }
}
