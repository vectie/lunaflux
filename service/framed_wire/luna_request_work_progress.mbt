///|
fn LunaFramedRequestWorkspace::fail_rule(
  self : LunaFramedRequestWorkspace,
  rule : FramedWireRule,
) -> Unit {
  self.fail(request_invalid(rule))
}

///|
fn LunaFramedRequestWorkspace::progress_prefix_one(
  self : LunaFramedRequestWorkspace,
) -> Unit {
  match self.phase_index {
    0 => if read_u32(self.storage, 0) != request_magic { self.fail_rule(Magic) }
    1 =>
      if read_u32(self.storage, 4) != request_version {
        self.fail_rule(Version)
      }
    2 => if read_u32(self.storage, 8) != 1 { self.fail_rule(Kind) }
    3 => {
      let declared = read_u32(self.storage, 12)
      if (declared & 0x80000000) != 0 {
        self.fail_rule(Length)
        return
      }
      let length = declared.reinterpret_as_int()
      if length < REQUEST_HEADER_BYTES ||
        length > self.limits.max_frame_bytes() ||
        length > self.storage.length() {
        self.fail_rule(Length)
        return
      }
      self.declared_length = length
    }
    _ => abort("Luna framed request prefix invariant")
  }
  if self.phase == LUNA_FRAMED_REQUEST_FAILED {
    return
  }
  self.phase_index += 1
  if self.phase_index == 4 {
    self.phase = LUNA_FRAMED_REQUEST_RECEIVING_BODY
    self.phase_index = 0
  }
}

///|
fn LunaFramedRequestWorkspace::progress_checksum_one(
  self : LunaFramedRequestWorkspace,
) -> Unit {
  if self.phase_index < self.declared_length {
    if self.phase_index < 16 || self.phase_index >= 20 {
      self.checksum = (self.checksum ^ self.storage[self.phase_index].to_uint()) *
        16777619
    }
    self.phase_index += 1
    return
  }
  if read_u32(self.storage, 16) != self.checksum {
    self.fail_rule(Checksum)
    return
  }
  self.phase = LUNA_FRAMED_REQUEST_HEADER
  self.phase_index = 0
}

///|
fn LunaFramedRequestWorkspace::progress_header_one(
  self : LunaFramedRequestWorkspace,
) -> Unit {
  let inference = self.limits.inference_limits()
  match self.phase_index {
    0 => {
      self.flags = read_u32(self.storage, 20)
      if (self.flags & 0xffffffc0) != 0 {
        self.fail_rule(Flags)
      }
    }
    1 =>
      if read_u32(self.storage, 68) != 0 ||
        read_u32(self.storage, 252) != 0 ||
        read_i32(self.storage, 248) != REQUEST_HEADER_BYTES ||
        read_i32(self.storage, 244) !=
        self.declared_length - REQUEST_HEADER_BYTES {
        self.fail_rule(Reserved)
      }
    2 => {
      self.input_count = read_i32(self.storage, 28)
      self.stop_token_count = read_i32(self.storage, 32)
      self.stop_string_count = read_i32(self.storage, 36)
      self.declared_stop_bytes = read_i32(self.storage, 40)
      self.scope_length = read_i32(self.storage, 44)
      self.trace_length = read_i32(self.storage, 48)
      if self.input_count <= 0 ||
        self.stop_token_count < 0 ||
        self.stop_string_count < 0 ||
        self.declared_stop_bytes < 0 ||
        self.scope_length <= 0 ||
        self.trace_length < 0 {
        self.fail_rule(Length)
      }
    }
    3 => {
      let input_limit = if (self.flags & 1) != 0 {
        inference.max_text_bytes()
      } else {
        inference.max_input_tokens()
      }
      if self.input_count > input_limit ||
        self.stop_token_count > inference.max_stop_token_ids() ||
        self.stop_string_count > inference.max_stop_strings() ||
        self.declared_stop_bytes >
        self.stop_string_count * inference.max_stop_string_bytes() ||
        self.scope_length > inference.max_cache_scope_bytes() ||
        self.trace_length > inference.max_trace_bytes() {
        self.fail_rule(Length)
      }
    }
    4 => if read_i32(self.storage, 24) != 1 { self.fail_rule(Version) }
    5 => if read_u64(self.storage, 104) == 0UL { self.fail_rule(Scalar) }
    6 => {
      let maximum = read_i32(self.storage, 52)
      let context = read_i32(self.storage, 56)
      if maximum <= 0 ||
        maximum > inference.max_new_tokens() ||
        context <= 0 ||
        context > inference.max_context_tokens() ||
        maximum > context ||
        ((self.flags & 1) == 0 && self.input_count > context - maximum) {
        self.fail_rule(Scalar)
      }
    }
    7 => {
      let status = self.sampling_header_status()
      if status == 1 {
        self.fail_rule(CanonicalEncoding)
      } else if status == 2 {
        self.fail_rule(Scalar)
      }
    }
    8 => if read_u64(self.storage, 88) == 0UL { self.fail_rule(Scalar) }
    9 => {
      let deadline = read_u64(self.storage, 96)
      if deadline == 0UL || deadline > inference.max_deadline_millis() {
        self.fail_rule(Scalar)
      }
    }
    10 =>
      if read_i32(self.storage, 240) < 0 || read_i32(self.storage, 240) > 2 {
        self.fail_rule(Scalar)
      }
    _ => abort("Luna framed request header invariant")
  }
  if self.phase == LUNA_FRAMED_REQUEST_FAILED {
    return
  }
  self.phase_index += 1
  if self.phase_index == 6 {
    self.phase = LUNA_FRAMED_REQUEST_DIGEST
    self.phase_index = 0
  } else if self.phase_index == 11 {
    self.phase = LUNA_FRAMED_REQUEST_LAYOUT
    self.phase_index = -2
  }
}

///|
fn LunaFramedRequestWorkspace::progress_digest_one(
  self : LunaFramedRequestWorkspace,
) -> Unit {
  let offset = if self.phase_index < 64 {
    112 + self.phase_index
  } else {
    176 + self.phase_index - 64
  }
  if !luna_request_lower_hex_byte(self.storage[offset]) {
    self.fail_rule(Identity)
    return
  }
  self.phase_index += 1
  if self.phase_index == 128 {
    self.phase = LUNA_FRAMED_REQUEST_HEADER
    self.phase_index = 6
  }
}

///|
fn LunaFramedRequestWorkspace::progress_layout_one(
  self : LunaFramedRequestWorkspace,
) -> Unit {
  if self.phase_index == -2 {
    self.cursor = request_append(
      REQUEST_HEADER_BYTES,
      if (self.flags & 1) != 0 {
        self.input_count
      } else {
        self.input_count * 4
      },
      self.declared_length,
    ) catch {
      error => {
        self.fail(error)
        return
      }
    }
    self.stop_token_offset = self.cursor
    self.phase_index = -1
    return
  }
  if self.phase_index == -1 {
    self.cursor = request_append(
      self.cursor,
      self.stop_token_count * 4,
      self.declared_length,
    ) catch {
      error => {
        self.fail(error)
        return
      }
    }
    self.phase_index = 0
    return
  }
  if self.phase_index < self.stop_string_count {
    request_room(self.cursor, 4, self.declared_length) catch {
      error => {
        self.fail(error)
        return
      }
    }
    let item_length = read_u32(self.storage, self.cursor).reinterpret_as_int()
    if item_length <= 0 ||
      item_length > self.limits.inference_limits().max_stop_string_bytes() {
      self.fail_rule(StopTable)
      return
    }
    self.stop_string_offsets[self.phase_index] = self.cursor + 4
    self.stop_string_lengths[self.phase_index] = item_length
    self.cursor += 4
    self.cursor = request_append(self.cursor, item_length, self.declared_length) catch {
      error => {
        self.fail(error)
        return
      }
    }
    self.actual_stop_bytes += item_length
    self.phase_index += 1
    return
  }
  if self.phase_index == self.stop_string_count &&
    self.actual_stop_bytes != self.declared_stop_bytes {
    self.fail_rule(StopTable)
    return
  }
  if self.phase_index == self.stop_string_count {
    self.scope_offset = self.cursor
    self.cursor = request_append(
      self.cursor,
      self.scope_length,
      self.declared_length,
    ) catch {
      error => {
        self.fail(error)
        return
      }
    }
    self.phase_index += 1
    return
  }
  if self.phase_index == self.stop_string_count + 1 {
    self.trace_offset = self.cursor
    self.cursor = request_append(
      self.cursor,
      self.trace_length,
      self.declared_length,
    ) catch {
      error => {
        self.fail(error)
        return
      }
    }
    self.phase_index += 1
    return
  }
  if self.cursor != self.declared_length ||
    ((self.flags & 32) != 0) != (self.trace_length > 0) {
    self.fail_rule(CanonicalEncoding)
  } else {
    self.phase = LUNA_FRAMED_REQUEST_INPUT
    self.phase_index = 0
    self.reset_utf8()
  }
}

///|
fn LunaFramedRequestWorkspace::progress_input_one(
  self : LunaFramedRequestWorkspace,
) -> Unit {
  if self.phase_index < self.input_count {
    if (self.flags & 1) != 0 {
      if !self.advance_utf8(self.storage[self.input_offset + self.phase_index]) {
        self.fail_rule(Input)
        return
      }
    } else {
      let token = read_i32(
        self.storage,
        self.input_offset + self.phase_index * 4,
      )
      if token < 0 || token > self.limits.inference_limits().max_token_id() {
        self.fail_rule(Input)
        return
      }
    }
    self.phase_index += 1
    return
  }
  if (self.flags & 1) != 0 && self.utf8_remaining != 0 {
    self.fail_rule(Input)
    return
  }
  self.phase = LUNA_FRAMED_REQUEST_STOP_TOKENS
  self.phase_index = 0
}

///|
fn LunaFramedRequestWorkspace::progress_stop_token_one(
  self : LunaFramedRequestWorkspace,
) -> Unit {
  if self.phase_index < self.stop_token_count {
    let token = read_i32(
      self.storage,
      self.stop_token_offset + self.phase_index * 4,
    )
    if token < 0 || token > self.limits.inference_limits().max_token_id() {
      self.fail_rule(StopTable)
      return
    }
    self.phase_index += 1
    return
  }
  self.phase = LUNA_FRAMED_REQUEST_STOP_TOKEN_DUPLICATES
  self.phase_index = 0
  self.nested_index = 0
}

///|
fn LunaFramedRequestWorkspace::progress_stop_token_duplicate_one(
  self : LunaFramedRequestWorkspace,
) -> Unit {
  if self.phase_index >= self.stop_token_count {
    self.phase = LUNA_FRAMED_REQUEST_STOP_STRINGS
    self.phase_index = 0
    self.byte_index = 0
    self.reset_utf8()
    return
  }
  if self.nested_index >= self.phase_index {
    self.phase_index += 1
    self.nested_index = 0
    return
  }
  if read_i32(self.storage, self.stop_token_offset + self.phase_index * 4) ==
    read_i32(self.storage, self.stop_token_offset + self.nested_index * 4) {
    self.fail_rule(StopTable)
    return
  }
  self.nested_index += 1
}

///|
fn LunaFramedRequestWorkspace::progress_stop_string_one(
  self : LunaFramedRequestWorkspace,
) -> Unit {
  if self.phase_index >= self.stop_string_count {
    self.phase = LUNA_FRAMED_REQUEST_STOP_STRING_DUPLICATES
    self.phase_index = 0
    self.nested_index = 0
    self.byte_index = 0
    return
  }
  let length = self.stop_string_lengths[self.phase_index]
  if self.byte_index < length {
    let byte = self.storage[self.stop_string_offsets[self.phase_index] +
      self.byte_index]
    if !self.advance_utf8(byte) {
      self.fail_rule(StopTable)
      return
    }
    self.byte_index += 1
    return
  }
  if self.utf8_remaining != 0 {
    self.fail_rule(StopTable)
    return
  }
  self.phase_index += 1
  self.byte_index = 0
  self.reset_utf8()
}

///|
fn LunaFramedRequestWorkspace::progress_stop_string_duplicate_one(
  self : LunaFramedRequestWorkspace,
) -> Unit {
  if self.phase_index >= self.stop_string_count {
    self.phase = LUNA_FRAMED_REQUEST_SCOPE
    self.phase_index = 0
    return
  }
  if self.nested_index >= self.phase_index {
    self.phase_index += 1
    self.nested_index = 0
    self.byte_index = 0
    return
  }
  if self.stop_string_lengths[self.phase_index] !=
    self.stop_string_lengths[self.nested_index] {
    self.nested_index += 1
    self.byte_index = 0
    return
  }
  let length = self.stop_string_lengths[self.phase_index]
  if self.byte_index >= length {
    self.fail_rule(StopTable)
    return
  }
  if self.storage[self.stop_string_offsets[self.phase_index] + self.byte_index] !=
    self.storage[self.stop_string_offsets[self.nested_index] + self.byte_index] {
    self.nested_index += 1
    self.byte_index = 0
    return
  }
  self.byte_index += 1
}

///|
fn LunaFramedRequestWorkspace::progress_scope_one(
  self : LunaFramedRequestWorkspace,
) -> Unit {
  if self.phase_index < self.scope_length {
    if !luna_request_safe_identifier_byte(
        self.storage[self.scope_offset + self.phase_index],
      ) {
      self.fail_rule(Payload)
      return
    }
    self.phase_index += 1
    return
  }
  self.phase = LUNA_FRAMED_REQUEST_TRACE
  self.phase_index = 0
}

///|
fn LunaFramedRequestWorkspace::progress_trace_one(
  self : LunaFramedRequestWorkspace,
) -> Unit {
  if self.phase_index < self.trace_length {
    if !luna_request_safe_identifier_byte(
        self.storage[self.trace_offset + self.phase_index],
      ) {
      self.fail_rule(Payload)
      return
    }
    self.phase_index += 1
    return
  }
  self.phase = LUNA_FRAMED_REQUEST_READY
  self.phase_index = 0
}
