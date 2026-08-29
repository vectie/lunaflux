///|
fn credential_error(
  rule : InheritedInferenceCredentialRule,
) -> InheritedInferenceCredentialError {
  InheritedInferenceCredentialFailed(rule)
}

///|
pub fn prepare_optional_inherited_inference_credential_v1() -> InheritedInferenceCredentialOwner? raise InheritedInferenceCredentialError {
  // Allocate the complete bounded secret workspace before moving FD 6.
  let header = FixedArray::make(
    INHERITED_INFERENCE_CREDENTIAL_HEADER_BYTES,
    b'\x00',
  )
  let credential = FixedArray::make(
    INHERITED_INFERENCE_CREDENTIAL_MAX_BYTES,
    b'\x00',
  )
  let native = @native.open_fixed()
  match @native.open_status(native) {
    0 =>
      Some({
        native,
        header,
        credential,
        header_cursor: 0,
        credential_cursor: 0,
        credential_length: 0,
        phase: 0,
        failure_rule: 0,
      })
    1 => None
    _ => raise credential_error(CredentialDescriptorInvalid)
  }
}

///|
fn InheritedInferenceCredentialOwner::fail_closed(
  self : InheritedInferenceCredentialOwner,
  rule : InheritedInferenceCredentialRule,
) -> InheritedInferenceCredentialProgress {
  ignore(@native.close(self.native))
  wipe_credential_bytes(self.header, self.credential)
  self.phase = 3
  self.failure_rule = match rule {
    CredentialDescriptorUnavailable => 1
    CredentialDescriptorInvalid => 2
    CredentialFrameInvalid => 3
    CredentialChannelLost => 4
    CredentialStateInvalid => 5
    CredentialPolicyInvalid => 6
  }
  InheritedInferenceCredentialFailedProgress
}

///|
fn InheritedInferenceCredentialOwner::latched_error(
  self : InheritedInferenceCredentialOwner,
) -> InheritedInferenceCredentialError {
  credential_error(
    match self.failure_rule {
      3 => CredentialFrameInvalid
      4 => CredentialChannelLost
      6 => CredentialPolicyInvalid
      _ => CredentialStateInvalid
    },
  )
}

///|
fn InheritedInferenceCredentialOwner::read_header(
  self : InheritedInferenceCredentialOwner,
) -> InheritedInferenceCredentialProgress {
  let remaining = INHERITED_INFERENCE_CREDENTIAL_HEADER_BYTES -
    self.header_cursor
  let count = @native.try_read(
    self.native,
    self.header,
    self.header_cursor,
    remaining,
  )
  if count > 0 && count <= remaining {
    self.header_cursor += count
  } else if count == 0 {
    return InheritedInferenceCredentialPending
  } else {
    return self.fail_closed(CredentialChannelLost)
  }
  if self.header_cursor != INHERITED_INFERENCE_CREDENTIAL_HEADER_BYTES {
    return InheritedInferenceCredentialPending
  }
  self.credential_length = inference_credential_header_length(self.header)
  if self.credential_length == 0 {
    return self.fail_closed(CredentialFrameInvalid)
  }
  self.phase = 1
  InheritedInferenceCredentialPending
}

///|
fn InheritedInferenceCredentialOwner::read_credential(
  self : InheritedInferenceCredentialOwner,
) -> InheritedInferenceCredentialProgress {
  if self.credential_cursor < self.credential_length {
    let remaining = self.credential_length - self.credential_cursor
    let count = @native.try_read(
      self.native,
      self.credential,
      self.credential_cursor,
      remaining,
    )
    if count > 0 && count <= remaining {
      self.credential_cursor += count
    } else if count == 0 {
      return InheritedInferenceCredentialPending
    } else {
      return self.fail_closed(CredentialChannelLost)
    }
  }
  if self.credential_cursor != self.credential_length {
    return InheritedInferenceCredentialPending
  }
  match @native.peek(self.native) {
    -3 => {
      if @native.close(self.native) != 0 {
        return self.fail_closed(CredentialChannelLost)
      }
      self.phase = 2
      InheritedInferenceCredentialReady
    }
    0 => InheritedInferenceCredentialPending
    _ => self.fail_closed(CredentialFrameInvalid)
  }
}

///|
pub fn InheritedInferenceCredentialOwner::progress(
  self : InheritedInferenceCredentialOwner,
) -> InheritedInferenceCredentialProgress {
  match self.phase {
    0 => self.read_header()
    1 => self.read_credential()
    2 => InheritedInferenceCredentialReady
    _ => InheritedInferenceCredentialFailedProgress
  }
}

///|
pub fn InheritedInferenceCredentialOwner::take_startup_auth_policy_v1(
  self : InheritedInferenceCredentialOwner,
  maximum_credential_bytes~ : Int,
) -> InheritedInferenceAuthPolicy raise InheritedInferenceCredentialError {
  if self.phase != 2 ||
    !@native.is_closed(self.native) ||
    maximum_credential_bytes <= 0 ||
    maximum_credential_bytes > INHERITED_INFERENCE_CREDENTIAL_MAX_BYTES ||
    self.credential_length > maximum_credential_bytes {
    if self.phase == 3 {
      raise self.latched_error()
    }
    raise credential_error(CredentialStateInvalid)
  }
  let result : Result[@api_auth.LunaApiAuthPolicy, @api_auth.LunaApiAuthError] = Ok(
    @api_auth.LunaApiAuthPolicy::new(
      self.credential,
      source_offset=0,
      length=self.credential_length,
      maximum_credential_bytes~,
    ),
  ) catch {
    error => Err(error)
  }
  wipe_credential_bytes(self.header, self.credential)
  self.phase = 4
  match result {
    Ok(policy) =>
      {
        policy,
        credential_length: self.credential_length,
        maximum_credential_bytes,
        consumed: false,
      }
    Err(_) => raise credential_error(CredentialPolicyInvalid)
  }
}

///|
/// Consumes the opaque startup policy only when the authenticated instance
/// policy admits its actual source length. The scalar bound is not a secret;
/// neither the actual length nor credential bytes are exposed.
pub fn InheritedInferenceAuthPolicy::take_for_policy(
  self : InheritedInferenceAuthPolicy,
  maximum_credential_bytes : Int,
) -> @api_auth.LunaApiAuthPolicy raise InheritedInferenceCredentialError {
  if self.consumed ||
    maximum_credential_bytes <= 0 ||
    maximum_credential_bytes > self.maximum_credential_bytes ||
    self.credential_length > maximum_credential_bytes {
    raise credential_error(CredentialPolicyInvalid)
  }
  self.consumed = true
  self.policy
}

///|
/// Idempotently wipes an unconsumed or retained startup authentication policy.
/// This exposes no credential state and also invalidates every stale alias.
pub fn InheritedInferenceAuthPolicy::close(
  self : InheritedInferenceAuthPolicy,
) -> Unit {
  self.policy.close()
  self.consumed = true
}

///|
pub fn InheritedInferenceCredentialOwner::close(
  self : InheritedInferenceCredentialOwner,
) -> Unit raise InheritedInferenceCredentialError {
  if self.phase == 4 {
    return
  }
  wipe_credential_bytes(self.header, self.credential)
  self.phase = 4
  if @native.close(self.native) != 0 {
    raise credential_error(CredentialChannelLost)
  }
}
