///|
fn LunaHttp1Workspace::progress_auth_begin_one(
  self : LunaHttp1Workspace,
) -> Unit {
  let write = self.auth_verifier.begin_verify(self.credential_length) catch {
    _ => {
      self.fail(LunaHttp1Authorization, LunaHttp1Invalid)
      return
    }
  }
  if self.auth_writes.length() != 0 || self.auth_works.length() != 0 {
    abort("Luna HTTP/1 authentication begin invariant")
  }
  self.auth_writes.push(write)
  self.auth_cursor = 0
  self.phase = LUNA_HTTP1_AUTH_WRITE
}

///|
fn LunaHttp1Workspace::progress_auth_write_one(
  self : LunaHttp1Workspace,
) -> Unit {
  if self.auth_writes.length() != 1 || self.auth_works.length() != 0 {
    abort("Luna HTTP/1 authentication write invariant")
  }
  if self.auth_cursor >= self.credential_length {
    abort("Luna HTTP/1 authentication credential invariant")
  }
  let result = self.auth_writes[0].push_byte(
    self.storage[self.credential_offset + self.auth_cursor],
  ) catch {
    _ => {
      self.fail(LunaHttp1Authorization, LunaHttp1Invalid)
      return
    }
  }
  if result.work_units() != 1 {
    abort("Luna HTTP/1 authentication work-unit invariant")
  }
  self.auth_cursor += 1
  if result.is_complete() {
    self.phase = LUNA_HTTP1_AUTH_FINISH
  } else if !result.is_accepting() {
    abort("Luna HTTP/1 authentication progress invariant")
  }
}

///|
fn LunaHttp1Workspace::progress_auth_finish_one(
  self : LunaHttp1Workspace,
) -> Unit {
  if self.auth_writes.length() != 1 || self.auth_works.length() != 0 {
    abort("Luna HTTP/1 authentication finish invariant")
  }
  let work = self.auth_writes[0].finish() catch {
    _ => {
      self.fail(LunaHttp1Authorization, LunaHttp1Invalid)
      return
    }
  }
  self.auth_writes.clear()
  self.auth_works.push(work)
  self.phase = LUNA_HTTP1_AUTH_PROGRESS
}

///|
fn LunaHttp1Workspace::progress_auth_work_one(
  self : LunaHttp1Workspace,
) -> Unit {
  if self.auth_writes.length() != 0 || self.auth_works.length() != 1 {
    abort("Luna HTTP/1 authentication work invariant")
  }
  ignore(self.auth_works[0].progress()) catch {
    _ => {
      self.fail(LunaHttp1Authorization, LunaHttp1Invalid)
      return
    }
  }
  let units = self.auth_works[0].last_work_units() catch {
    _ => {
      self.fail(LunaHttp1Authorization, LunaHttp1Invalid)
      return
    }
  }
  if units != 1 {
    abort("Luna HTTP/1 child-quantum invariant")
  }
  let state = self.auth_works[0].state() catch {
    _ => {
      self.fail(LunaHttp1Authorization, LunaHttp1Invalid)
      return
    }
  }
  match state {
    LunaApiAuthVerificationPending => ()
    LunaApiAuthVerificationAccepted | LunaApiAuthVerificationRejected =>
      self.phase = LUNA_HTTP1_AUTH_DECIDE
    LunaApiAuthVerificationFailed =>
      self.fail(LunaHttp1Authorization, LunaHttp1Invalid)
  }
}

///|
fn LunaHttp1Workspace::progress_auth_decide_one(
  self : LunaHttp1Workspace,
) -> Unit {
  if self.auth_writes.length() != 0 || self.auth_works.length() != 1 {
    abort("Luna HTTP/1 authentication decision invariant")
  }
  let decision = self.auth_works[0].take_decision() catch {
    _ => {
      self.fail(LunaHttp1Authorization, LunaHttp1Invalid)
      return
    }
  }
  self.auth_works.clear()
  self.wipe_credential()
  if decision.is_rejected() {
    self.fail(LunaHttp1Authorization, LunaHttp1Rejected)
    return
  }
  if !decision.is_accepted() {
    abort("Luna HTTP/1 authentication decision invariant")
  }
  if self.body_length == 0 {
    self.phase = LUNA_HTTP1_FINALIZING
  } else if self.input_ended {
    self.fail(LunaHttp1Body, LunaHttp1PrematureEnd)
  } else {
    self.phase = LUNA_HTTP1_RECEIVING_BODY
  }
}
