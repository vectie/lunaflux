///|
fn luna_openai_semantic_token_id(
  event : @luna_event.LunaEventView,
) -> Int raise LunaOpenAICompatError {
  let token = event.token() catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
  }
  token.token_id() catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
  }
}

///|
fn luna_openai_semantic_position(
  event : @luna_event.LunaEventView,
) -> Int raise LunaOpenAICompatError {
  let token = event.token() catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
  }
  token.position() catch {
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
  }
}

///|
fn LunaOpenAICompatWorkspace::semantic_payload_length(
  self : LunaOpenAICompatWorkspace,
  event : @luna_event.LunaEventView,
) -> Int raise LunaOpenAICompatError {
  match self.event_kind {
    LUNA_OPENAI_EVENT_TOKEN => {
      let token = event.token() catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
      }
      token.delta_length() catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
      }
    }
    LUNA_OPENAI_EVENT_COMPLETED => {
      let completed = event.completed() catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
      }
      completed.final_delta_length() catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
      }
    }
    LUNA_OPENAI_EVENT_FAILED => {
      let failed = event.failed() catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
      }
      failed.code_length() catch {
        _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
      }
    }
    _ => abort("Luna OpenAI compatibility payload-kind invariant")
  }
}

///|
fn luna_openai_valid_usage(
  inference : @inference.InferenceLimits,
  input : Int,
  cached : Int,
  output : Int,
  total : Int,
) -> Bool {
  input >= 0 &&
  input <= inference.max_context_tokens() &&
  cached >= 0 &&
  cached <= input &&
  output >= 0 &&
  output <= inference.max_new_tokens() &&
  input <= inference.max_context_tokens() - output &&
  total == input + output
}

///|
fn LunaOpenAICompatWorkspace::finish_setup(
  self : LunaOpenAICompatWorkspace,
) -> Unit {
  self.setup_cursor = 0
  self.part_index = 0
  self.phase = LUNA_OPENAI_COMPAT_CONFIGURE_PART
}

///|
fn LunaOpenAICompatWorkspace::setup_accepted(
  self : LunaOpenAICompatWorkspace,
  event : @luna_event.LunaEventView,
) -> Unit raise LunaOpenAICompatError {
  ignore(
    event.accepted() catch {
      _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
    },
  )
  self.finish_setup()
}

///|
fn LunaOpenAICompatWorkspace::setup_token(
  self : LunaOpenAICompatWorkspace,
  event : @luna_event.LunaEventView,
) -> Unit raise LunaOpenAICompatError {
  match self.setup_cursor {
    0 => {
      self.token_id = luna_openai_semantic_token_id(event)
      if self.token_id < 0 || self.token_id > self.inference.max_token_id() {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatScalar)
      }
      self.setup_cursor = 1
    }
    1 => {
      self.position = luna_openai_semantic_position(event)
      if self.position < 0 || self.position >= self.inference.max_new_tokens() {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatScalar)
      }
      self.setup_cursor = 2
    }
    _ => {
      self.payload_length = self.semantic_payload_length(event)
      if self.payload_length < 0 ||
        self.payload_length > self.inference.max_decoded_delta_bytes() {
        raise LunaOpenAICompatInvalid(LunaOpenAICompatPayload)
      }
      self.finish_setup()
    }
  }
}

///|
fn LunaOpenAICompatWorkspace::setup_usage_scalar(
  self : LunaOpenAICompatWorkspace,
  event : @luna_event.LunaEventView,
  completed : Bool,
) -> Unit raise LunaOpenAICompatError {
  if completed {
    let value = event.completed() catch {
      _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
    }
    match self.setup_cursor {
      1 =>
        self.input_tokens = value.input_tokens() catch {
          _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
        }
      2 =>
        self.cached_input_tokens = value.cached_input_tokens() catch {
          _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
        }
      3 =>
        self.output_tokens = value.output_tokens() catch {
          _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
        }
      _ =>
        self.total_tokens = value.total_tokens() catch {
          _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
        }
    }
  } else {
    let value = event.usage() catch {
      _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
    }
    match self.setup_cursor {
      0 =>
        self.input_tokens = value.input_tokens() catch {
          _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
        }
      1 =>
        self.cached_input_tokens = value.cached_input_tokens() catch {
          _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
        }
      2 =>
        self.output_tokens = value.output_tokens() catch {
          _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
        }
      _ =>
        self.total_tokens = value.total_tokens() catch {
          _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
        }
    }
  }
}

///|
fn LunaOpenAICompatWorkspace::setup_usage(
  self : LunaOpenAICompatWorkspace,
  event : @luna_event.LunaEventView,
) -> Unit raise LunaOpenAICompatError {
  self.setup_usage_scalar(event, false)
  if self.setup_cursor == 3 {
    if !luna_openai_valid_usage(
        self.inference,
        self.input_tokens,
        self.cached_input_tokens,
        self.output_tokens,
        self.total_tokens,
      ) {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatScalar)
    }
    self.finish_setup()
  } else {
    self.setup_cursor += 1
  }
}

///|
fn LunaOpenAICompatWorkspace::setup_completed(
  self : LunaOpenAICompatWorkspace,
  event : @luna_event.LunaEventView,
) -> Unit raise LunaOpenAICompatError {
  if self.setup_cursor == 0 {
    let completed = event.completed() catch {
      _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
    }
    let reason = completed.finish_reason() catch {
      _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
    }
    self.finish_reason = match reason {
      StopSequence => 0
      TokenLimit => 1
      Cancelled => 2
    }
    self.setup_cursor = 1
  } else if self.setup_cursor <= 4 {
    self.setup_usage_scalar(event, true)
    if self.setup_cursor == 4 &&
      !luna_openai_valid_usage(
        self.inference,
        self.input_tokens,
        self.cached_input_tokens,
        self.output_tokens,
        self.total_tokens,
      ) {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatScalar)
    }
    self.setup_cursor += 1
  } else {
    self.payload_length = self.semantic_payload_length(event)
    if self.payload_length < 0 ||
      self.payload_length > self.inference.max_decoded_delta_bytes() {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatPayload)
    }
    self.finish_setup()
  }
}

///|
fn LunaOpenAICompatWorkspace::setup_failed(
  self : LunaOpenAICompatWorkspace,
  event : @luna_event.LunaEventView,
) -> Unit raise LunaOpenAICompatError {
  if self.setup_cursor == 0 {
    let failed = event.failed() catch {
      _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
    }
    self.retryable = failed.retryable() catch {
      _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
    }
    self.setup_cursor = 1
  } else {
    self.payload_length = self.semantic_payload_length(event)
    if self.payload_length <= 0 ||
      self.payload_length > LUNA_OPENAI_COMPAT_MAX_ERROR_CODE_BYTES {
      raise LunaOpenAICompatInvalid(LunaOpenAICompatPayload)
    }
    self.finish_setup()
  }
}

///|
fn LunaOpenAICompatWorkspace::progress_setup(
  self : LunaOpenAICompatWorkspace,
  event : @luna_event.LunaEventView,
) -> Unit raise LunaOpenAICompatError {
  match self.event_kind {
    LUNA_OPENAI_EVENT_ACCEPTED => self.setup_accepted(event)
    LUNA_OPENAI_EVENT_TOKEN => self.setup_token(event)
    LUNA_OPENAI_EVENT_USAGE => self.setup_usage(event)
    LUNA_OPENAI_EVENT_COMPLETED => self.setup_completed(event)
    LUNA_OPENAI_EVENT_FAILED => self.setup_failed(event)
    _ => raise LunaOpenAICompatInvalid(LunaOpenAICompatSemanticEvent)
  }
}
