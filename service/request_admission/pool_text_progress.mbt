///|
fn LunaRequestPreparationPool::progress_text_input_copy(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let write = match lane.tokenizer_input_write {
    Some(value) => value
    None => abort("Luna tokenizer-input invariant")
  }
  let text = match lane.require_request().input() {
    Text(value) => value
    TokenIds(_) => abort("Luna tokenizer-input source invariant")
  }
  if lane.copy_index < text.utf8_bytes() {
    write.push_byte(text.utf8()[lane.copy_index]) catch {
      _ => raise Invalid(Tokenization)
    }
    lane.copy_index += 1
    return
  }
  lane.tokenizer_work = Some(
    write.finish() catch {
      _ => raise Invalid(Tokenization)
    },
  )
  lane.tokenizer_input_write = None
  lane.copy_index = 0
  lane.phase = LUNA_PREPARATION_TEXT_TOKENIZE
}

///|
fn LunaRequestPreparationPool::progress_text_tokenize(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let work = match lane.tokenizer_work {
    Some(value) => value
    None => abort("Luna tokenizer-work invariant")
  }
  let progress = work.progress() catch {
    SpecialTokenRejected(token_id=_) | OutputTooLong(actual=_, maximum=_) => {
      // These terminal tokenizer failures clear the worker before raising, so
      // the retained Work alias is already stale and must not be aborted.
      lane.tokenizer_work = None
      raise Invalid(Tokenization)
    }
    _ => raise Invalid(Tokenization)
  }
  match progress {
    LunaTokenizerPending => ()
    LunaTokenizerReady => lane.phase = LUNA_PREPARATION_TEXT_BEGIN_COPY
  }
}

///|
fn LunaRequestPreparationPool::progress_text_begin_copy(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let work = match lane.tokenizer_work {
    Some(value) => value
    None => abort("Luna tokenizer-copy invariant")
  }
  let count = work.token_count() catch { _ => raise Invalid(Tokenization) }
  if count <= 0 || count > lane.input_capacity {
    raise Invalid(Input)
  }
  lane.token_write = Some(
    lane.token_storage.begin(token_count=count) catch {
      _ => raise Invalid(Input)
    },
  )
  lane.copy_index = 0
  lane.phase = LUNA_PREPARATION_TEXT_COPY
}

///|
fn LunaRequestPreparationPool::progress_text_copy(
  self : LunaRequestPreparationPool,
  lane : LunaRequestPreparationLane,
) -> Unit raise RequestAdmissionError {
  ignore(self)
  let tokenizer_work = match lane.tokenizer_work {
    Some(value) => value
    None => abort("Luna tokenizer-copy work invariant")
  }
  let write = match lane.token_write {
    Some(value) => value
    None => abort("Luna token-write invariant")
  }
  let count = tokenizer_work.token_count() catch {
    _ => raise Invalid(Tokenization)
  }
  if lane.copy_index < count {
    let status = tokenizer_work.token_status(lane.copy_index) catch {
      _ => raise Invalid(Tokenization)
    }
    if !status.is_ready() {
      raise Invalid(Tokenization)
    }
    write.push(status.token_id()) catch {
      _ => raise Invalid(Input)
    }
    lane.copy_index += 1
    return
  }
  let lease = write.seal() catch { _ => raise Invalid(Input) }
  lane.token_write = None
  lane.token_lease = Some(lease)
  lane.tokens = Some(try! lease.token_buffer())
  tokenizer_work.abort() catch {
    _ => raise Invalid(Tokenization)
  }
  lane.tokenizer_work = None
  if lane.typed_direct {
    lane.output_work = Some(
      lane.output_workspace.begin_luna_request_semantics(
        lane.require_semantic_lease().view() catch {
          _ => raise Invalid(RequestContract)
        },
        special_policy=Skip,
      ) catch {
        _ => raise Invalid(OutputContract)
      },
    )
    lane.phase = LUNA_PREPARATION_OUTPUT_SETUP
  } else if lane.framed_direct {
    lane.phase = LUNA_PREPARATION_SEMANTIC_VALIDATE
  } else {
    lane.begin_semantic_import()
  }
}
