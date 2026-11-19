///|
priv enum Driver {
  Text(@request.Runner)
  Image(@image_runtime.Execution)
}

///|
fn Driver::begin(
  self : Self,
  prompt : @prompt.PreparedTextPrompt,
  image : (@multimodal.ImagePrompt, @vision.Input)?,
  maximum_new_tokens~ : Int32,
  stop_token_ids~ : ReadOnlyArray[Int32],
  context_ceiling~ : Int32,
) -> Unit raise {
  match (self, image) {
    (Text(runner), None) =>
      runner.begin(
        prompt,
        maximum_new_tokens~,
        stop_token_ids~,
        context_ceiling~,
      )
    (Image(execution), None) =>
      execution.begin_text(
        prompt,
        maximum_new_tokens~,
        stop_token_ids~,
        context_ceiling~,
      )
    (Image(execution), Some((prompt, input))) =>
      execution.begin_image(
        prompt,
        input,
        maximum_new_tokens~,
        stop_token_ids~,
        context_ceiling~,
      )
    _ => raise @session.SessionChatError::UnsupportedRequest
  }
}

///|
fn Driver::submit_next(self : Self) -> Bool raise {
  match self {
    Text(runner) => runner.submit_next()
    Image(execution) => execution.submit_next()
  }
}

///|
fn Driver::poll(self : Self) -> Bool raise {
  match self {
    Text(runner) => runner.poll()
    Image(execution) => execution.poll()
  }
}

///|
fn Driver::cancel(self : Self) -> Unit raise {
  match self {
    Text(runner) => runner.cancel()
    Image(execution) => execution.cancel()
  }
}

///|
fn Driver::emitted_tokens(self : Self) -> ArrayView[Int32] {
  match self {
    Text(runner) => runner.emitted_tokens()
    Image(execution) => execution.emitted_tokens()
  }
}

///|
fn Driver::finish_reason(self : Self) -> @inference.FinishReason? {
  match self {
    Text(runner) => runner.finish_reason()
    Image(execution) => execution.finish_reason()
  }
}

///|
fn Driver::sampled_tokens(self : Self) -> Int32 {
  match self {
    Text(runner) => runner.sampled_tokens()
    Image(execution) => execution.sampled_tokens()
  }
}
