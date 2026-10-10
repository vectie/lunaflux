///|
// Parse once, before checkpoint access or device preparation. The joint command
// currently runs one request; its steps are synchronous prepared pipeline frames.
fn joint_cancellation_arguments(
  arguments : ArrayView[String],
) -> (ArrayView[String], Int) raise {
  let (plain, cancellation) = @prompt_file.split_request_cancellation_options(
    arguments,
  )
  match cancellation {
    None => (plain, 0)
    Some(plan) => {
      guard plain is [_, "run-joint", _, ..] else {
        fail("request cancellation is supported only by run-joint")
      }
      plan.validate_request_count(1)
      (plain, plan.after_steps_for(0))
    }
  }
}

///|
// Effects stay explicit at the command boundary. advance completes one prepared
// frame before returning. There is no hidden progress/poll limit and no output
// publication here. A cancelled request must drain/close before it is reported.
fn advance_joint_frames(
  advance : () -> @execution.PipelineStage raise,
  cancel : () -> Unit,
  close : () -> Unit raise,
  cancel_after_frames : Int,
) -> (@execution.PipelineStage, Int) raise {
  guard cancel_after_frames >= 0 else {
    fail("invalid cancellation frame count")
  }
  for completed = 0 {
    if cancel_after_frames > 0 && completed == cancel_after_frames {
      cancel()
      close()
      return (Cancelled, completed)
    }
    match advance() {
      Completed => return (Completed, completed + 1)
      Cancelled => {
        close()
        return (Cancelled, completed)
      }
      Failed | Closed => fail("joint request failed before completion")
      Conditioning | Denoising | VideoDecode | AudioDecode =>
        continue completed + 1
    }
  }
}
