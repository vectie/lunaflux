///|
/// Rank-local request ports and execution effects. A partial rank never owns
/// model completion. Its metadata uploads retire before submission, and its
/// retained sequence advances only after a whole-pipeline commit.
pub struct SerialDecoderRank {
  priv worker : SerialDecoderWorker
}

///|
pub fn SerialDecoderRank::new(
  frame : @frame.SerialDecoderFrame,
  outputs_token~ : Bool,
) -> Self {
  let (_, _, vocabulary) = frame.geometry()
  {
    worker: {
      frame,
      inputs: [],
      pools: [],
      payloads: [],
      sampled: None,
      effects: None,
      whole_model: false,
      outputs_token,
      vocabulary,
      stage_failed: false,
      stage_token: 0,
      state: Fresh,
    },
  }
}

///|
/// Startup-only borrowed ports for the existing model stage preparation API.
/// The rank retains their ownership; whole-model execution APIs reject them.
pub fn SerialDecoderRank::ports(self : Self) -> SerialDecoderWorker {
  self.worker
}

///|
pub fn SerialDecoderRank::outputs_token(self : Self) -> Bool {
  self.worker.outputs_token
}

///|
pub fn SerialDecoderRank::produces_token(self : Self) -> Bool raise {
  self.worker.outputs_token && self.worker.frame.produces_token()
}

///|
pub fn SerialDecoderRank::live_rows(self : Self) -> Int raise {
  self.worker.frame.live_rows()
}

///|
pub fn SerialDecoderRank::bind_executor(
  self : Self,
  context : @device.Context,
  submit~ : (Int) -> Unit raise,
  poll~ : () -> Bool raise,
  close~ : () -> Unit raise,
  append_descriptors~ : ArrayView[@device.Allocation],
) -> Unit raise {
  self.worker.bind_effects(context, submit~, poll~, close~, append_descriptors~)
}

///|
/// Stage the same validated wire plan on every rank. The writer is rank-local
/// lease storage, not a response to the caller; commit aborts it without output.
pub fn SerialDecoderRank::begin(
  self : Self,
  plan : @wire.ValidatedPlanFrame,
  writer : @wire.CompletionFrameWriter,
) -> Unit raise {
  self.worker.begin_inputs(plan, writer)
}

///|
/// Poll metadata DMA only. It never submits a model queue. The caller may now
/// wait for incoming activation completion before invoking submit().
pub fn SerialDecoderRank::inputs_ready(self : Self) -> Bool raise {
  let worker = self.worker
  match worker.state {
    InputsReady => true
    Uploading => {
      worker.state = Failed
      if worker.uploads_done() {
        worker.state = InputsReady
        true
      } else {
        worker.state = Uploading
        false
      }
    }
    _ => raise @frame.SerialDecoderError::Busy
  }
}

///|
/// Incoming activation must be fully uploaded first. A stage-specific owner
/// enforces its activation lease in the bound submit effect.
pub fn SerialDecoderRank::submit(self : Self) -> Unit raise {
  let worker = self.worker
  guard worker.state is InputsReady && worker.effects is Some(effects) else {
    raise @frame.SerialDecoderError::Busy
  }
  worker.state = Failed
  (effects.submit)(worker.frame.live_rows())
  worker.state = Executing
}

///|
fn SerialDecoderWorker::invalid_samples(
  self : Self,
  sample_port : Int,
) -> Bool raise {
  if !self.outputs_token {
    return false
  }
  for row in 0..<self.frame.output_rows() {
    let value = load_i32(self.payloads[sample_port], row~)
    if value < 0 || value >= self.vocabulary {
      return true
    }
  }
  false
}

///|
/// Queue and execution-error/sample DMA retirement, not pipeline completion.
/// Egress validates vocabulary bounds. Ingress/interior never read token output.
pub fn SerialDecoderRank::progress(self : Self) -> Bool raise {
  let worker = self.worker
  guard worker.effects is Some(effects) else {
    raise @frame.SerialDecoderError::Busy
  }
  match worker.state {
    Executing => {
      worker.state = Failed
      if !(effects.poll)() {
        worker.state = Executing
        return false
      }
      worker.download()
      worker.state = Downloading
      false
    }
    Downloading => {
      worker.state = Failed
      if !worker.downloads_done() {
        worker.state = Downloading
        return false
      }
      let sample_port = worker.frame.input_count()
      let failed = worker.payloads[sample_port + 1:].any(bytes => {
        load_i32(bytes) != 0
      })
      let token = load_i32(worker.payloads[sample_port])
      worker.stage_token = token
      worker.stage_failed = failed || worker.invalid_samples(sample_port)
      worker.state = Retired
      true
    }
    Retired => true
    _ => raise @frame.SerialDecoderError::Busy
  }
}

///|
pub fn SerialDecoderRank::execution_failed(self : Self) -> Bool raise {
  guard self.worker.state is Retired else {
    raise @frame.SerialDecoderError::Busy
  }
  self.worker.stage_failed
}

///|
pub fn SerialDecoderRank::sampled_token(self : Self) -> Int raise {
  let worker = self.worker
  guard worker.state is Retired &&
    !worker.stage_failed &&
    worker.outputs_token &&
    worker.frame.output_rows() == 1 else {
    raise @frame.SerialDecoderError::Busy
  }
  worker.stage_token
}

///|
/// Retired all-position samples remain in startup-owned host storage until
/// commit/rollback. This scalar accessor neither allocates nor downloads again.
pub fn SerialDecoderRank::sampled_token_at(self : Self, row : Int) -> Int raise {
  let worker = self.worker
  guard worker.state is Retired &&
    !worker.stage_failed &&
    worker.outputs_token &&
    row >= 0 &&
    row < worker.frame.output_rows() else {
    raise @frame.SerialDecoderError::Busy
  }
  load_i32(worker.payloads[worker.frame.input_count()], row~)
}

///|
pub fn SerialDecoderRank::output_rows(self : Self) -> Int raise {
  guard self.worker.state is Retired else {
    raise @frame.SerialDecoderError::Busy
  }
  if self.worker.outputs_token {
    self.worker.frame.output_rows()
  } else {
    0
  }
}

///|
/// Call only after the model-state snapshot has restored and retired. Rejected
/// draft execution is not a model failure. Actual local/peer failure must go
/// through commit(pipeline_failed=true) and cannot return a reusable frontier.
pub fn SerialDecoderRank::rollback_retired(self : Self) -> Unit raise {
  let worker = self.worker
  guard worker.state is Retired && !worker.stage_failed else {
    raise @frame.SerialDecoderError::Busy
  }
  worker.state = Failed
  worker.frame.discard_stage()
  worker.state = Ready
}

///|
/// Commit only after all stage results and activation transfers retire. Any
/// local/peer failure terminally poisons history. No canonical completion is
/// emitted here; the coordinator owns its independent completion frame.
pub fn SerialDecoderRank::commit(
  self : Self,
  pipeline_failed~ : Bool,
) -> Unit raise {
  let worker = self.worker
  guard worker.state is Retired else { raise @frame.SerialDecoderError::Busy }
  worker.state = Failed
  worker.frame.retire_stage(
    execution_failed=pipeline_failed || worker.stage_failed,
  )
  if !worker.frame.is_faulted() {
    worker.state = Ready
  }
}

///|
pub fn SerialDecoderRank::release_request(self : Self) -> Unit raise {
  self.worker.release_request()
}

///|
pub fn SerialDecoderRank::close(self : Self) -> Unit raise {
  self.worker.close()
}
