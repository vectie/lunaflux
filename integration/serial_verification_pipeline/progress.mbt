///|
pub async fn TwoRankGreedyVerification::progress(
  self : Self,
  local_now_millis~ : UInt64,
) -> Bool {
  guard !self.busy &&
    self.phase is (Preparing | Executing | Restoring | Committing | Releasing) else {
    raise @control.StageControlError::InvalidState
  }
  self.busy = true
  defer {
    self.busy = false
  }
  errdefer {
    self.phase = Failed
  }
  guard local_now_millis >= self.tick && local_now_millis < self.deadline else {
    raise @control.StageControlError::Failed
  }
  self.tick = local_now_millis
  match self.phase {
    Releasing => {
      if !self.remote.poll(local_now_millis~) {
        return false
      }
      self.phase = Ready
      true
    }
    Preparing => {
      if !self.local_ready {
        self.local_ready = self.ingress.inputs_ready()
      }
      if !self.replaying && self.local_ready {
        if !self.save_begun {
          self.local_state.save()
          self.save_begun = true
        }
        if !self.save_done {
          self.save_done = self.local_state.saved()
        }
      }
      if !self.remote_ready {
        self.remote_ready = self.remote.poll(local_now_millis~)
      }
      if self.local_ready &&
        self.remote_ready &&
        (self.replaying || self.save_done) {
        self.ingress.submit()
        let rows = self.ingress.live_rows()
        (self.begin_activation)(self.epoch, rows, local_now_millis)
        self.remote.run(local_now_millis~)
        self.local_ready = false
        self.remote_ready = false
        self.phase = Executing
      }
      false
    }
    Executing => {
      if !self.local_ready {
        self.local_ready = self.ingress.progress()
      }
      if !self.activation_done {
        self.activation_done = (self.poll_activation)(local_now_millis)
      }
      if !self.remote_ready {
        self.remote_ready = self.remote.poll(local_now_millis~)
      }
      if self.local_ready && self.activation_done && self.remote_ready {
        self.failed = self.ingress.execution_failed() ||
          self.remote.execution_failed()
        if !self.failed && !self.replaying {
          guard self.remote.output_rows() == self.live else {
            raise @control.StageControlError::InvalidMessage
          }
          // Copy before commit/rollback overwrites the protocol's receipt buffer.
          for i in 0..<self.live {
            self.outputs[i] = self.remote.sampled_token_at(i)
          }
          let decision = @greedy.verify_greedy_prefix(
            self.inputs[:self.live],
            self.outputs[:self.live],
            remaining_new_tokens=self.remaining,
            vocabulary_size=self.vocabulary,
            stop_tokens=self.stops,
          )
          self.decision = Some(decision)
          if decision.rollback_required() {
            self.remote.rollback(local_now_millis~)
            self.local_state.restore()
            self.remote_ready = false
            self.restore_done = false
            self.phase = Restoring
            return false
          }
        }
        self.remote.commit(pipeline_failed=self.failed, local_now_millis~)
        self.phase = Committing
      }
      false
    }
    Restoring => {
      if !self.restore_done {
        self.restore_done = self.local_state.restored()
      }
      if !self.remote_ready {
        self.remote_ready = self.remote.poll(local_now_millis~)
      }
      if self.restore_done && self.remote_ready {
        self.ingress.rollback_retired()
        self.save_begun = false
        self.save_done = false
        guard self.decision is Some(decision) else {
          raise @control.StageControlError::InvalidState
        }
        let original = self.original.current()
        let replay = self.replay.encode_prefix_replay(
          original,
          accepted_rows=decision.accepted_input_rows(),
          sequence_value=original.sequence_value() + 1UL,
        )
        self.ingress.begin(replay, self.completions.writer_for(replay))
        self.epoch += 1UL
        self.remote.prepare(replay, epoch=self.epoch, local_now_millis~)
        self.replaying = true
        self.local_ready = false
        self.remote_ready = false
        self.activation_done = false
        self.phase = Preparing
      }
      false
    }
    Committing => {
      if !self.remote.poll(local_now_millis~) {
        return false
      }
      self.ingress.commit(pipeline_failed=self.failed)
      if !self.replaying {
        self.local_state.keep()
        self.save_begun = false
        self.save_done = false
      }
      self.phase = if self.failed { Failed } else { Committed }
      true
    }
    _ => raise @control.StageControlError::InvalidState
  }
}
