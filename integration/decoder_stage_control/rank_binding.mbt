///|
/// Runtime effect adapter, not compiler IR. Canonical transport and commit
/// ordering are shared; the two rank types retain distinct history contracts.
priv enum StageRank {
  Serial(@rank.SerialDecoderRank)
  Batched(@batch_rank.BatchedDecoderRank)
}

///|
fn StageRank::begin(
  self : Self,
  plan : @wire.ValidatedPlanFrame,
  writer : @wire.CompletionFrameWriter,
) -> Unit raise {
  match self {
    Serial(r) => r.begin(plan, writer)
    Batched(r) => r.begin(plan, writer)
  }
}

///|
fn StageRank::inputs_ready(self : Self) -> Bool raise {
  match self {
    Serial(r) => r.inputs_ready()
    Batched(r) => r.inputs_ready()
  }
}

///|
fn StageRank::submit(self : Self) -> Unit raise {
  match self {
    Serial(r) => r.submit()
    Batched(r) => r.submit()
  }
}

///|
fn StageRank::progress(self : Self) -> Bool raise {
  match self {
    Serial(r) => r.progress()
    Batched(r) => r.progress()
  }
}

///|
fn StageRank::execution_failed(self : Self) -> Bool raise {
  match self {
    Serial(r) => r.execution_failed()
    Batched(r) => r.execution_failed()
  }
}

///|
fn StageRank::commit(self : Self, pipeline_failed~ : Bool) -> Unit raise {
  match self {
    Serial(r) => r.commit(pipeline_failed~)
    Batched(r) => r.commit(pipeline_failed~)
  }
}

///|
fn StageRank::live_rows(self : Self) -> Int raise {
  match self {
    Serial(r) => r.live_rows()
    Batched(r) => r.live_rows()
  }
}

///|
fn StageRank::output_rows(self : Self) -> Int raise {
  match self {
    Serial(r) => r.output_rows()
    Batched(r) => r.output_rows()
  }
}

///|
fn StageRank::produces_token(self : Self) -> Bool raise {
  match self {
    Serial(r) => r.produces_token()
    Batched(r) => r.output_rows() > 0
  }
}

///|
fn StageRank::sampled_token_at(self : Self, row : Int) -> Int raise {
  match self {
    Serial(r) => r.sampled_token_at(row)
    Batched(r) => {
      let samples = r.samples()
      guard row >= 0 && row < samples.length() else {
        raise StageControlError::InvalidState
      }
      samples[row]
    }
  }
}

///|
fn StageRank::sampled_token(self : Self) -> Int raise {
  self.sampled_token_at(0)
}

///|
fn StageRank::release_request(self : Self) -> Unit raise {
  match self {
    Serial(r) => r.release_request()
    Batched(_) => raise StageControlError::InvalidMessage
  }
}

///|
fn StageRank::release_one(
  self : Self,
  id : UInt64,
  generation : UInt64,
) -> Unit raise {
  match self {
    Serial(_) => raise StageControlError::InvalidMessage
    Batched(r) => r.retire_request(id, generation)
  }
}

///|
fn StageRank::rollback_retired(self : Self) -> Unit raise {
  match self {
    Serial(r) => r.rollback_retired()
    Batched(_) => raise StageControlError::InvalidMessage
  }
}
