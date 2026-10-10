///|
fn state_transaction(
  owner : @snapshot.DeviceStateSnapshot,
) -> @control.StageStateTransaction {
  @control.StageStateTransaction::new(
    begin_save=() => owner.begin(),
    poll_save=() => owner.poll_completion(),
    commit=() => owner.commit(),
    begin_restore=() => owner.rollback(),
    poll_restore=() => owner.poll_completion(),
  )
}

///|
fn prepare_verification_session(
  startup : Startup,
  rank : @worker.SerialDecoderRank,
  client : @control.DecoderStageClient,
  stage : @deepseek.DeepSeekDecoderStage,
  edge : @deepseek.DeepSeekActivationEdge,
  backup : @snapshot.DeviceStateSnapshot,
  stop~ : Int,
) -> @verification.TwoRankGreedyVerification raise {
  @verification.TwoRankGreedyVerification::new(
    rank,
    client,
    worker_limits(startup),
    inference_limits(startup),
    local_state=state_transaction(backup),
    close_local_state=() => {
      stage.drain()
      backup.close_after_retirement()
    },
    begin_activation=(epoch, rows, now) => {
      edge.begin(epoch~, live_rows=rows, local_now_millis=now)
    },
    poll_activation=now => edge.poll(local_now_millis=now),
    close_activation=() => edge.close(),
    stop_tokens=if stop == -1 { [] } else { [stop] },
  )
}

///|
// The caller owns the resident verifier. A request retires its KV/frontier
// state, never its model, snapshot, connection or activation resources.
async fn generate_speculative(
  startup : Startup,
  owner : @generation.SpeculativeGreedyGeneration,
  client : @control.DecoderStageClient,
  verifier : @verification.TwoRankGreedyVerification,
  clock : @clock.MonotonicClock,
  first_epoch~ : UInt64,
) -> UInt64 {
  let proposal = FixedArray::make(startup.rows, 0)
  let target = FixedArray::make(startup.rows, 0)
  let mut epoch = first_epoch
  let mut blocks = 0
  let mut accepted = 0
  let mut submitted = 0
  let mut replays = 0
  while owner.finish_reason() is Running {
    client.predict(
      local_now_millis=clock.now_millis(),
      committed_seed=owner.token_at(owner.token_count() - 1),
    )
    while !client.poll(local_now_millis=clock.now_millis()) {
      @async.pause()
    }
    let count = client.draft_count()
    for i in 0..<count {
      proposal[i] = client.draft_token_at(i)
    }
    let plan = owner.next_verification(proposal[:count], sequence_value=epoch)
    submitted += plan.token_count()
    verifier.begin(
      plan,
      epoch~,
      remaining_new_tokens=owner.remaining_new_tokens(),
      local_now_millis=clock.now_millis(),
    )
    while !verifier.progress(local_now_millis=clock.now_millis()) {
      @async.pause()
    }
    let receipt = verifier.result()
    if receipt.rollback_required() {
      replays += 1
    }
    let rows = receipt.emitted_token_count()
    for i in 0..<rows {
      target[i] = verifier.token_at(i)
    }
    owner.record_commit(receipt, target[:rows])
    blocks += 1
    accepted += rows
    epoch = verifier.last_epoch() + 1UL
  }
  verifier.release_request(local_now_millis=clock.now_millis())
  while !verifier.progress(local_now_millis=clock.now_millis()) {
    @async.pause()
  }
  // Render only after both request owners retire, not during block submission.
  println(
    "dspark_verified_blocks=\{blocks} committed_input_rows=\{accepted} submitted_input_rows=\{submitted} accepted_prefix_replays=\{replays}",
  )
  epoch
}
