///|
priv struct TokenSample {
  position : Int32
  input_token : Int32
  output_token : Int32
  host_us : Double
  dispatch_us : Double
  logits_us : Double
  snapshots : Array[CycleSnapshot]
  mut logits : Bytes
  first_request : Int32
  last_request : Int32
}

///|
priv struct Trial {
  tokens : Array[Int32]
  samples : Array[TokenSample]
  host_us : Double
  deferred_read_us : Double
}

///|
async fn resident_trial(
  jtag : Jtag,
  source : Source,
  plans : Array[@qwen.DecodeStep],
  programs : Array[Array[ResidentProgram]],
  prompt : Array[Int32],
  generation : Int32,
  fast~ : Bool = false,
  chain~ : Bool = false,
  capture~ : Bool = true,
) -> Trial {
  let tokens = prompt.copy()
  let samples : Array[TokenSample] = []
  let initial = cycle_snapshot(decode_hex(jtag.request("S")))
  assert_true(initial.sequence == initial.after && (initial.status & 11U) == 1U)
  let mut sequence = initial.sequence
  let queued = chain &&
    fast &&
    @env.get_env_var("PBLAZE3_TOKEN_QUEUE") == Some("1")
  let clock = @bench.monotonic_clock_start()
  for position in 0..<(prompt.length() + generation - 1) {
    let first_request = jtag.sequence + 1
    let start = @bench.monotonic_clock_start()
    let plan = plans[position]
    let token = tokens[position]
    let embedding = source.step_input(plan, token)
    let token_setup = if queued {
      assert_eq(programs[position].length(), 1)
      Some((plan.logits_address(), plan.embedding_address(), embedding))
    } else {
      jtag.write(plan.embedding_address(), embedding)
      if fast {
        jtag.set_word(0x80100048UL, plan.logits_address().to_uint())
      } else {
        jtag.write(plan.constants_address(), plan.constants())
      }
      None
    }
    let dispatch = @bench.monotonic_clock_start()
    let snapshots : Array[CycleSnapshot] = []
    for program in programs[position] {
      let snapshot = jtag.resident_execute(
        program,
        sequence,
        chain~,
        token_setup~,
      )
      sequence = snapshot.sequence
      snapshots.push(snapshot)
    }
    let dispatch_us = @bench.monotonic_clock_end(dispatch)
    let read = @bench.monotonic_clock_start()
    let logits = if fast {
      b""
    } else {
      jtag.read(plan.logits_address(), 151936 * 4)
    }
    let token_id = if fast {
      match snapshots.last().unwrap().selected_token {
        Some(id) => id
        None => jtag.word(0x8010004cUL)
      }
    } else {
      0U
    }
    let logits_us = @bench.monotonic_clock_end(read)
    let generated = if fast {
      guard token_id < 151936U else {
        fail("incomplete or invalid FPGA argmax")
      }
      token_id.reinterpret_as_int()
    } else {
      let values = Array::makei(151936, i => {
        Float::reinterpret_from_uint(decode_word(logits, i))
      })
      greedy(values[:])
    }
    if position >= prompt.length() - 1 {
      tokens.push(generated)
    }
    let host_us = @bench.monotonic_clock_end(start)
    samples.push({
      position,
      input_token: token,
      output_token: generated,
      host_us,
      dispatch_us,
      logits_us,
      snapshots,
      logits,
      first_request,
      last_request: jtag.sequence,
    })
    if position >= prompt.length() - 1 && terminal_token(generated) {
      break
    }
  }
  let host_us = @bench.monotonic_clock_end(clock)
  let deferred_read_us = if fast && capture {
    let read_start = @bench.monotonic_clock_start()
    for sample in samples {
      sample.logits = jtag.read(plans[sample.position].logits_address(), 607744)
    }
    @bench.monotonic_clock_end(read_start)
  } else {
    0.0
  }
  { tokens, samples, host_us, deferred_read_us, }
}

///|
fn sample_json(
  sample : TokenSample,
  prompt_count : Int32,
  fast : Bool,
  hz : Int32,
) -> Json {
  let core = sample.snapshots.fold(init=0UL, (a, b) => a + b.core_cycles)
  let loader = sample.snapshots.fold(init=0UL, (a, b) => a + b.loader_cycles)
  {
    "position": sample.position,
    "first_transport_request": sample.first_request,
    "last_transport_request": sample.last_request,
    "input_token": sample.input_token,
    "output_token": sample.output_token,
    "phase": if sample.position < prompt_count {
      "prefill"
    } else {
      "decode"
    },
    "host_us": sample.host_us,
    "dispatch_us": sample.dispatch_us,
    "logits_read_us": if fast {
      0.0
    } else {
      sample.logits_us
    },
    "result_read_us": sample.logits_us,
    "core_cycles": core,
    "loader_cycles": loader,
    "core_seconds": core.to_double() / hz.to_double(),
    "loader_seconds": loader.to_double() / hz.to_double(),
    "launches": sample.snapshots,
  }
}

///|
async fn measure_trials(
  jtag : Jtag,
  source : Source,
  plans : Array[@qwen.DecodeStep],
  programs : Array[Array[ResidentProgram]],
  prompt : Array[Int32],
  generation : Int32,
  fast~ : Bool,
  chain~ : Bool,
  loader_samples~ : Array[Json]? = None,
  core_detail~ : Bool = false,
) -> Array[Trial] {
  // Warm-up writes fresh KV positions from zero; later trials overwrite the
  // same positions before consuming them, so no prior request's KV is reused.
  ignore(
    resident_trial(
      jtag,
      source,
      plans,
      programs,
      prompt,
      1,
      fast~,
      chain~,
      capture=false,
    ),
  )
  let trials : Array[Trial] = []
  for trial in 0..<3 {
    let before = if loader_samples is Some(_) {
      jtag.trial_counters(core_detail)
    } else {
      []
    }
    let result = resident_trial(
      jtag,
      source,
      plans,
      programs,
      prompt,
      generation,
      fast~,
      chain~,
    )
    if loader_samples is Some(log) {
      let after = jtag.trial_counters(core_detail)
      log.push({
        "trial": trial,
        "before": before,
        "after": after,
        "delta": Array::makei(before.length(), i => after[i] - before[i]),
      })
    }
    trials.push(result)
    println(
      "Measured trial \{trial}: \{result.tokens.length() - prompt.length()} tokens in \{result.host_us / 1000000.0}s",
    )
  }
  trials
}

///|
/// Isolated single-word CSR reads surround the full trial, outside its timer.
async fn Jtag::trial_counters(self : Self, detail : Bool) -> Array[UInt32] {
  let words = []
  let base = if detail { 0x80100268UL } else { 0x80100248UL }
  for i in 0..<(if detail { 13 } else { 5 }) {
    words.push(self.word(base + i.to_uint64() * 4UL))
  }
  words
}
