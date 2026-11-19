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
async fn benchmark(
  source : Source,
  plans : Array[@qwen.DecodeStep],
  tokenizer : @tokenizer.TokenizerSpec,
  prompt : Array[Int32],
  generation : Int32,
  output : String,
  endpoint : String,
  qualification : String,
  clock_report~ : String?,
  loader_ab~ : Bool = false,
  unary_ab~ : Bool = false,
  generate~ : Bool = false,
  require_resident_weights~ : Bool = false,
  fast~ : Bool = false,
  chain~ : Bool = false,
) -> Unit {
  guard endpoint.has_prefix("127.0.0.1:") else { fail("loopback required") }
  let connection = @socket.Tcp::connect(@socket.Addr::parse(endpoint))
  defer connection.close()
  let jtag = Jtag::{
    connection,
    sequence: 0,
    owner: @device.LaunchOwner::new(),
  }
  let (jtag_hz, hz) = benchmark_transport(jtag.request("I"))
  let transport = if jtag_hz is None { "pcie-xdma" } else { "jtag" }
  if clock_report is Some(path) {
    assert_eq(
      hz,
      routed_core_clock(@fs.read_file(path).text()),
      msg="bridge and local routed clocks differ",
    )
  }
  let meta = jtag.request("META").split(" ").to_array()
  assert_eq(meta.length(), 2)
  assert_eq(@string.parse_int64(meta[0]), source.file.stamp().byte_count())
  assert_eq(
    decode_hex(meta[1]),
    read_range(source.file, 0L, source.payload.to_int()),
  )
  for
    (address, value) in [
      (0x80000000UL, 15U),
      (0x80100000UL, 0x50423330U),
      (0x80100004UL, 0x10001U),
      (0x80100080UL, 0x80000000U),
      (0x80100084UL, 0U),
      (0x80100088UL, 31U),
      (0x80100090UL, 1U),
    ] {
    assert_eq(jtag.word(address), value)
  }
  if chain {
    assert_eq(jtag.word(0x8010009cUL), 0x43484e31U)
  }
  let cache = unary_ab || generate || require_resident_weights
  if plans[0].uses_indirect_bf16() {
    assert_eq(jtag.word(0x801002a0UL), 0x42463136U)
  }
  if cache {
    assert_eq(jtag.word(0x80100260UL), 0x53465531U)
  }
  let id = if cache { Some(resident_identity(source, plans)) } else { None }
  let (reuse_weights, reuse_programs) = match id {
    Some(id) => jtag.resident_matches(id)
    None => (false, false)
  }
  guard !require_resident_weights || reuse_weights else {
    let name = if generate { "generate-warm" } else { "benchmark-warm" }
    fail("\{name} requires matching weights in board DDR; no weights uploaded")
  }
  if cache {
    jtag.set_word(0x80100264UL, 0U)
  }
  if loader_ab || cache {
    assert_eq(jtag.word(0x80100240UL), 0x50425231U)
    jtag.set_word(0x80100244UL, 0U)
  }
  jtag.set_word(0x8010003cUL, 0U)
  if fast {
    assert_eq(jtag.word(0x80100094UL), 1U)
    // Select the contiguous counter mirror and single-word busy polling.
    assert_eq(jtag.request("FAST"), "1")
    jtag.set_word(0x80100044UL, 151936U)
  }
  if cache && !reuse_programs {
    // Invalidate before touching a resident region. A failed upload cannot
    // publish a partial bundle. A reset during upload clears the CSR.
    if !reuse_weights {
      jtag.set_word(0x8010029cUL, 0U)
      jtag.write(record_address, words_bytes(Array::make(16, 0U)[:]))
      jtag.set_word(0x8010029cUL, 1U)
    } else {
      jtag.write(record_address + 32UL, words_bytes(Array::make(8, 0U)[:]))
    }
  }
  if !reuse_weights {
    jtag.qualify(qualification, output)
    @fs.mkdir(output + "/resident-qualification")
    jtag.qualify(
      qualification,
      output + "/resident-qualification",
      resident=source.layout.reserve().address + 131072UL,
    )
    if chain {
      @fs.mkdir(output + "/chain-qualification")
      jtag.qualify(
        qualification,
        output + "/chain-qualification",
        resident=source.layout.reserve().address + 131072UL,
        chain=true,
      )
    }
    // Transport qualification is outside timing and outside every model region.
    for round in 0..<16 {
      let probe = Bytes::from_array(
        Array::makei(65536, i => {
          ((i * 37 + i / 251 + round * 71) % 256).to_byte()
        })[:],
      )
      let address = source.layout.reserve().address + 3584UL
      jtag.write(address, probe)
      assert_eq(jtag.read(address, probe.length()), probe)
    }
  }
  let upload_start = @bench.monotonic_clock_start()
  let verify_weights = !cache ||
    @env.get_env_var("PBLAZE3_VERIFY_WEIGHTS") == Some("1")
  let uploaded = if reuse_weights {
    0L
  } else {
    jtag.upload_weights(source, output, verify=verify_weights)
  }
  let upload_us = @bench.monotonic_clock_end(upload_start)
  let programs = jtag.upload_programs(
    source,
    plans,
    output,
    reuse=reuse_programs,
  )
  if id is Some(identity) {
    if !reuse_programs {
      if !reuse_weights {
        assert_true(
          source.file.stamp() == identity.source_stamp,
          msg="checkpoint changed during upload",
        )
        assert_eq(
          jtag.request("DIGEST"),
          identity.checkpoint,
          msg="transport checkpoint changed during upload",
        )
        jtag.write(record_address, identity.weights)
      }
      jtag.write(record_address + 32UL, identity.programs)
    }
    assert_eq(jtag.word(0x8010029cUL), 1U, msg="DDR reset during preparation")
    let record : Json = {
      "checkpoint_sha256": identity.checkpoint,
      "host_transport": transport,
      "weights_reused": reuse_weights,
      "programs_reused": reuse_programs,
      "weight_bytes_uploaded": uploaded,
      "weight_bytes_read_back": Json(if verify_weights { uploaded } else { 0L }),
      "weight_upload_us": upload_us,
    }
    @fs.write_file(
      output + "/residency.json",
      record.stringify(indent=2),
      create_mode=CreateNew,
    )
  }
  if fast {
    for plan in plans {
      jtag.write(plan.constants_address(), plan.constants())
    }
  }
  // The control and candidate share this upload and the same routed image.
  let launches = if chain {
    programs.map(xs => [packed_chain(xs)])
  } else {
    programs
  }
  if generate {
    let trial = resident_trial(
      jtag,
      source,
      plans,
      launches,
      prompt,
      generation,
      fast=true,
      chain=true,
      capture=false,
    )
    let answer = tokenizer.decode_text(
      trial.tokens[prompt.length():],
      special_policy=Preserve,
    )
    let result : Json = {
      "host_transport": transport,
      "jtag_clock_hz": jtag_hz,
      "core_clock_hz": hz,
      "tokens": trial.tokens,
      "generated_text": answer,
      "host_us": trial.host_us,
      "positions": trial.samples.map(s => {
        sample_json(s, prompt.length(), true, hz)
      }),
    }
    @fs.write_file(
      output + "/generation.json",
      result.stringify(indent=2),
      create_mode=CreateNew,
    )
    println(answer)
    return
  }
  let loader_samples : Array[Json] = []
  if unary_ab {
    jtag.set_word(0x80100264UL, 1U)
  }
  if loader_ab {
    jtag.set_word(0x80100244UL, 1U)
  }
  let control = if chain && !require_resident_weights {
    Some(
      measure_trials(
        jtag,
        source,
        plans,
        if loader_ab || unary_ab {
          launches
        } else {
          programs
        },
        prompt,
        generation,
        fast~,
        chain=loader_ab || unary_ab,
        core_detail=unary_ab,
        loader_samples=if loader_ab || unary_ab {
          Some(loader_samples)
        } else {
          None
        },
      ),
    )
  } else {
    None
  }
  if loader_ab {
    jtag.set_word(0x80100244UL, 0U)
  }
  if unary_ab {
    jtag.set_word(0x80100264UL, 0U)
  }
  let trials = measure_trials(
    jtag,
    source,
    plans,
    launches,
    prompt,
    generation,
    fast~,
    chain~,
    core_detail=unary_ab,
    loader_samples=if loader_ab || unary_ab {
      Some(loader_samples)
    } else {
      None
    },
  )
  if loader_ab || unary_ab {
    @fs.write_file(
      output +
      (if unary_ab {
        "/core-detail-counters.json"
      } else {
        "/loader-counters.json"
      }),
      Json(loader_samples).stringify(indent=2),
      create_mode=CreateNew,
    )
  }
  if control is Some(legacy) {
    let reports : Array[Json] = []
    for i, trial in legacy {
      assert_eq(trial.tokens, trials[i].tokens)
      for j, sample in trial.samples {
        assert_eq(sample.logits, trials[i].samples[j].logits)
      }
      reports.push({
        "host_us": trial.host_us,
        "samples": trial.samples.map(s => {
          sample_json(s, prompt.length(), fast, hz)
        }),
      })
    }
    @fs.write_file(
      output +
      (if unary_ab {
        "/same-image-legacy-unary.json"
      } else if loader_ab {
        "/same-image-scalar-loader.json"
      } else {
        "/same-image-segment-control.json"
      }),
      Json::array(reports).stringify(indent=2),
      create_mode=CreateNew,
    )
  }
  // Timing is over. Persist raw observations before expensive validation, so
  // a reference failure preserves the actual measurement without claiming PASS.
  let metadata : Array[Json] = []
  for t, trial in trials {
    for sample in trial.samples {
      @fs.write_file(
        output + "/trial\{t}-p\{sample.position}-logits.bin",
        sample.logits,
        create_mode=CreateNew,
      )
    }
    let generated = trial.tokens.length() - prompt.length()
    let decode = trial.samples.filter(s => s.position >= prompt.length())
    let decode_us = decode.fold(init=0.0, (a, b) => a + b.host_us)
    metadata.push({
      "trial": t,
      "tokens": trial.tokens,
      "generated_tokens": generated,
      "generated_text": tokenizer.decode_text(
        trial.tokens[prompt.length():],
        special_policy=Preserve,
      ),
      "generation_host_us": trial.host_us,
      "generation_tokens_per_second": generated.to_double() *
      1000000.0 /
      trial.host_us,
      "decode_tokens": decode.length(),
      "decode_host_us": decode_us,
      "decode_tokens_per_second": if decode.is_empty() {
        0.0
      } else {
        decode.length().to_double() * 1000000.0 / decode_us
      },
      "deferred_logits_read_us": trial.deferred_read_us,
      "samples": trial.samples.map(s => {
        sample_json(s, prompt.length(), fast, hz)
      }),
    })
  }
  @fs.write_file(
    output + "/measurements.json",
    Json::array(metadata.copy()).stringify(indent=2),
    create_mode=CreateNew,
  )
  validate_trials(source, plans, trials, output)
  if loader_ab {
    // Outside both A/B runs. Retire the identical resident segments with a
    // separate snapshot per segment to isolate the unhideable first load.
    let split = resident_trial(
      jtag,
      source,
      plans,
      programs,
      prompt,
      generation,
      fast=true,
    )
    assert_eq(split.tokens, trials[0].tokens)
    assert_eq(split.samples.length(), trials[0].samples.length())
    for p, sample in split.samples {
      assert_eq(sample.logits, trials[0].samples[p].logits)
      assert_eq(sample.snapshots.length(), programs[p].length())
    }
    @fs.write_file(
      output + "/prefetch-segments.json",
      Json(split.samples.map(s => sample_json(s, prompt.length(), true, hz))).stringify(
        indent=2,
      ),
      create_mode=CreateNew,
    )
  }
  let result : Json = {
    "schema": if chain {
      "lunaflux.pblaze3-resident-benchmark/v3"
    } else if fast {
      "lunaflux.pblaze3-resident-benchmark/v2"
    } else {
      "lunaflux.pblaze3-resident-benchmark/v1"
    },
    "status": "PASS",
    "physical_execution": true,
    "device_reads_bf16_embedding": plans[0].uses_indirect_bf16(),
    "core_clock_hz": hz,
    "routed_clock_report": clock_report,
    "host_transport": transport,
    "jtag_clock_hz": jtag_hz,
    "queued_token_setup": chain &&
    fast &&
    @env.get_env_var("PBLAZE3_TOKEN_QUEUE") == Some("1"),
    "weight_bytes_uploaded_and_readback_verified": Json(
      if verify_weights {
        uploaded
      } else {
        0L
      },
    ),
    "weight_bytes_uploaded": uploaded,
    "weight_upload_us": upload_us,
    "weights_resident": true,
    "programs_resident_in_ddr": true,
    "instruction_staging": "FPGA DDR-to-program-RAM loader; counted separately",
    "timer": "monotonic microseconds",
    "warmup_generated_tokens": 1,
    "fpga_greedy": fast,
    "timed_output_bytes_per_position": if fast {
      4
    } else {
      607744
    },
    "timed_embedding_write_bytes_per_position": if plans[0].uses_indirect_bf16() {
      4
    } else {
      4096
    },
    "timing_includes": if plans[0].uses_indirect_bf16() {
      "resident BF16 embedding row address, argmax arm, packed program dispatch/polling, FPGA execution including embedding fetch and selected token read"
    } else if fast {
      "embedding, argmax arm, packed program dispatch/polling, FPGA execution and selected token read"
    } else {
      "embedding and constants, host dispatch/polling, FPGA execution, all logits download and host greedy sampling"
    },
    "timing_excludes": "preparation, initial upload, warm-up, filesystem evidence output and all ISA/equation reference execution",
    "trials": metadata,
  }
  @fs.write_file(
    output + "/benchmark-result.json",
    result.stringify(indent=2),
    create_mode=CreateNew,
  )
  println(
    "Resident physical benchmark PASS; full result: \{output}/benchmark-result.json",
  )
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
