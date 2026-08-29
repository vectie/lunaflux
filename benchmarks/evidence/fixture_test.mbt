///|
fn test_digest(unit : Char) -> @evidence.BenchmarkDigest raise {
  let builder = StringBuilder(size_hint=64)
  for _ in 0..<64 {
    builder.write_char(unit)
  }
  @evidence.BenchmarkDigest::from_sha256(builder.to_string())
}

///|
fn test_environment() -> @evidence.BenchmarkEnvironment raise {
  @evidence.BenchmarkEnvironment::new(
    hardware=test_digest('1'),
    driver=test_digest('2'),
    toolkit=test_digest('3'),
    model=test_digest('4'),
    tokenizer=test_digest('5'),
    protocol=test_digest('6'),
    corpus=test_digest('7'),
    randomization=test_digest('8'),
  )
}

///|
fn test_engines() -> Array[@evidence.BenchmarkEngineIdentity] raise {
  [
    @evidence.BenchmarkEngineIdentity::new(
      engine=LunaFlux,
      revision=test_digest('9'),
      executable=test_digest('a'),
    ),
    @evidence.BenchmarkEngineIdentity::new(
      engine=Vllm,
      revision=test_digest('b'),
      executable=test_digest('c'),
    ),
    @evidence.BenchmarkEngineIdentity::new(
      engine=Sglang,
      revision=test_digest('d'),
      executable=test_digest('e'),
    ),
  ]
}

///|
fn test_profiles() -> Array[@evidence.BenchmarkProfile] {
  [
    Latency,
    Chat,
    LongPrefill,
    DecodeHeavy,
    PrefixRich,
    PrefixCold,
    Saturation,
    Churn,
    Mixed,
  ]
}

///|
fn test_workloads() -> Array[@evidence.BenchmarkWorkload] raise {
  let workloads : Array[@evidence.BenchmarkWorkload] = []
  for profile in test_profiles() {
    workloads.push(
      @evidence.BenchmarkWorkload::new(profile~, configuration=test_digest('f')),
    )
  }
  workloads
}

///|
fn test_engine_tag(engine : @evidence.BenchmarkEngine) -> Int {
  match engine {
    LunaFlux => 0
    Vllm => 1
    Sglang => 2
  }
}

///|
fn test_position(engine : @evidence.BenchmarkEngine, ordinal : Int) -> Int {
  let tag = test_engine_tag(engine)
  match ordinal {
    1 => tag
    2 => (tag + 2) % 3
    _ => (tag + 1) % 3
  }
}

///|
fn test_trials(balanced : Bool) -> Array[@evidence.BenchmarkTrial] raise {
  let trials : Array[@evidence.BenchmarkTrial] = []
  let engines = [@evidence.LunaFlux, Vllm, Sglang]
  for profile in test_profiles() {
    for ordinal in 1..<=3 {
      for engine in engines {
        trials.push(
          @evidence.BenchmarkTrial::new(
            engine~,
            profile~,
            ordinal~,
            order_position=if balanced {
              test_position(engine, ordinal)
            } else {
              test_engine_tag(engine)
            },
            submitted=100,
            completed=91,
            rejected=2,
            timed_out=3,
            cancelled=3,
            failed=1,
            correctness_passed=true,
            raw_events=test_digest('a'),
            correctness=test_digest('b'),
            summary=test_digest('c'),
          ),
        )
      }
    }
  }
  trials
}
