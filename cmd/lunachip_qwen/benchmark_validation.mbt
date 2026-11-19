///|
/// Validation happens only after every trial's timer has stopped. Equal token
/// histories and complete equal logit bytes let one reference replay cover all
/// repeated trials; no sampled-logit approximation is used.
async fn validate_trials(
  source : Source,
  plans : Array[@qwen.DecodeStep],
  trials : Array[Trial],
  output : String,
) -> Unit {
  assert_true(!trials.is_empty())
  let baseline = trials[0]
  for trial in trials[1:] {
    assert_eq(trial.tokens, baseline.tokens)
    assert_eq(trial.samples.length(), baseline.samples.length())
    for i, sample in trial.samples {
      assert_eq(sample.logits, baseline.samples[i].logits)
    }
  }
  let emulator = Emulator::new(source)
  let reference = Reference::new(source)
  let evidence : Array[Json] = []
  for sample in baseline.samples {
    let plan = plans[sample.position]
    emulator.write(
      plan.embedding_address(),
      source.step_input(plan, sample.input_token),
    )
    emulator.write(plan.constants_address(), plan.constants())
    for layer in 0..<plan.program_count() {
      emulator.execute(plan.program(layer).unwrap())
    }
    let golden = Array::makei(151936, i => {
      emulator.read(plan.logits_address() + i.to_uint64() * 4UL)
    })
    @fs.write_file(
      output + "/p\{sample.position}-emulator-logits.bin",
      float_bytes(golden[:]),
      create_mode=CreateNew,
    )
    assert_eq(
      sample.logits,
      float_bytes(golden[:]),
      msg="FPGA/ISA logits differ",
    )
    let expected = reference.step(sample.input_token, sample.position)
    assert_eq(
      sample.output_token,
      greedy(golden[:]),
      msg="argmax differs from full exact logits",
    )
    @fs.write_file(
      output + "/p\{sample.position}-reference-logits.bin",
      float_bytes(expected[:]),
      create_mode=CreateNew,
    )
    assert_eq(sample.output_token, greedy(expected[:]))
    let mut max_error = 0.0
    for i in 0..<151936 {
      max_error = max_error.max(
        (golden[i].to_double() - expected[i].to_double()).abs(),
      )
    }
    assert_true(max_error <= 0.05)
    evidence.push({
      "position": sample.position,
      "exact_logit_words_per_trial": 151936,
      "trials": trials.length(),
      "reference_max_absolute_error": max_error,
    })
    @fs.write_file(
      output + "/validation-progress.json",
      Json::array(evidence.copy()).stringify(indent=2),
    )
    println(
      "Post-timing full-logit validation passed position \{sample.position}",
    )
  }
}
