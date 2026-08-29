///|
pub fn LunaTilePairedBenchmarkObservation::new(
  candidate_time_ns~ : Int64,
  fallback_time_ns~ : Int64,
) -> LunaTilePairedBenchmarkObservation raise LunaTileOptimizerPromotionError {
  if candidate_time_ns <= 0L ||
    fallback_time_ns <= 0L ||
    candidate_time_ns > 9_000_000_000L ||
    fallback_time_ns > 9_000_000_000L {
    raise LunaTileOptimizerPromotionInvalid(Benchmark)
  }
  { candidate_time_ns, fallback_time_ns }
}

///|
/// Admits a paired benchmark only when every pair and the bounded aggregate
/// beat the exact fallback by at least `minimum_win_ppm`.
pub fn admit_luna_tile_paired_benchmark_win(
  path~ : LunaTileOptimizerPath,
  candidate~ : LunaTileOptimizerDigest,
  fallback~ : LunaTileOptimizerDigest,
  physical~ : LunaTileOptimizerDigest,
  observations~ : ArrayView[LunaTilePairedBenchmarkObservation],
  minimum_win_ppm~ : Int,
) -> AdmittedLunaTilePairedBenchmarkWin raise LunaTileOptimizerPromotionError {
  if candidate == fallback ||
    observations.length() < 3 ||
    observations.length() > 1024 ||
    minimum_win_ppm <= 0 ||
    minimum_win_ppm > 500_000 {
    raise LunaTileOptimizerPromotionInvalid(Benchmark)
  }
  let mut candidate_total = 0L
  let mut fallback_total = 0L
  for observation in observations {
    if observation.candidate_time_ns >= observation.fallback_time_ns ||
      candidate_total > 9_000_000_000_000L - observation.candidate_time_ns ||
      fallback_total > 9_000_000_000_000L - observation.fallback_time_ns {
      raise LunaTileOptimizerPromotionMismatch(NonWin)
    }
    candidate_total += observation.candidate_time_ns
    fallback_total += observation.fallback_time_ns
  }
  let observed_win_ppm = ((fallback_total - candidate_total) *
  1_000_000L /
  fallback_total).to_int()
  if observed_win_ppm < minimum_win_ppm {
    raise LunaTileOptimizerPromotionMismatch(NonWin)
  }
  let output = StringBuilder()
  output.write_string("schema=lunaflux-lunatile-paired-benchmark-win.v1\n")
  output.write_string("path=" + path.stable_name() + "\n")
  output.write_string("candidate_sha256=" + candidate.as_hex() + "\n")
  output.write_string("fallback_sha256=" + fallback.as_hex() + "\n")
  output.write_string("physical_evidence_sha256=" + physical.as_hex() + "\n")
  output.write_string(
    "observation_count=" + observations.length().to_string() + "\n",
  )
  for index, observation in observations {
    output.write_string(
      "pair_" +
      index.to_string() +
      "=" +
      observation.candidate_time_ns.to_string() +
      "," +
      observation.fallback_time_ns.to_string() +
      "\n",
    )
  }
  output.write_string(
    "candidate_total_ns=" + candidate_total.to_string() + "\n",
  )
  output.write_string("fallback_total_ns=" + fallback_total.to_string() + "\n")
  output.write_string("observed_win_ppm=" + observed_win_ppm.to_string() + "\n")
  output.write_string("minimum_win_ppm=" + minimum_win_ppm.to_string() + "\n")
  output.write_string("manifest_bindable=false\npromotion_authority=absent\n")
  let canonical = @utf8.encode(output.to_string())
  {
    path,
    candidate,
    fallback,
    physical,
    observation_count: observations.length(),
    candidate_total_ns: candidate_total,
    fallback_total_ns: fallback_total,
    observed_win_ppm,
    canonical,
    digest: luna_tile_promotion_digest(canonical),
  }
}

///|
pub fn AdmittedLunaTilePairedBenchmarkWin::digest(
  self : AdmittedLunaTilePairedBenchmarkWin,
) -> LunaTileOptimizerDigest {
  self.digest
}

///|
pub fn AdmittedLunaTilePairedBenchmarkWin::observed_win_ppm(
  self : AdmittedLunaTilePairedBenchmarkWin,
) -> Int {
  self.observed_win_ppm
}
