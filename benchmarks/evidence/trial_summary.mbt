///|
/// Integer-only scalars derived from one complete, bounded request event set.
pub struct BenchmarkTrialSummary {
  priv submitted : Int
  priv completed : Int
  priv rejected : Int
  priv timed_out : Int
  priv cancelled : Int
  priv failed : Int
  priv input_tokens : Int64
  priv generated_tokens : Int64
  priv elapsed_ns : Int64
  priv queue_ns : BenchmarkIntegerSummary
  priv ttft_ns : BenchmarkIntegerSummary
  priv service_ns : BenchmarkIntegerSummary
  priv decode_ns : BenchmarkIntegerSummary
  priv end_to_end_ns : BenchmarkIntegerSummary
}

///|
pub fn BenchmarkTrialSummary::submitted(self : BenchmarkTrialSummary) -> Int {
  self.submitted
}

///|
pub fn BenchmarkTrialSummary::completed(self : BenchmarkTrialSummary) -> Int {
  self.completed
}

///|
pub fn BenchmarkTrialSummary::rejected(self : BenchmarkTrialSummary) -> Int {
  self.rejected
}

///|
pub fn BenchmarkTrialSummary::timed_out(self : BenchmarkTrialSummary) -> Int {
  self.timed_out
}

///|
pub fn BenchmarkTrialSummary::cancelled(self : BenchmarkTrialSummary) -> Int {
  self.cancelled
}

///|
pub fn BenchmarkTrialSummary::failed(self : BenchmarkTrialSummary) -> Int {
  self.failed
}

///|
pub fn BenchmarkTrialSummary::input_tokens(
  self : BenchmarkTrialSummary,
) -> Int64 {
  self.input_tokens
}

///|
pub fn BenchmarkTrialSummary::generated_tokens(
  self : BenchmarkTrialSummary,
) -> Int64 {
  self.generated_tokens
}

///|
pub fn BenchmarkTrialSummary::elapsed_ns(self : BenchmarkTrialSummary) -> Int64 {
  self.elapsed_ns
}

///|
pub fn BenchmarkTrialSummary::queue_ns(
  self : BenchmarkTrialSummary,
) -> BenchmarkIntegerSummary {
  self.queue_ns
}

///|
pub fn BenchmarkTrialSummary::ttft_ns(
  self : BenchmarkTrialSummary,
) -> BenchmarkIntegerSummary {
  self.ttft_ns
}

///|
pub fn BenchmarkTrialSummary::service_ns(
  self : BenchmarkTrialSummary,
) -> BenchmarkIntegerSummary {
  self.service_ns
}

///|
pub fn BenchmarkTrialSummary::decode_ns(
  self : BenchmarkTrialSummary,
) -> BenchmarkIntegerSummary {
  self.decode_ns
}

///|
pub fn BenchmarkTrialSummary::end_to_end_ns(
  self : BenchmarkTrialSummary,
) -> BenchmarkIntegerSummary {
  self.end_to_end_ns
}
