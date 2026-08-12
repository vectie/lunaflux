///|
/// Stable sampling-mode vocabulary exposed by the native protocol.
pub(all) enum SamplingMode {
  Greedy
  Sample
} derive(Debug, Eq)

///|
/// Validated v1 sampling parameters.
///
/// Greedy is represented separately from probabilistic sampling so zero
/// temperature never acquires an implicit or adapter-specific meaning.
pub struct SamplingParameters {
  mode : SamplingMode
  temperature : Double
  top_k : Int?
  top_p : Double?
} derive(Debug, Eq)

///|
pub fn SamplingParameters::greedy() -> SamplingParameters {
  { mode: Greedy, temperature: 1.0, top_k: None, top_p: None }
}

///|
/// Constructs temperature sampling with optional top-k and top-p filters.
pub fn SamplingParameters::sample(
  temperature~ : Double,
  top_k? : Int,
  top_p? : Double,
  limits : InferenceLimits,
) -> SamplingParameters raise InferenceContractError {
  if temperature.is_nan() || temperature.is_inf() {
    raise invalid(Temperature, NonFinite, 0)
  }
  if temperature <= 0.0 || temperature > limits.max_temperature() {
    raise invalid(Temperature, OutOfRange, 0)
  }
  match top_k {
    Some(value) if value <= 0 || value > limits.max_top_k() =>
      raise invalid(TopK, OutOfRange, 0)
    _ => ()
  }
  match top_p {
    Some(value) if value.is_nan() || value.is_inf() =>
      raise invalid(TopP, NonFinite, 0)
    Some(value) if value <= 0.0 || value > 1.0 =>
      raise invalid(TopP, OutOfRange, 0)
    _ => ()
  }
  { mode: Sample, temperature, top_k, top_p }
}

///|
pub fn SamplingParameters::mode(self : SamplingParameters) -> SamplingMode {
  self.mode
}

///|
pub fn SamplingParameters::temperature(self : SamplingParameters) -> Double {
  self.temperature
}

///|
pub fn SamplingParameters::top_k(self : SamplingParameters) -> Int? {
  self.top_k
}

///|
pub fn SamplingParameters::top_p(self : SamplingParameters) -> Double? {
  self.top_p
}

///|
fn SamplingParameters::validate(
  self : SamplingParameters,
  limits : InferenceLimits,
) -> Unit raise InferenceContractError {
  match self.mode {
    Greedy => ()
    Sample =>
      ignore(
        SamplingParameters::sample(
          temperature=self.temperature,
          top_k?=self.top_k,
          top_p?=self.top_p,
          limits,
        ),
      )
  }
}
