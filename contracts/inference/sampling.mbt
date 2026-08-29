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
  priv mode : SamplingMode
  priv temperature : Double
  priv top_k : Int?
  priv top_p : Double?
} derive(Debug, Eq)

///|
/// Allocation-free, already-validated sampling scalars for bounded ingress
/// handoff. Only the request assembly boundary materializes the public object.
#valtype
pub struct LunaValidatedSampling {
  priv mode : SamplingMode
  priv temperature : Double
  priv has_top_k : Bool
  priv top_k : Int
  priv has_top_p : Bool
  priv top_p : Double
}

///|
pub fn LunaValidatedSampling::greedy() -> Self {
  {
    mode: Greedy,
    temperature: 1.0,
    has_top_k: false,
    top_k: 0,
    has_top_p: false,
    top_p: 0.0,
  }
}

///|
pub fn LunaValidatedSampling::sample(
  temperature~ : Double,
  has_top_k~ : Bool,
  top_k~ : Int,
  has_top_p~ : Bool,
  top_p~ : Double,
  limits : InferenceLimits,
) -> Self raise InferenceContractError {
  if temperature.is_nan() || temperature.is_inf() {
    raise invalid(Temperature, NonFinite, 0)
  }
  if temperature <= 0.0 || temperature > limits.max_temperature() {
    raise invalid(Temperature, OutOfRange, 0)
  }
  if has_top_k && (top_k <= 0 || top_k > limits.max_top_k()) {
    raise invalid(TopK, OutOfRange, 0)
  }
  if has_top_p && (top_p.is_nan() || top_p.is_inf()) {
    raise invalid(TopP, NonFinite, 0)
  }
  if has_top_p && (top_p <= 0.0 || top_p > 1.0) {
    raise invalid(TopP, OutOfRange, 0)
  }
  { mode: Sample, temperature, has_top_k, top_k, has_top_p, top_p }
}

///|
pub fn LunaValidatedSampling::materialize(self : Self) -> SamplingParameters {
  {
    mode: self.mode,
    temperature: self.temperature,
    top_k: if self.has_top_k {
      Some(self.top_k)
    } else {
      None
    },
    top_p: if self.has_top_p {
      Some(self.top_p)
    } else {
      None
    },
  }
}

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
    Greedy => {
      if self.temperature != 1.0 {
        raise invalid(Temperature, Inconsistent, 0)
      }
      match self.top_k {
        Some(_) => raise invalid(TopK, Inconsistent, 0)
        None => ()
      }
      match self.top_p {
        Some(_) => raise invalid(TopP, Inconsistent, 0)
        None => ()
      }
    }
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
