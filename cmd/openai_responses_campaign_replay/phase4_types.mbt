///|
const PHASE4_PROFILE_TRIALS : Int = 9

///|
const PHASE4_CORRECTNESS_TRIALS : Int = 81

///|
const PHASE4_BASIS_POINTS : Int64 = 10000L

///|
const PHASE4_MAX_COLD_REGRESSION_BPS : Int = 1000

///|
const PHASE4_MAX_TRIAL_ELAPSED_NS : Int64 = 86400000000000L

///|
const PHASE4_MAX_TRIAL_GENERATED_TOKENS : Int64 = 1000000000L

///|
const PHASE4_MAX_DECLARATION_BYTES : Int = 1048576

///|
const PHASE4_MAX_CAPTURE_BYTES : Int = 67108864

///|
const PHASE4_MAX_CAPTURE_SET_BYTES : Int64 = 536870912L

///|
const PHASE4_MAX_CORRECTNESS_BYTES : Int = 4096

///|
const PHASE4_MAX_CORRECTNESS_INVENTORY_BYTES : Int = 16384

///|
suberror Phase4PrefixGateError {
  Phase4InvalidPolicy
  Phase4InvalidCorrectness
  Phase4InvalidMeasurement
  Phase4IntegerOverflow
} derive(Debug, Eq)

///|
priv struct Phase4ProfileAggregate {
  completed : Int64
  input_tokens : Int64
  generated_tokens : Int64
  elapsed_ns : Int64
  ttft_ns : Int64
  end_to_end_ns : Int64
}

///|
priv struct Phase4ProfileMatrix {
  lunaflux : Phase4ProfileAggregate
  vllm : Phase4ProfileAggregate
  sglang : Phase4ProfileAggregate
}

///|
priv struct Phase4PrefixGateEvidence {
  canonical : Bytes
  digest : @evidence.BenchmarkDigest
  passed : Bool
}

///|
priv struct Phase4PolicyEngine {
  name : String
  revision : String
  image : String
  configuration : String
  executable : String
}

///|
priv struct Phase4ExternalPolicy {
  digest : String
  declaration : String
  hardware : String
  driver : String
  toolkit : String
  model : String
  tokenizer : String
  corpus : String
  protocol : String
  randomization : String
  prefix_rich_workload : String
  prefix_cold_workload : String
  engines : FixedArray[Phase4PolicyEngine]
  capture_inventory : String
  correctness_inventory : String
  campaign_tool : String
  correctness_verifier : String
  identity_verifier : String
  process_supervisor : String
  gate_executable : String
  fd_exec_helper : String
  cold_regression_budget_bps : Int
}

///|
fn Phase4PrefixGateEvidence::canonical_bytes(
  self : Phase4PrefixGateEvidence,
) -> Bytes {
  self.canonical
}

///|
fn Phase4PrefixGateEvidence::digest(
  self : Phase4PrefixGateEvidence,
) -> @evidence.BenchmarkDigest {
  self.digest
}

///|
fn Phase4PrefixGateEvidence::passed(self : Phase4PrefixGateEvidence) -> Bool {
  self.passed
}
