///|
/// Bounded stage at which selected-model admission failed.
pub(all) suberror LlamaAdmissionError {
  Configuration(@config_reader.ModelConfigReaderError)
  Safetensors(@safetensors_reader.SafetensorsReaderError)
  Plan(@plan.PlanValidationError)
  Weights(@llama_weights.LlamaWeightsError)
} derive(Debug, Eq)
