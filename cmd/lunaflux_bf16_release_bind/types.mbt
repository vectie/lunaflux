///|
const APPROVED_CONFIG_SHA256 : String = "9197475bfcc987a4f9361dbc22b33397b101372c137c228b6a6fd7e4adf21622"

///|
const APPROVED_MODEL_SHA256 : String = "852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30"

///|
const RELEASE_TOKENS_PER_PAGE : Int = 8

///|
const RELEASE_TOTAL_PAGE_COUNT : Int = 32

///|
const RELEASE_MAX_SEQUENCE_TOKENS : Int = 256

///|
const RELEASE_MAX_PAGE_TABLE_ENTRIES : Int = 32

///|
suberror LunaBf16ReleaseBindError {
  InvalidArguments
  ModelAdmission
  CompiledSetAdmission(String)
  KernelBinding
  Publication
} derive(Debug, Eq)

///|
struct ReleaseBindArguments {
  model_root : String
  compiled_root : String
  toolchain_manifest : String
  toolchain_sha256 : String
  compiler_major : UInt
  compiler_minor : UInt
  compiler_patch : UInt
  absolute_new_output : String
}

///|
struct ReleaseBindInputs {
  plan : @plan.ModelPlan
  weights : @device_materialize.DeviceWeightFileInspection
}

///|
struct ReleaseBindEvidence {
  model_plan_sha256 : String
  manifest_sha256 : String
  bootstrap_sha256 : String
  inventory_sha256 : String
  plan_sha256 : String
}
