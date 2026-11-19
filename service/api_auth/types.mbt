///|
const LUNA_API_AUTH_MAX_CREDENTIAL_BYTES : Int32 = 4096

///|
const LUNA_API_AUTH_MAX_STEP_WORK_UNITS : Int32 = 65536

///|
const LUNA_API_AUTH_IDLE : Int32 = 0

///|
const LUNA_API_AUTH_WRITING : Int32 = 1

///|
const LUNA_API_AUTH_PADDING : Int32 = 2

///|
const LUNA_API_AUTH_ACCEPTED : Int32 = 3

///|
const LUNA_API_AUTH_REJECTED : Int32 = 4

///|
const LUNA_API_AUTH_FAILED : Int32 = 5

///|
const LUNA_API_AUTH_CLOSED : Int32 = 6

///|
pub(all) enum LunaApiAuthRule {
  LunaApiAuthPolicy
  LunaApiAuthCredential
  LunaApiAuthSource
  LunaApiAuthWorkUnits
  LunaApiAuthLifecycle
  LunaApiAuthEpoch
} derive(Debug, Eq)

///|
pub(all) enum LunaApiAuthIssue {
  LunaApiAuthInvalid
  LunaApiAuthEmpty
  LunaApiAuthTooLong
  LunaApiAuthOutOfRange
  LunaApiAuthBusy
  LunaApiAuthNotReady
  LunaApiAuthStale
  LunaApiAuthExhausted
} derive(Debug, Eq)

///|
pub(all) suberror LunaApiAuthError {
  LunaApiAuthFailed(rule~ : LunaApiAuthRule, issue~ : LunaApiAuthIssue)
} derive(Debug, Eq)

///|
/// Scalar authentication outcome. It carries no credential evidence.
#valtype
pub struct LunaApiAuthDecision {
  priv state : Int32
}

///|
pub fn LunaApiAuthDecision::is_accepted(self : LunaApiAuthDecision) -> Bool {
  self.state == 0
}

///|
pub fn LunaApiAuthDecision::is_rejected(self : LunaApiAuthDecision) -> Bool {
  self.state == 1
}

///|
/// Immutable startup-owned credential policy.
///
/// The policy deliberately has no `Debug` implementation and exposes no
/// credential, storage, digest, or bound getter.
pub struct LunaApiAuthPolicy {
  priv expected : FixedArray[Byte]
  priv mut expected_length : Int32
  priv mut maximum_credential_bytes : Int32
  priv mut closed : Bool
}

///|
#valtype
priv struct LunaApiAuthComparison {
  difference : UInt32
  comparison_count : Int32
}

///|
#valtype
pub struct LunaApiAuthStepBudget {
  priv work_units : Int32
}

///|
pub fn LunaApiAuthStepBudget::new(
  work_units : Int32,
) -> LunaApiAuthStepBudget raise LunaApiAuthError {
  if work_units <= 0 || work_units > LUNA_API_AUTH_MAX_STEP_WORK_UNITS {
    raise luna_api_auth_error(LunaApiAuthWorkUnits, LunaApiAuthOutOfRange)
  }
  { work_units, }
}

///|
pub fn LunaApiAuthStepBudget::work_units(self : LunaApiAuthStepBudget) -> Int32 {
  self.work_units
}

///|
#valtype
pub struct LunaApiAuthWriteProgress {
  priv state : Int32
  priv work_units : Int32
}

///|
pub fn LunaApiAuthWriteProgress::is_accepting(
  self : LunaApiAuthWriteProgress,
) -> Bool {
  self.state == 0
}

///|
pub fn LunaApiAuthWriteProgress::is_complete(
  self : LunaApiAuthWriteProgress,
) -> Bool {
  self.state == 1
}

///|
pub fn LunaApiAuthWriteProgress::work_units(
  self : LunaApiAuthWriteProgress,
) -> Int32 {
  self.work_units
}

///|
#valtype
pub(all) enum LunaApiAuthVerificationState {
  LunaApiAuthVerificationPending
  LunaApiAuthVerificationAccepted
  LunaApiAuthVerificationRejected
  LunaApiAuthVerificationFailed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaApiAuthVerificationProgress {
  LunaApiAuthVerificationIdle
  LunaApiAuthVerificationAdvanced
} derive(Debug, Eq)

///|
pub struct LunaApiAuthVerifierWorkspace {
  priv policy : LunaApiAuthPolicy
  priv budget : LunaApiAuthStepBudget
  priv mut epoch : UInt64
  priv mut phase : Int32
  priv mut required_input_bytes : Int32
  priv mut cursor : Int32
  priv mut difference : UInt32
  priv mut failure_rule : Int32
  priv mut failure_issue : Int32
  priv mut last_work_units : Int32
  priv mut total_work_units : UInt64
}

///|
#valtype
pub struct LunaApiAuthVerificationWrite {
  priv workspace : LunaApiAuthVerifierWorkspace
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaApiAuthVerificationWork {
  priv workspace : LunaApiAuthVerifierWorkspace
  priv epoch : UInt64
}

///|
pub extend LunaApiAuthRule with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaApiAuthRule with Eq::{not_equal, equal}

///|
pub extend LunaApiAuthIssue with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaApiAuthIssue with Eq::{not_equal, equal}

///|
pub extend LunaApiAuthError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend LunaApiAuthError with Eq::{not_equal, equal}

///|
pub extend LunaApiAuthVerificationState with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend LunaApiAuthVerificationState with Eq::{not_equal, equal}

///|
pub extend LunaApiAuthVerificationProgress with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend LunaApiAuthVerificationProgress with Eq::{not_equal, equal}
