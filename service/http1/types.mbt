///|
const LUNA_HTTP1_MIN_HEAD_BYTES : Int32 = 64

///|
const LUNA_HTTP1_MAX_HEAD_BYTES : Int32 = 65536

///|
const LUNA_HTTP1_MAX_BODY_BYTES : Int32 = 16777216

///|
const LUNA_HTTP1_MAX_STEP_WORK_UNITS : Int32 = 65536

///|
const LUNA_HTTP1_IDLE : Int32 = 0

///|
const LUNA_HTTP1_RECEIVING_HEAD : Int32 = 1

///|
const LUNA_HTTP1_SCANNING_LINE : Int32 = 2

///|
const LUNA_HTTP1_VALIDATING_REQUEST : Int32 = 3

///|
const LUNA_HTTP1_VALIDATING_HEADER_NAME : Int32 = 4

///|
const LUNA_HTTP1_VALIDATING_HEADER_VALUE : Int32 = 5

///|
const LUNA_HTTP1_VALIDATING_REQUIRED : Int32 = 6

///|
const LUNA_HTTP1_AUTH_BEGIN : Int32 = 7

///|
const LUNA_HTTP1_AUTH_WRITE : Int32 = 8

///|
const LUNA_HTTP1_AUTH_FINISH : Int32 = 9

///|
const LUNA_HTTP1_AUTH_PROGRESS : Int32 = 10

///|
const LUNA_HTTP1_AUTH_DECIDE : Int32 = 11

///|
const LUNA_HTTP1_RECEIVING_BODY : Int32 = 12

///|
const LUNA_HTTP1_FINALIZING : Int32 = 13

///|
const LUNA_HTTP1_READY : Int32 = 14

///|
const LUNA_HTTP1_VIEW_LEASED : Int32 = 15

///|
const LUNA_HTTP1_FAILED : Int32 = 16

///|
const LUNA_HTTP1_CLOSED : Int32 = 17

///|
const LUNA_HTTP1_HEADER_UNKNOWN : Int32 = 0

///|
const LUNA_HTTP1_HEADER_AUTHORIZATION : Int32 = 1

///|
const LUNA_HTTP1_HEADER_CONTENT_LENGTH : Int32 = 2

///|
const LUNA_HTTP1_HEADER_CONTENT_TYPE : Int32 = 3

///|
const LUNA_HTTP1_HEADER_TRANSFER_ENCODING : Int32 = 4

///|
#valtype
pub(all) enum LunaHttp1Route {
  LunaHttp1Responses
  LunaHttp1ChatCompletions
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaHttp1WorkState {
  LunaHttp1ReceivingHead
  LunaHttp1ValidatingHead
  LunaHttp1Authenticating
  LunaHttp1ReceivingBody
  LunaHttp1Ready
  LunaHttp1Failed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum LunaHttp1Progress {
  LunaHttp1Idle
  LunaHttp1Advanced
} derive(Debug, Eq)

///|
pub(all) enum LunaHttp1Rule {
  LunaHttp1Limits
  LunaHttp1Source
  LunaHttp1RequestLine
  LunaHttp1Method
  LunaHttp1RouteRule
  LunaHttp1Version
  LunaHttp1Header
  LunaHttp1Authorization
  LunaHttp1ContentLength
  LunaHttp1ContentType
  LunaHttp1TransferEncoding
  LunaHttp1Head
  LunaHttp1Body
  LunaHttp1Lifecycle
} derive(Debug, Eq)

///|
pub(all) enum LunaHttp1Issue {
  LunaHttp1Invalid
  LunaHttp1Missing
  LunaHttp1Duplicate
  LunaHttp1Conflicting
  LunaHttp1Unsupported
  LunaHttp1TooLong
  LunaHttp1OutOfRange
  LunaHttp1Rejected
  LunaHttp1PrematureEnd
  LunaHttp1Trailing
  LunaHttp1Busy
  LunaHttp1NotReady
  LunaHttp1Stale
  LunaHttp1Exhausted
} derive(Debug, Eq)

///|
pub(all) suberror LunaHttp1Error {
  LunaHttp1Failed(rule~ : LunaHttp1Rule, issue~ : LunaHttp1Issue)
} derive(Debug, Eq)

///|
#valtype
pub struct LunaHttp1Limits {
  priv max_head_bytes : Int32
  priv max_body_bytes : Int32
}

///|
#valtype
pub struct LunaHttp1StepBudget {
  priv work_units : Int32
}

///|
pub struct LunaHttp1Workspace {
  priv limits : LunaHttp1Limits
  priv budget : LunaHttp1StepBudget
  priv storage : FixedArray[Byte]
  priv auth_verifier : @api_auth.LunaApiAuthVerifierWorkspace
  priv auth_writes : Array[@api_auth.LunaApiAuthVerificationWrite]
  priv auth_works : Array[@api_auth.LunaApiAuthVerificationWork]
  priv mut epoch : UInt64
  priv mut phase : Int32
  priv mut failure_rule : Int32
  priv mut failure_issue : Int32
  priv mut head_length : Int32
  priv mut body_length : Int32
  priv mut body_written : Int32
  priv mut terminator_state : Int32
  priv mut input_ended : Bool
  priv mut scan_cursor : Int32
  priv mut scan_expect_lf : Bool
  priv mut line_start : Int32
  priv mut line_end : Int32
  priv mut request_line_complete : Bool
  priv mut parse_component : Int32
  priv mut parse_index : Int32
  priv mut route_responses : Bool
  priv mut route_chat : Bool
  priv mut header_kind : Int32
  priv mut header_name_index : Int32
  priv mut header_authorization : Bool
  priv mut header_content_length : Bool
  priv mut header_content_type : Bool
  priv mut header_transfer_encoding : Bool
  priv mut value_index : Int32
  priv mut value_leading_space : Bool
  priv mut value_number : Int32
  priv mut value_has_digit : Bool
  priv mut value_content_type : Bool
  priv mut credential_offset : Int32
  priv mut credential_length : Int32
  priv mut credential_padding : Bool
  priv mut authorization_seen : Bool
  priv mut content_length_seen : Bool
  priv mut content_type_seen : Bool
  priv mut duplicate_content_length : Bool
  priv mut route_code : Int32
  priv mut auth_cursor : Int32
  priv mut last_work_units : Int32
  priv mut total_work_units : UInt64
}

///|
#valtype
pub struct LunaHttp1Work {
  priv workspace : LunaHttp1Workspace
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaHttp1View {
  priv workspace : LunaHttp1Workspace
  priv epoch : UInt64
}

///|
pub extend LunaHttp1Route with @debug.Debug::{to_repr}

///|
pub extend LunaHttp1Route with Eq::{not_equal, equal}

///|
pub extend LunaHttp1WorkState with @debug.Debug::{to_repr}

///|
pub extend LunaHttp1WorkState with Eq::{not_equal, equal}

///|
pub extend LunaHttp1Progress with @debug.Debug::{to_repr}

///|
pub extend LunaHttp1Progress with Eq::{not_equal, equal}

///|
pub extend LunaHttp1Rule with @debug.Debug::{to_repr}

///|
pub extend LunaHttp1Rule with Eq::{not_equal, equal}

///|
pub extend LunaHttp1Issue with @debug.Debug::{to_repr}

///|
pub extend LunaHttp1Issue with Eq::{not_equal, equal}

///|
pub extend LunaHttp1Error with @debug.Debug::{to_repr}

///|
pub extend LunaHttp1Error with Eq::{not_equal, equal}
