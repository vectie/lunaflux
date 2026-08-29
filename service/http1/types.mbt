///|
const LUNA_HTTP1_MIN_HEAD_BYTES : Int = 64

///|
const LUNA_HTTP1_MAX_HEAD_BYTES : Int = 65536

///|
const LUNA_HTTP1_MAX_BODY_BYTES : Int = 16777216

///|
const LUNA_HTTP1_MAX_STEP_WORK_UNITS : Int = 65536

///|
const LUNA_HTTP1_IDLE : Int = 0

///|
const LUNA_HTTP1_RECEIVING_HEAD : Int = 1

///|
const LUNA_HTTP1_SCANNING_LINE : Int = 2

///|
const LUNA_HTTP1_VALIDATING_REQUEST : Int = 3

///|
const LUNA_HTTP1_VALIDATING_HEADER_NAME : Int = 4

///|
const LUNA_HTTP1_VALIDATING_HEADER_VALUE : Int = 5

///|
const LUNA_HTTP1_VALIDATING_REQUIRED : Int = 6

///|
const LUNA_HTTP1_AUTH_BEGIN : Int = 7

///|
const LUNA_HTTP1_AUTH_WRITE : Int = 8

///|
const LUNA_HTTP1_AUTH_FINISH : Int = 9

///|
const LUNA_HTTP1_AUTH_PROGRESS : Int = 10

///|
const LUNA_HTTP1_AUTH_DECIDE : Int = 11

///|
const LUNA_HTTP1_RECEIVING_BODY : Int = 12

///|
const LUNA_HTTP1_FINALIZING : Int = 13

///|
const LUNA_HTTP1_READY : Int = 14

///|
const LUNA_HTTP1_VIEW_LEASED : Int = 15

///|
const LUNA_HTTP1_FAILED : Int = 16

///|
const LUNA_HTTP1_CLOSED : Int = 17

///|
const LUNA_HTTP1_HEADER_UNKNOWN : Int = 0

///|
const LUNA_HTTP1_HEADER_AUTHORIZATION : Int = 1

///|
const LUNA_HTTP1_HEADER_CONTENT_LENGTH : Int = 2

///|
const LUNA_HTTP1_HEADER_CONTENT_TYPE : Int = 3

///|
const LUNA_HTTP1_HEADER_TRANSFER_ENCODING : Int = 4

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
  priv max_head_bytes : Int
  priv max_body_bytes : Int
}

///|
#valtype
pub struct LunaHttp1StepBudget {
  priv work_units : Int
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
  priv mut phase : Int
  priv mut failure_rule : Int
  priv mut failure_issue : Int
  priv mut head_length : Int
  priv mut body_length : Int
  priv mut body_written : Int
  priv mut terminator_state : Int
  priv mut input_ended : Bool
  priv mut scan_cursor : Int
  priv mut scan_expect_lf : Bool
  priv mut line_start : Int
  priv mut line_end : Int
  priv mut request_line_complete : Bool
  priv mut parse_component : Int
  priv mut parse_index : Int
  priv mut route_responses : Bool
  priv mut route_chat : Bool
  priv mut header_kind : Int
  priv mut header_name_index : Int
  priv mut header_authorization : Bool
  priv mut header_content_length : Bool
  priv mut header_content_type : Bool
  priv mut header_transfer_encoding : Bool
  priv mut value_index : Int
  priv mut value_leading_space : Bool
  priv mut value_number : Int
  priv mut value_has_digit : Bool
  priv mut value_content_type : Bool
  priv mut credential_offset : Int
  priv mut credential_length : Int
  priv mut credential_padding : Bool
  priv mut authorization_seen : Bool
  priv mut content_length_seen : Bool
  priv mut content_type_seen : Bool
  priv mut duplicate_content_length : Bool
  priv mut route_code : Int
  priv mut auth_cursor : Int
  priv mut last_work_units : Int
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
