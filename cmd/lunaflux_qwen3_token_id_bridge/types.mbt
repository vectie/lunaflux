///|
suberror BridgeRuntimeError {
  InvalidArgument(String)
  StartupIdentity(String)
  HttpRequest(String)
  NativeProtocol(String)
} derive(Debug)

///|
struct BridgeConfig {
  listen_addr : @socket.Addr
  runtime_addr : @socket.Addr
  tokenizer : @tokenizer.TokenizerSpec
  model_content_sha256 : String
  model_plan_sha256 : String
  max_input_tokens : Int
  max_output_tokens : Int
  max_token_id : Int
  max_context_tokens : Int
}

///|
struct HttpRequest {
  path : String
  verb : String
  body : String
}
