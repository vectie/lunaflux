///|
/// Amount of diagnostic detail emitted by the process.
pub(all) enum DiagnosticMode {
  Normal
  Verbose
} derive(Debug, Eq)

///|
/// Fields accepted by the service configuration constructor.
pub(all) enum ServiceField {
  ListenHost
  ListenPort
  MaxRequestBytes
  GracefulDrainMilliseconds
} derive(Debug, Eq)

///|
/// Bounded validation categories for service settings.
pub(all) enum ServiceIssue {
  Empty
  TooLong
  ContainsControlCharacter
  BelowMinimum
  AboveMaximum
} derive(Debug, Eq)

///|
/// Payload-safe service configuration validation failure.
pub(all) suberror ServiceConfigError {
  Invalid(field~ : ServiceField, issue~ : ServiceIssue)
} derive(Debug, Eq)

///|
/// Immutable process and diagnostic settings.
pub struct ServiceConfig {
  listen_host : String
  listen_port : Int32
  max_request_bytes : Int32
  graceful_drain_milliseconds : Int32
  diagnostic_mode : DiagnosticMode
} derive(Debug, Eq)

///|
/// Constructs validated service settings.
pub fn new(
  listen_host : String,
  listen_port : Int32,
  max_request_bytes : Int32,
  graceful_drain_milliseconds : Int32,
  diagnostic_mode : DiagnosticMode,
) -> ServiceConfig raise ServiceConfigError {
  if listen_host.length() == 0 {
    raise Invalid(field=ListenHost, issue=Empty)
  }
  if listen_host.length() > 253 {
    raise Invalid(field=ListenHost, issue=TooLong)
  }
  for char in listen_host {
    if char < ' ' || char == '\u{7f}' {
      raise Invalid(field=ListenHost, issue=ContainsControlCharacter)
    }
  }
  if listen_port < 1 {
    raise Invalid(field=ListenPort, issue=BelowMinimum)
  }
  if listen_port > 65535 {
    raise Invalid(field=ListenPort, issue=AboveMaximum)
  }
  if max_request_bytes < 1024 {
    raise Invalid(field=MaxRequestBytes, issue=BelowMinimum)
  }
  if max_request_bytes > 16777216 {
    raise Invalid(field=MaxRequestBytes, issue=AboveMaximum)
  }
  if graceful_drain_milliseconds < 0 {
    raise Invalid(field=GracefulDrainMilliseconds, issue=BelowMinimum)
  }
  if graceful_drain_milliseconds > 600000 {
    raise Invalid(field=GracefulDrainMilliseconds, issue=AboveMaximum)
  }
  {
    listen_host,
    listen_port,
    max_request_bytes,
    graceful_drain_milliseconds,
    diagnostic_mode,
  }
}

///|
pub fn ServiceConfig::listen_host(self : ServiceConfig) -> String {
  self.listen_host
}

///|
pub fn ServiceConfig::listen_port(self : ServiceConfig) -> Int32 {
  self.listen_port
}

///|
pub fn ServiceConfig::max_request_bytes(self : ServiceConfig) -> Int32 {
  self.max_request_bytes
}

///|
pub fn ServiceConfig::graceful_drain_milliseconds(
  self : ServiceConfig,
) -> Int32 {
  self.graceful_drain_milliseconds
}

///|
pub fn ServiceConfig::diagnostic_mode(self : ServiceConfig) -> DiagnosticMode {
  self.diagnostic_mode
}

///|
pub extend DiagnosticMode with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend DiagnosticMode with Eq::{not_equal, equal}

///|
pub extend ServiceField with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ServiceField with Eq::{not_equal, equal}

///|
pub extend ServiceIssue with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ServiceIssue with Eq::{not_equal, equal}

///|
pub extend ServiceConfigError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ServiceConfigError with Eq::{not_equal, equal}

///|
pub extend ServiceConfig with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ServiceConfig with Eq::{not_equal, equal}
