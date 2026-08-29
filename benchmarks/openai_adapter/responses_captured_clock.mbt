///|
fn OpenAIResponsesTrialObserver::collector_submit_observed(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  input_tokens : Int64,
) -> Unit raise OpenAIBaselineAdapterError {
  match self.captured_timestamp {
    Some(timestamp) =>
      self.collector.submit_captured(
        request_ordinal~,
        input_tokens~,
        timestamp~,
      ) catch {
        _ => raise OpenAIBaselineCollectorRejected
      }
    None =>
      self.collector.submit(request_ordinal~, input_tokens~) catch {
        _ => raise OpenAIBaselineCollectorRejected
      }
  }
}

///|
fn OpenAIResponsesTrialObserver::collector_admit_observed(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
) -> Unit raise OpenAIBaselineAdapterError {
  match self.captured_timestamp {
    Some(timestamp) =>
      self.collector.admit_captured(request_ordinal~, timestamp~) catch {
        _ => raise OpenAIBaselineCollectorRejected
      }
    None =>
      self.collector.admit(request_ordinal~) catch {
        _ => raise OpenAIBaselineCollectorRejected
      }
  }
}

///|
fn OpenAIResponsesTrialObserver::collector_first_token_observed(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
) -> Unit raise OpenAIBaselineAdapterError {
  match self.captured_timestamp {
    Some(timestamp) =>
      self.collector.first_token_captured(request_ordinal~, timestamp~) catch {
        _ => raise OpenAIBaselineCollectorRejected
      }
    None =>
      self.collector.first_token(request_ordinal~) catch {
        _ => raise OpenAIBaselineCollectorRejected
      }
  }
}

///|
fn OpenAIResponsesTrialObserver::collector_terminate_observed(
  self : OpenAIResponsesTrialObserver,
  request_ordinal : Int,
  outcome : @evidence.BenchmarkRequestOutcome,
  generated_tokens : Int64,
) -> Unit raise OpenAIBaselineAdapterError {
  match self.captured_timestamp {
    Some(timestamp) =>
      self.collector.terminate_captured(
        request_ordinal~,
        outcome~,
        generated_tokens~,
        timestamp~,
      ) catch {
        _ => raise OpenAIBaselineCollectorRejected
      }
    None =>
      self.collector.terminate(request_ordinal~, outcome~, generated_tokens~) catch {
        _ => raise OpenAIBaselineCollectorRejected
      }
  }
}
