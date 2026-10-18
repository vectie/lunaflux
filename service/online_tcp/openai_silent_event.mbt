///|
/// A renderer may consume a semantic token without publishing visible bytes
/// (e.g. a reasoning section). That event still needs its delivery ACK, not
/// an empty socket write or a connection cut. This applies equally to SSE and
/// buffered JSON; a terminal event must retain its terminal framing.
fn luna_openai_silent_nonterminal(length : Int, terminal : Bool) -> Bool {
  length == 0 && !terminal
}
