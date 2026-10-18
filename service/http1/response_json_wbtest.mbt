///|
test "JSON response heads publish exact decimal lengths without a body copy" {
  for budget in [1, 17, LUNA_HTTP1_MAX_STEP_WORK_UNITS] {
    let workspace = LunaHttp1ResponseWorkspace::new(
      LunaHttp1ResponseStepBudget::new(budget),
    )
    for length in [0, 9, 10, 99, 100, 2147483647] {
      for failed in [false, true] {
        let work = workspace.begin_json(content_length=length, failed~)
        let view = luna_http1_response_ready(work, budget)
        let status = if failed { 500 } else { 200 }
        let reason = if failed { "Internal Server Error" } else { "OK" }
        luna_http1_assert_response(
          view,
          status,
          "HTTP/1.1 \{status} \{reason}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: \{length}\r\n\r\n",
        )
        view.release()
      }
    }
    try workspace.begin_json(content_length=-1) catch {
      LunaHttp1Failed(rule=LunaHttp1Limits, issue=LunaHttp1Invalid) => ()
      _ => fail("unexpected content length failure")
    } noraise {
      _ => fail("negative content length accepted")
    }
  }
}
