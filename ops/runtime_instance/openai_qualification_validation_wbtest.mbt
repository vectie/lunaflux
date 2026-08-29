///|
test "qualification provenance never grants production readiness" {
  assert_eq(health_for_phase(OpenAiQualificationReady), RuntimeHealthy)
  assert_eq(
    readiness_for_phase(OpenAiQualificationReady, true),
    RuntimeNotReady,
  )
}

///|
test "qualification contract rejects missing and header-hostile credentials" {
  let credentials : FixedArray[FixedArray[Byte]] = [
    [],
    [b'a', b'\r'],
    [b'a', b'\n'],
    [b'a', b' '],
  ]
  for credential in credentials {
    try
      RuntimeOpenAiQualificationValidationContract::new(credential, 1, 1, 1024)
    catch {
      RuntimeOpenAiQualificationValidationFailed(OpenAiQualificationContract) =>
        ()
      _ => abort("unexpected qualification contract rejection")
    } noraise {
      _ => abort("hostile qualification credential admitted")
    }
  }
}

///|
test "Responses evidence markers must remain in canonical order" {
  assert_true(
    qualification_markers_ordered(
      "response.created response.output_text.delta response.usage " +
      "response.completed data: [DONE]",
    ),
  )
  assert_false(
    qualification_markers_ordered(
      "response.created response.completed response.output_text.delta " +
      "response.usage data: [DONE]",
    ),
  )
}
