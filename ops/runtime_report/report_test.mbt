///|
test "public runtime report surface remains admission-derived and typed" {
  let build_report : (@descriptor_file.RuntimeDescriptorAdmission) -> @runtime_report.RuntimeReport raise @runtime_report.RuntimeReportError = @runtime_report.build
  let render_report : (@runtime_report.RuntimeReport) -> String = @runtime_report.render
  ignore(build_report)
  ignore(render_report)
  debug_inspect(
    @runtime_report.DeviceBytesOverflow,
    content="DeviceBytesOverflow",
  )
}
