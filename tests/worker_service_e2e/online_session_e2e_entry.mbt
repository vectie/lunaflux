///|
fn main {
  match @env.args() {
    [_, executable, failure_executable, persistent_executable] =>
      try! run(
        @utf8.encode(executable),
        Some(@utf8.encode(failure_executable)),
        Some(@utf8.encode(persistent_executable)),
      )
    [_, executable, failure_executable] =>
      try! run(
        @utf8.encode(executable),
        Some(@utf8.encode(failure_executable)),
        None,
      )
    [_, executable] => try! run(@utf8.encode(executable), None, None)
    _ => println("worker executable path is required")
  }
}
