///|
#borrow(executable, launch, stdout_path, stderr_path, drain_trigger, child_pid, drain_acknowledged, child_exit_code)
extern "c" fn supervise_qwen3_runtime(
  executable : Bytes,
  launch : Bytes,
  stdout_path : Bytes,
  stderr_path : Bytes,
  drain_trigger : Bytes,
  child_pid : Ref[Int],
  drain_acknowledged : Ref[Int],
  child_exit_code : Ref[Int],
) -> Int = "lunaflux_qwen3_supervise_runtime"

///|
fn qwen3_supervisor_usage() -> Unit {
  println(
    "usage: qwen3_bf16_serving_supervisor ABSOLUTE_LUNAFLUX_EXECUTABLE ABSOLUTE_DEPLOYMENT_ROOT#sha256=HEX ABSOLUTE_NEW_STDOUT ABSOLUTE_NEW_STDERR ABSOLUTE_NEW_DRAIN_TRIGGER",
  )
  @sys.exit(2)
}

///|
fn main {
  let args = @env.args()
  if args.length() != 6 {
    qwen3_supervisor_usage()
    return
  }
  let child_pid = Ref(-1)
  let drain_acknowledged = Ref(0)
  let child_exit_code = Ref(-1)
  let status = supervise_qwen3_runtime(
    @utf8.encode(args[1]),
    @utf8.encode(args[2]),
    @utf8.encode(args[3]),
    @utf8.encode(args[4]),
    @utf8.encode(args[5]),
    child_pid,
    drain_acknowledged,
    child_exit_code,
  )
  println("schema=lunaflux-qwen3-native-supervisor.v1")
  println("supervisor_status=" + status.to_string())
  println("child_pid=" + child_pid.val.to_string())
  println("drain_acknowledged=" + drain_acknowledged.val.to_string())
  println("child_exit_code=" + child_exit_code.val.to_string())
  if status != 0 || drain_acknowledged.val != 1 || child_exit_code.val != 0 {
    @sys.exit(1)
  }
  println("child_closed=1")
}
