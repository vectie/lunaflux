///|
/// External SIGTERM regression for the runtime's signal-wait accounting.
/// Protected cleanup must remain runnable while waiting for new asynchronous
/// I/O. No model, GPU, worker subprocess or production busy loop is involved.
async fn run_async_signal_cleanup_evidence() -> Unit {
  defer @async.protect_from_cancel(() => {
    println("signal-cleanup starts")
    let listener = @socket.TcpServer(@socket.Addr::parse("127.0.0.1:0"))
    defer listener.close()
    println("signal-cleanup completes")
  })
  println("signal-cleanup ready")
  while true {
    @async.sleep(1000)
  }
}
