///|
/// Exact seven-module AOT boundary shared with the offline source/build layout.
/// These are CUBIN bytes, never CUDA source to compile during startup.
pub(all) struct ModuleImages {
  envelope : Bytes
  recurrent : Bytes
  moe : Bytes
  attention_common : Bytes
  attention_prefill : Bytes
  attention_decode : Bytes
  text : Bytes
}

///|
/// Read each staged CUBIN once. Deployment owns file immutability/identity;
/// loading does not hash payloads or treat build receipts as qualification.
pub fn ModuleImages::read(
  root : @approved_fs.ApprovedRoot,
  maximum_module_bytes~ : Int64,
) -> Self raise {
  guard maximum_module_bytes > 0L else { fail("invalid CUBIN size ceiling") }
  let read = fn(name : String) raise {
    let file = root.open_file(@approved_fs.ApprovedRelativeLocator::new(name))
    defer file.close()
    let bytes = file.read_immutable_snapshot(maximum_bytes=maximum_module_bytes)
    guard !bytes.is_empty() else { fail("empty CUBIN") }
    bytes
  }
  {
    envelope: read("decoder_envelope.cubin"),
    recurrent: read("recurrent_branch.cubin"),
    moe: read("moe.cubin"),
    attention_common: read("attention_common.cubin"),
    attention_prefill: read("attention_prefill.cubin"),
    attention_decode: read("attention_decode.cubin"),
    text: read("text_io.cubin"),
  }
}
