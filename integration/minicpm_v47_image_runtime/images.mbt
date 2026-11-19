///|
/// Nine staged AOT CUBINs. Source/build ownership remains offline.
pub(all) struct ModuleImages {
  text : @runtime.ModuleImages
  vision : Bytes
  image_scatter : Bytes
}

///|
pub fn ModuleImages::read(
  root : @approved_fs.ApprovedRoot,
  maximum_module_bytes~ : Int64,
) -> Self raise {
  let text = @runtime.ModuleImages::read(root, maximum_module_bytes~)
  let read = fn(name : String) raise {
    let file = root.open_file(@approved_fs.ApprovedRelativeLocator::new(name))
    defer file.close()
    let bytes = file.read_immutable_snapshot(maximum_bytes=maximum_module_bytes)
    guard !bytes.is_empty() else { fail("empty CUBIN") }
    bytes
  }
  {
    text,
    vision: read("vision_encoder.cubin"),
    image_scatter: read("image_scatter.cubin"),
  }
}
