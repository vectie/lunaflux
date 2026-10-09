///|
test "checkpoint inventory is bounded metadata, not an executable file list" {
  let hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  let line = hash + "  model-00001.safetensors"
  let inventory = @checkpoint_inventory.CheckpointInventory::parse(
    line + "\n" + hash + "  sub/model-00002.safetensors\n",
    maximum_bytes=1024,
    maximum_shards=2,
  )
  assert_eq(inventory.shards().length(), 2)
  let unicode = hash + "  中文.safetensors"
  try
    @checkpoint_inventory.CheckpointInventory::parse(
      unicode,
      maximum_bytes=unicode.length(),
      maximum_shards=1,
    )
  catch {
    @checkpoint_inventory.CheckpointInventoryError::InvalidLimits => ()
    e => raise e
  } noraise {
    _ => fail("UTF-8 inventory byte limit treated as code units")
  }
  for
    text in [
      "",
      hash + "  config.json",
      hash + " model.safetensors",
      line + "\n" + line,
    ] {
    try
      @checkpoint_inventory.CheckpointInventory::parse(
        text,
        maximum_bytes=1024,
        maximum_shards=2,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid checkpoint inventory accepted")
    }
  }
  try
    @checkpoint_inventory.CheckpointInventory::parse(
      line,
      maximum_bytes=line.length() - 1,
      maximum_shards=1,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("inventory byte limit ignored")
  }
  try
    @checkpoint_inventory.CheckpointInventory::parse(
      line + "\n" + hash + "  other.safetensors",
      maximum_bytes=1024,
      maximum_shards=1,
    )
  catch {
    @checkpoint_inventory.CheckpointInventoryError::TooManyShards => ()
    e => raise e
  } noraise {
    _ => fail("inventory shard limit ignored")
  }
}
