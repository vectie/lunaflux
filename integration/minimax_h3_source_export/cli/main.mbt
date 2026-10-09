///|
/// Offline text-to-video/audio source exporter. The digest is an externally
/// admitted model-content identity, not a hash invented by this command.
/// Request supports prompt row count; tokenization happens before deployment.
async fn main {
  let args = @env.args()
  guard args.length() == 12 else {
    fail(
      "usage: EXPORT_DIR TRANSFORMER_JSON AUDIO_VAE_JSON TEXT_CONFIG_JSON CONTENT_SHA256 HEIGHT WIDTH FRAMES STEPS TEXT_ROWS WORKSPACE_BYTES",
    )
  }
  let transformer = @fs.read_file(args[2]).binary()
  let audio_bytes = @fs.read_file(args[3]).binary()
  let text_bytes = @fs.read_file(args[4]).binary()
  let metadata = @config.parse_transformer(
    @utf8.decode(transformer),
    variant=Fl2va,
    content_digest=@identity.ContentDigest::from_sha256(args[5]),
    max_bytes=1048576,
    max_depth=32,
  )
  let audio = @config.parse_audio_vae(
    @utf8.decode(audio_bytes),
    max_bytes=1048576,
    max_depth=32,
  )
  let text = @text.parse_config(text_bytes, max_bytes=1048576)
  let model = @h3.build(metadata)
  let request = model
    .shapes()
    .resolve_request(
      height=@string.parse_int(args[6]),
      width=@string.parse_int(args[7]),
      requested_frames=@string.parse_int(args[8]),
      inference_steps=@string.parse_int(args[9]),
    )
  let plan = @loop.PreparedPlan::new(
    model,
    request,
    max_latent_elements=2147483647L,
  )
  let rows = @string.parse_int(args[10])
  guard rows <= text.max_positions else {
    fail("text rows exceed admitted position bound")
  }
  let text_plan = @text.TextEncoderPlan::new(rows~, epsilon=text.epsilon)
  let sequence = @sequence.Sequence::from_request(
    model,
    request,
    text_rows=rows,
    conditioning=[],
    max_rows=2147483647,
  )
  let bundle = @export.build(
    metadata,
    plan,
    sequence,
    audio,
    text_plan,
    text_config=text,
    visual_noise=0.0,
    audio_noise=0.0,
    vision_text_rows=[],
    max_metadata_bytes=1073741824L,
    max_workspace_bytes=@string.parse_int64(args[11]),
    tile_size=256,
    minimum_overlap=64,
    max_tiles=4096,
  )
  bundle.write(args[1])
  println("offline_source_modules=\{bundle.units.length()}")
  println("CUDA compilation and artifact admission are separate operations.")
}
