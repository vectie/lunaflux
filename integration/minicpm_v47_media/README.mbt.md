# MiniCPM-V 4.7 media preparation

This package joins decoded RGB8 frames, the pinned content-list prompt,
processor, canvas layout and vision input. It owns no filesystem, HTTP, CUDA,
camera SDK or robot control. A camera adapter and a PNG/JPEG file resolver both
provide `media/rgb8.Image`; the model preparation path is identical.

```mbt check
///|
fn _prepare_camera_frame(
  spec : @model.Spec,
  tokenizer : @tokenizer.TokenizerSpec,
  vision : @vision.Program,
  frame : @rgb8.Image,
) -> @minicpm_v47_media.Prepared raise {
  @minicpm_v47_media.prepare(
    spec,
    tokenizer,
    vision,
    "Describe the observation before proposing a skill.",
    [Text("What is in front of the robot?"), Image(frame)],
    maximum_tokens=4096,
    maximum_prepared_pixel_bytes=33554432L,
  )
}
```

Image plans, prompt placeholders, crop layout, pixel packing and feature counts
are derived together, in content occurrence order. Crop/token/payload/working
budgets are checked before resize. Text-only content lists retain the same
newline/trim template and return a text prompt for reuse in the image runtime.

`prepare_requests` reads the explicit local queue format
`lunaflux.minicpm-v47-media-requests.v1`, shown in the
[example](../../cmd/minicpm_v47_generate/media-requests.example.json). The caller
supplies a path-to-RGB resolver. It may use an approved media root, but this
package itself cannot fetch paths or URLs. Content parts are `text`/`text` or
`image`/`path`; sampling/tools/unknown fields are rejected. One queue has at most
32 requests, each with up to 64 parts/eight images. The queue-wide prepared
pixel ceiling is separate from per-image resize working memory (default
256 MiB). `Plan::preparation_bytes` exposes the latter numeric payload bound;
decoded source frames and small object headers remain caller-owned.

The local executable's `inspect-media` prepares the full selected checkpoint's
headers/tokenizer and PNG/JPEG inputs without CUDA or weight payload reads.
`run-media` uses the same preparation and the combined nine-module Resident.
Media decoding/resizing and prompt construction finish before token execution.
`serve-media` accepts inline OpenAI image parts; the original `serve` keeps its
text-only session. Remote media, video sampling and E1 camera SDK integration
remain separate adapters.

Tests use actual tokenizer assets and known RGB/PNG/JPEG fixtures. Native
scripted-driver coverage continues through multi-image vision, decoder/EOS and
text reuse; it does not compute CUDA model arithmetic or establish grounding.

`normalize_chat` now adapts buffered OpenAI chat content strings/lists and inline
PNG/JPEG `image_url` data URIs. It retains generation fields for canonical
admission, rejects duplicate object keys, and returns normalized text JSON plus
typed RGB content. Local image IDs, outer trim and content-list newlines match
the pinned template. `system_present` records whether the original request had
a system message; it avoids inserting an empty system role when none was sent.
Call `prepare` with that flag and the admitted prompt budget. Remote/file URLs,
tools/video parts and unsupported image detail are rejected; this adapter never
fetches media. `serve-media` connects it to the combined runtime. Video sampling
and the E1 camera SDK remain separate adapters.
