# MiniCPM-V 4.7 image runtime

This package composes the existing text decoder, vision execution and feature
insertion frame into one single-flight runtime. It targets the full selected
MiniCPM-V 4.7 35B-A3B checkpoint. It owns no HTTP, camera or robot logic.

`ModuleImages::read` consumes the seven existing text CUBINs plus
`vision_encoder.cubin` and `image_scatter.cubin`. Decoder module ownership is
shared with the text runtime through `ModuleSet`; there is no second loader or
model implementation. All CUDA compilation remains offline.

`Execution` borrows immutable weights and the external context. It owns both
streams, all frames, mutable allocations and request queues. `Resident` also
owns both weight sets and reuses one bounded upload scratch buffer sequentially.
`bootstrap.prepare_images` prepares both branches from one config, tokenizer
and header inspection, without reading full weight payloads or opening CUDA.

```mbt check
///|
async test "combined allocation payload is explicit without a device" {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let operators = @operators.DecoderOperators::new(spec, maximum_rows=32)
  let text = @bootstrap.text_program(
    operators,
    @bootstrap.compile_spark_attention(operators),
  )
  let program = @multimodal.Program::new(text)
  let vision = @vision.Program::new(
    spec,
    maximum_patches=16384,
    maximum_crop_patches=8192,
  )
  let budget = @minicpm_v47_image_runtime.Budget::new(
    program,
    vision,
    text_weight_bytes=69319779584L,
    vision_weight_bytes=1105972064L,
  )
  assert_eq(budget.total_allocation_bytes, 71008215248L)
  let execution = @minicpm_v47_image_runtime.Execution::new(program, vision)
  execution.close()
}
```

The default prepared profile has 71,008,215,248 bytes of explicit device
allocation payload: text/vision weights, both mutable storage sets, text output
workspace and the image map/header. Module, driver and queue overhead is extra.
This is a planned allocation total, not a measured Spark memory footprint.

For an image request:

1. Prepare the prompt and RGB pixels outside token scheduling. Build the vision
   `Input` from those same image plans, in prompt occurrence order.
2. Call `begin_image`. It validates decoder request parameters and enqueues
   vision, without waiting for it.
3. Call `poll` until true, or `wait` once. Vision completion then binds the
   feature map through two metadata uploads; feature values stay on device.
4. Drive `submit_next` and `poll`/`wait` for the standard decoder chunks/tokens.
   Observe emitted tokens and the finish reason through this owner.

`begin_text` disables the image map and starts a text request with the same
decoder. New requests are rejected until the current request finishes or is
cancelled. Completing vision alone does not allow its output storage to be
overwritten while the decoder is pending. `cancel` retires both queues and
permits reuse after a failed enqueue/readback. Close the whole owner before
releasing borrowed weights/context; close remains retryable after partial
startup and during active vision or text work. Do not independently close the
execution borrowed from a Resident.

Native scripted-driver coverage exercises nine-module preparation, asynchronous
vision-to-decoder ordering, image/text reuse, cancellation and failure recovery,
partial startup and active close. Token steps allocate no heap/native resources
and add no transfers beyond the existing metadata/sample law. These tests do
not evaluate CUDA model arithmetic. Full checkpoint upload, NVCC compilation,
Spark numerical/quality acceptance remains pending. Buffered HTTP media is
implemented through `serve-media`. The
[local media executable](../../cmd/minicpm_v47_generate/README.md) now prepares
PNG/JPEG queues and invokes this Resident through `run-media`; its no-GPU
inspection is verified, while real checkpoint generation remains unexecuted.
