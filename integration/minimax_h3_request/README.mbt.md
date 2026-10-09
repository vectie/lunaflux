# Prepared H3 joint request

`Request::prepare_model` connects already prepared model modules into one
request: encoder/refiner/reference producers, typed packed conditioning, every
scheduled complete prediction with both RF updates, VideoVAE, then AudioVAE.
`advance` performs no compilation, weight materialization or launch-list building.
Both decoders must complete before either final output callback is allowed.

The caller prepares all modules and buffers before transferring stages. Bind
downstream queues with the encoder/conditioning `with_destination` APIs; those
borrow **future destinations**, not completed results. `PreparedStage::own`
erases the individual frame type once at startup. Producer order must reflect
dependencies: vision before text, text before refiner, reference encoding before
the conditioning assembler. The assembler is appended by `prepare_model`.

Bind both decoders to this request's `LatentStorage.with_region` destinations,
not another allocation with the same shape. Audio preparation must pass
`prepared_plan=plan` and its packed-latent `AudioInputProgram`; standalone raw
channel-major decoding intentionally cannot join this full-request facade.
Both decoders must still be Prepared. Already submitted/completed/cancelled
owners are rejected before ownership transfer.

`waveform_samples` reports the actual output samples per channel. Like the
reference, AudioVAE preserves the rounded latent duration rather than trimming
to `request.audio_samples()`: 124 video frames produce 207 latent frames and
165,600 waveform samples per channel, not 165,333. Budget the decoder's actual
output bytes and use the published sample count for downstream audio handling.

At initial validation or prediction binding failure no stage transfers. Once
queue construction starts, the request retains all supplied stage owners,
including on failure. Always retain and close the request; failed close retries
the current resource, and consumers release before producers. Module, weight,
latent, rotary, prediction-metadata and scratch owners are borrowed and close
only after this request. Preparation/advance/close are single-host-thread calls.

This package is a numeric prepared-request entry point, not a file codec or HTTP
server. It does not supply fake conditioning or select a fallback decoder.
Generated CUDA remains unvalidated on physical hardware in this implementation
round; host tests and MoonBit compilation are not full-model inference proof.
