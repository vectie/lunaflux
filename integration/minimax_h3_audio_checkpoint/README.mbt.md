# Checkpoint-backed AudioVAE preparation

`CheckpointAudioDecoder` joins the original H3 AudioVAE's 16 streamed weight
groups with all seven upsample/AMP stages, the frontend and waveform tail.
`AudioDecoderMemoryPlan` resolves retained scratch, bridges, normalization
caches, input and output before device allocation. Raw weights and AOT module
storage remain separately budgeted; callers subtract them from the device cap.

The same `AudioDecoderSources` builds component and complete-request AOT
modules. No runtime compilation, Python wrapper or per-kernel host wait is
introduced. Immutable raw weights remain borrowed under `WeightLease` until
all requests and this decoder have closed.

Close requests first, then decoder, lease, streamed weight owner, stream and
context. Failed preparation retains partial modules/stages for deterministic
close. A prepared raw audio request consumes channel-major destandardized F32
latents; it is not a substitute for the packed full-denoiser `AudioInputProgram`
join or proof of full text-to-video/audio generation.
