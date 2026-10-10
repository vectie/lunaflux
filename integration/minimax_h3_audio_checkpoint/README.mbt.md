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

The prepare_joint_request entry point instead borrows the actual joint
AudioState, prepends the caller-owned packed stereo permutation/destandardization
program and retains the exact storage/plan receipt required by the complete
request. It neither downloads nor substitutes the latent. The matches_plan
query is pure geometry, not an allocation-identity test. The joint_workspace_bytes
query counts the decoder, both input-bridge regions and one waveform output;
raw weights, AOT modules, statistics and the external joint latent arena are
separate budgets. Close the consuming request before its input program and this
decoder.
