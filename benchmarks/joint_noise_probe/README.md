# CPU-only initial-noise attribution

This diagnostic reads the original small H3 configurations and a supplied
model association label, constructs the ordinary generic noise plan, and writes
bounded initial video/audio F32 tensors. It loads no model weights and opens
no CUDA context. It reports the obsolete v1 effective key only to reproduce a
historical experiment under the v2 explicit-seed semantics; this is not payload
authentication or a production compatibility fallback.

Run `moon run --target native benchmarks/joint_noise_probe CONFIG_ROOT LABEL
SEED NEW_OUTPUT_DIRECTORY`. The fixed geometry is the existing 32×32/120-frame,
five-evaluation correctness fixture. Compare tensor values, not just CLI seeds:
v1 included execution-plan identity in its RNG key, and even changing audio
publication rounding rerolled every latent. V2 uses only seed/stream/counter.

Neither algorithm is PyTorch CUDA RNG-compatible. Whole-model comparisons
require identical initial tensors as well as weights, conditioning and steps.
