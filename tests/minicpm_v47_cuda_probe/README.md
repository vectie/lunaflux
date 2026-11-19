# Selected-model CUDA deployment probe

This standalone native diagnostic loads all nine compiled v19 modules through
the real CUDA driver, then launches the production image-scatter kernel1,000
times. It verifies two distinct feature rows and preservation of an unmapped
text row, for6,144 exact BF16 word checks. It owns its context retention,
modules and device buffers and releases them on success/error.

On the Spark:

```sh
c++ -std=c++17 -O2 -I/usr/local/cuda-13.0/include probe.cpp -L/usr/local/cuda-13.0/lib64/stubs -lcuda -o minicpm-cuda-probe
./minicpm-cuda-probe CUBIN_ROOT
```

The link-time stub resolves to the installed real driver at execution. The
probe requires observed compute capability12.1 and uses no JIT, checkpoint,
Python or scripted driver. The real GB10 run passed on2026-11-17. This proves
module loading and the selected copy/scatter operation only. Decoder/vision
arithmetic, complete-model outputs and throughput remain separate gates.
