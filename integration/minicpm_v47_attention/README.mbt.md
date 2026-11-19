# MiniCPM-V 4.7 paged attention compilation

This adapter supplies the selected 16-query/2-KV-head, width-256 geometry to the
existing LunaFlux paged-attention compiler. It generates ordinary prefill and
decode sources over BF16 K/V planes. It introduces no replacement cache or
attention implementation. The common ABI now admits width 256, while individual
advanced lowerings retain their narrower constraints.

Hardware target, target identity, compiler capabilities and optional genuine
tuning records are explicit inputs. The preparation command declares an
**unqualified software profile**: sm121, 256 threads, 49152 shared bytes, one
pipeline stage and workgroup target, no async copy. That profile produces
unmeasured scalar prefill/decode sources; it is neither a hardware observation
nor a selected performance configuration. More parallel/partitioned variants
remain subsequent integration and qualification work.

For one 4096-token context and 16-token pages, each full-attention layer needs
256 physical pages, 8192 BF16 values per page per plane, and 8,388,608 bytes for
both K and V. Input rows follow the existing 5120-value Q/K/V ABI.

`Program` owns the complete full-attention stage sequence: three BF16
projections, centered Q/K norms, shared rotary coefficients, in-place Q/K
rotation, bit-preserving QKV packing, the existing paged KV writer, read-only
prefill/decode attention, head-interleaved output gate and output projection.
Each phase has 13 ordered launches. Twelve common entrypoints share
`attention_common.cu`; prefill and decode retain separate translation units to
preserve helper namespaces. Preparation and runtime consume the same program.

At 32 rows, one caller-owned scratch arena uses 1,740,800 bytes and can be reused
serially across layers. Persistent K/V planes remain outside this arena. `Frame`
borrows modules, weights, metadata, input/output, workspace and cache planes;
cache offsets support views into the existing multi-layer arena. It owns only
14 function handles and prepared launch descriptions. Create enclosing ordered
executors from `Frame::launches`; close those executors before the frame, then
release the borrowed resources. Close partially prepared frames on failure.

The input is already decoder-normalized; residual addition belongs to the
enclosing decoder. The caller supplies valid `StepCounts`, CSR rows, sequence
lengths, causal positions and uniquely owned writable pages. Ordered KV stores
publish all current K/V rows before attention reads; causal masking prevents a
query from reading later positions in the same frame. The prepared frame has
not yet been executed on CUDA hardware.

Rotary T/H/W coordinates are separate from linear causal/cache token positions.
The canvas sequence builder must provide the former; do not substitute one
coordinate for the cache address. Existing online-softmax FP32 attention is a
different floating program from the eager fallback's materialized BF16 scores
and probabilities. Compare actual logits/task quality rather than asserting
bitwise parity. CPU source/ABI checks do not prove CUDA compilation or execution.
