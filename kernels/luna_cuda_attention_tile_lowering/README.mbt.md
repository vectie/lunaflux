# Luna CUDA attention tile lowering

This is the NVIDIA-specific lowering boundary for the functional attention tile
compiler. It maps generic arithmetic and memory classes to CUDA primitives and
derives launch geometry from the immutable schedule. CUDA matrix instructions,
subgroup width, asynchronous copy, `sm_*` identity, and launch limits exist
only here and must never flow back into model, scheduler, strategy, semantic IR,
or generic schedule packages.

The functional schedule can request one paged K/V address calculation per
logical key row. CUDA realizes that portable reuse decision with subgroup
broadcast across the vector fragments that consume the row; no CUDA primitive
is reflected back into the optimizer or schedule vocabulary.

For optimizer-selected online-softmax storage reuse, CUDA maps the query-local
fold state to registers and aliases the dead staged-key arena for probability
and terminal state. The generic compiler sees only disjoint lifetimes and a
smaller working set.

The shared-key query-map region maps to one matrix-B fragment load per
reduction slice, reused by independent query-subtile accumulators. CUDA owns
the fragment dimensions and register realization; the generic pass only
records the map/fold interchange and shared operand.

Query-owned prefill binds an ordered two-element probability consumer and a
row-factored operand address plan. CUDA emits each probability word immediately
after its denominator contributions and reuses one shared-address row origin
and XOR mask across independent PV column products. It does not merge their
accumulator instruction regions, add a new publication, or reassociate expf,
F32 additions or BF16 rounding. The generic plans define ordering and layout;
the backend owns their packed-word and byte-address realization. Source-level
temporary removal is not a hardware register-count or latency claim.

Approximate query-owned exponential candidates use IDs equal to the strict
Cartesian schedule ID plus 30,000 (for example strict 322 versus approximate
30322). They require `allow_approximate_exponential=true` in addition to the
alternative-softmax permission and are restricted to BF16 prefill. The law is
consumed by the physical fold and final numeric emitter, so source and recipe
identities differ even when launch geometry is identical. Existing strict
autotune records cannot silently select the approximate route. No request-path
JIT, filesystem check or global fastmath setting is introduced.

The Qwen candidate exporter exposes the same permission explicitly as
`--prefill-approximate-exp2 --prefill-candidate 30322`. Its experimental selected
symbol is `lunaflux_attention_prefill_tile_compiler_exp2_v1`; the source recipe
records the exponential and subnormal law. Selecting an approximate ordinary,
wide-query or retained-lifetime identity without the flag is rejected before
model admission. This is an offline qualification route, not a serving default.

Generic owned-score decode candidates 460–471 cover owners 2/4/8, explicit
non-contracted/contracted arithmetic and KV32/KV64. Candidates 460/461 preserve
the incumbent dual-score arithmetic exactly while selecting an independently
retired K/V ring, so arithmetic and transport can be compared separately.
Historical candidate IDs retain their paired pipeline.

Grouped-head matrix decode candidates 480–483 realize the pure head-row fold
using ordered matrix QK/PV with the separately declared BF16 probability law.
480/481 use cooperative KV32/KV64 storage; 482/483 use independent two-stage
operand rings with the identical row-swizzled layout. Empty partitions retain
zero denominator/numerator and inactive instruction rows never store output.
The whole partial-plus-merge chain must pass an independent numerical oracle,
sanitizer and latency comparison before runtime selection. Source generation
and native unit tests alone do not establish physical correctness or a speedup.
