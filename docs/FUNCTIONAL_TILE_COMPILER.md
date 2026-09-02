# Functional LunaTile compiler

LunaFlux compiles model operations through an immutable, backend-neutral tile
pipeline:

    model graph
      -> strategy problem
      -> semantic tile IR
      -> functional map/fold schedule
      -> device lowering
      -> content-addressed AOT source

The compiler is functional to improve optimization freedom and determinism,
not to add request-time abstraction. All compiler allocation and rewriting
happen before serving. The token-step path receives only selected launch
records and preallocated resources.

## Language rules

- Semantic programs are typed value graphs in A-normal/SSA form. A binding is
  immutable and refers only to earlier values.
- Algebraic data types describe operations, numeric contracts, memory spaces,
  reductions, and capabilities. Exhaustive matching makes a newly added
  operation visible to every compiler pass.
- Compiler phases have distinct types. For attention these are Request,
  Selected, Semantic, and Scheduled; a backend cannot lower an unelaborated or
  unscheduled program.
- Ordered recurrence is explicit. Attention uses a sequential online-softmax
  fold over paged K/V tiles, surrounded by parallel maps over query tiles,
  heads, and output columns.
- Hardware capabilities and offline measurements are immutable inputs.
  Compiler code does not read a global device context, benchmark cache, file,
  environment variable, or clock.
- Vendor terms are forbidden above device lowering. Warp width, WMMA, PTX,
  CUDA shared-memory limits, and CUDA Graph do not appear in model or generic
  schedule semantics.

## Optimization model

Optimization is expressed as semantics-preserving transformation plus cost
extraction:

1. Normalize the model graph into typed semantic combinators.
2. Generate legal tile strategies from portable capabilities.
3. Apply equational fusion and layout rewrites to immutable values.
4. Derive map/fold schedules without assigning device instructions.
5. Select a schedule using deterministic static cost or exact offline
   autotune records.
6. Lower the selected schedule to one backend.

The intended rewrite engine is persistent and equality-based: alternatives
share immutable subgraphs, legality predicates are pure, and extraction is a
separate cost function. Rewrites must not encode a model family or device
name. Backend-specific peepholes run only after generic extraction.

Partial evaluation specializes static shapes, dtypes, page geometry, and
capabilities at AOT time. Content digests memoize pure pass outputs and form
cache keys; a cache miss changes compilation cost, never program meaning.

## Effect boundary

The pure compiler returns bytes and immutable metadata. An outer driver owns
effects such as model inspection, device probing, benchmark execution,
toolchain invocation, and file publication. Results re-enter the compiler only
as validated values, such as an AttentionAutotuneRecord.

This is an algebraic-effect boundary in practical form: the compiler core
describes required inputs and outputs but does not perform the effects. Tests
can therefore replace the outer interpreter while exercising the identical
compiler functions.

## Current attention path

The implemented path is:

- luna_attention_strategy: pure candidate generation and autotune selection;
- luna_attention_tile_ir: immutable attention semantic value graph;
- luna_attention_tile_schedule: backend-neutral parallel maps and ordered K/V
  fold;
- luna_attention_tile_compiler: phase-typed pure pass composition;
- luna_cuda_attention_tile_lowering: CUDA instruction and launch mapping;
- luna_cuda_attention_tile_source: deterministic matrix-tiled source;
- luna_cuda_attention_autotune: immutable device-profile measurements;
- luna_cuda_attention_tile_aot: pure end-to-end CUDA AOT composition.

The current CUDA lowering uses matrix tiles for both QK and probability-value
multiplication while preserving the generic online-softmax fold. Alternative
backends may lower the same schedule to MFMA, XMX, SIMD, or library calls.

## Required next generalizations

- Move projection, normalization, RoPE, MLP, sampling, and speculative verify
  into the same semantic/value/schedule split.
- Add equality-saturation or an equivalent bounded persistent rewrite engine
  for fusion and layout alternatives.
- Represent execution-graph buckets as a pure schedule algebra; CUDA Graph and
  HIP Graph remain backend interpreters.
- Keep quantization as typed numeric/layout contracts in generic IR, with
  packing and instruction selection confined to device lowering.
- Preserve a direct interpreter for semantic differential tests while making
  optimized backend lowering the production route.
