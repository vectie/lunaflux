# BF16 binary CUDA source

This package renders counts-bounded, family-neutral BF16 binary arithmetic.
The current addition primitive converts both inputs to F32, adds with explicit
round-to-nearest semantics, and rounds once to BF16. It grants no launch or
execution authority.
