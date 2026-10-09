# Sigmoid correction CUDA source

This package renders a family-neutral correctness CUDA kernel for an F32
sigmoid followed by an F32 per-column correction addition. It deliberately
emits separate uncorrected and corrected outputs and grants no compilation or
execution authority. Selection, grouping, top-k, normalization, and scaling
remain downstream.
