# Exact DeepSeek V4 configuration admission

This family-owned parser admits only the five official schema-v1
`config.json` documents. It enforces byte and nesting limits, UTF-8 validity,
duplicate-key rejection, a closed top-level and nested field vocabulary, exact
hybrid-attention ratios, exact FP8 quantization metadata, and exact DSpark
fields. Remote-code metadata and unknown future semantics fail closed.

The parser returns `DeepSeekV4ModelMetadata` bound to a content digest that the
caller has already verified. It performs no file access, code loading, weight
materialization, plan construction, or device work, and intentionally does not
modify or widen the shared selected-model config reader.
