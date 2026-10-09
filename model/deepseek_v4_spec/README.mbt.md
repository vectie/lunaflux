# DeepSeek V4 semantic profiles

Schema v1 admits only exact official checkpoint identifiers and projects their
published configuration. It does not accept arbitrary DeepSeek-V4-like
dimensions or infer missing architecture fields.

Primary sources:

- the official DeepSeek-V4 checkpoint `config.json` files in the
  [`deepseek-ai`](https://huggingface.co/deepseek-ai) organization;
- the official `DeepSeek-V4-Flash-0731/inference/model.py` reference, which
  defines compressed attention, the learned indexer, hyper-connections, MoE,
  and DSpark composition;
- the DeepSeek-V4 technical report linked from those official model cards.

This package grants semantic metadata only. It does not grant an executable
model plan, device, kernel, or scheduler authority.
