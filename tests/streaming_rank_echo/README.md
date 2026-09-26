# Streaming rank protocol fixture

This native Linux test child uses the production rank configuration and
control-frame codecs with synthetic metadata. Rank zero finishes a transfer
after one poll; rank one finishes after three. It exercises the parent's
all-rank publication barrier without CUDA, NCCL, model files, or deployment
authority. Run it only through `tests/streaming_rank_group_e2e`.
