#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

binding=service/online_tcp/listener_bind.mbt
test -f "$binding"
test "$(rg -n 'TcpServer\(' service/online_tcp --glob '*.mbt' | wc -l | tr -d ' ')" -eq 1
rg -Fq 'TcpServer(addr, reuse_addr=true)' "$binding"
rg -Fq '`SO_REUSEPORT`' "$binding"
for owner in \
  service/online_tcp/server_prepare.mbt \
  service/online_tcp/openai_pool_prepare.mbt \
  service/online_tcp/openai_server_prepare.mbt \
  service/online_tcp/framed_pool_prepare.mbt \
  service/online_tcp/endpoint_prepare.mbt; do
  rg -Fq 'bind_luna_online_tcp_listener(addr)' "$owner"
done

printf '%s\n' 'online listener restart boundary gate passed'
