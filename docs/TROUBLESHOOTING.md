# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `glm53f-serve: built without the cuda feature` | The coordinator was built without `--features cuda,rdma` | Rerun `./build.sh coord`, which always builds with both features |
| Preflight: `MemAvailable … < 100 GiB` on a Spark | Another model is resident, or memory was stranded by a hard-killed MPS server | `nvidia-smi` and `docker ps` on that Spark. If nothing is running and memory is still low, reboot that Spark. |
| Preflight: `another GPU process is running` | Single-tenant rule | Stop the other workload. Never run two models on a Spark that serves ranks. |
| `/health` returns `503 … expert wire …` | A rank died or the fabric dropped. The coordinator latches until restarted. | `./start.sh recover` |
| `/health` 200 but completions hang or fail | A rank died less than ~2 min ago | `./start.sh probe`, then `./start.sh recover`. The watcher does this automatically. |
| Ranks never log `listening` | Fabric IPs missing, wrong rail, or a stale MPS server | `./start.sh status`; check `ip -br a` on the Spark and `$GLM_HOME/logs/ranks-latest/*.log` |
| Coordinator can't connect to ranks | RoCE GID or MTU mismatch, or the coordinator lacks an address on one of the rails | `ibv_devinfo`, `cat /sys/class/infiniband/*/ports/1/gids/*`, `ip -br a` on all three; MTU 9000 end to end |
| HTTP 400 on `tool_choice: required` | Upstream API limitation | Use `auto` |
| Much slower decode (~26 tok/s without drafter) | MPS is not running, so the two ranks time-slice | Check `pgrep -a nvidia-cuda-mps` on both Sparks; restart that Spark's tenant |
| The long prompt's TTFT is slower than you'd like | v1.1.0 `--decode-share 0.2` gives other streams a share | `EXTRA_ARGS=--decode-share 0` in `.env` if you serve one user |
