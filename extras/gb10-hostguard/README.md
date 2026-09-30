# extras/gb10-hostguard: keep a DGX Spark from hard-hanging

**Why:** on GB10, CUDA allocations come from unified memory and are *not* charged to a container's cgroup, so `docker --memory` limits do nothing. If two models end up resident, or one grows past what is free, the kernel thrashes for tens of minutes. Ping still answers, but SSH never completes and the only way out is a power cycle. We hit this once with two lab models on one Spark.

`gb10-hostguard` is a small root daemon (Python stdlib only, `mlockall`, `Nice=-15`, `OOMScoreAdjust=-1000`). It samples memory 1–4× per second and never forks at the critical moment.

| Rule | Default (gb10 / rtx) |
|---|---|
| MemAvailable < `HARD_GIB` once → kill GPU tenants (whole docker container via `cgroup.kill`), newest first | 4 / 5 GiB |
| MemAvailable < `KILL_GIB` for 3 samples → same | 6 / 10 GiB |
| PSI memory "full" avg10 > 25 sustained for 60 s → same | |
| More than one GPU tenant older than 20 s → kill the newest | on |
| Warn below `WARN_GIB` | 12 / 24 GiB |

A tenant is the cgroup of a process holding `/dev/nvidia*`. Only docker scopes and user sessions can be killed. `sshd`, `systemd*`, `nvidia-smi`, persistence and MPS daemons, `dcgm` and `dockerd` are never touched. Events go to `/var/log/gb10-hostguard/events.log`, with a 5 s sample log for forensics.

`install.sh gb10|rtx` also installs host hardening. **Read it before running it:**

- `OOMScoreAdjust=-1000` and `MemoryMin` for sshd and journald;
- `vm.min_free_kbytes=1 GiB`, `watermark_scale_factor=200`;
- `kernel.panic=30`, `panic_on_oops=1`, `softlockup_panic=1`. A locked-up kernel **reboots itself after 30 s**.
- the systemd hardware watchdog (`RuntimeWatchdogSec=60`), or softdog where none exists;
- a persistent journal.

```bash
sudo ./install.sh gb10     # on each DGX Spark
sudo ./install.sh rtx      # on the RTX coordinator (optional)
```

Tested on our kit with a CUDA memory hog allocating ~7 GiB/s. The guard killed it at 5.8 GiB free, with no kernel OOM and SSH answering in 0.35 s. Serving this model, each Spark keeps 27–35 GiB free, far above the floors.

Set `REQUIRE_GUARD=1` in `.env` to make preflight refuse to start without it.
