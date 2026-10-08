# extras/watch: keep-alive and telemetry

`glm53f-afd-watch` runs on the controller (the machine you run `./start.sh` from), once a minute, from cron or a
systemd timer. It reads the repo's `.env`, so it watches the stack `./start.sh up` started. Ported from
[v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0/extras/watch) to this version's
scripts.

```cron
* * * * * /path/to/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/extras/watch/glm53f-afd-watch >> $HOME/.local/state/glm-afd/watch/alerts.log 2>&1
```

or, as a systemd user timer:

```sh
systemd-run --user --unit=glm-afd-watch --on-active=60 --on-unit-active=60 \
  /path/to/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/extras/watch/glm53f-afd-watch
```

It prints only when it acts or needs you, so `alerts.log` (or cron's mail, or the journal) is your alert channel; pipe
it to ntfy, a chat webhook or similar if you like. Nothing starts the stack at boot: start it yourself with `./start.sh up`.

**Each run checks**
- `/health` answers 200, and no decode iteration has been running for more than `WATCH_STALL_S` (600 s);
- `/v1/models` lists `MODEL_NAME` or `MODEL_ALIAS`;
- a tiny streamed reply, thinking off, returns the run's nonce. `/health` can stay 200 for a stack whose expert
  node is gone, so the reply is the real check. A reply that waits past `WATCH_PROBE_TIMEOUT` (120 s) while `/health`
  shows tokens still flowing counts as busy (eight requests decoding, one queued), not as a failure;
- the attention log has no new line from the AFD watchdog (`afd: ...; in-flight requests fail, restart the serving
  stack`): the split cannot take a node back once that line is there.

**Keep-alive**
- After `WATCH_FAILS` (2) failed runs in a row it runs `./start.sh recover`, which does nothing to a healthy stack and
  otherwise stops the three containers with `SIGTERM` and starts them again, plus the MCDMA daemons when a link or a
  daemon is down.
- Circuit breaker: at most `WATCH_MAX_RECOVERIES` (3) recoveries in `WATCH_WINDOW_H` (6) hours, then it latches: it
  stops acting, alerts, and keeps recording. A recover that needs you (rc 3: a container still running after its
  `SIGTERM`, or a Spark whose memory did not come back) latches at once. Clear it with
  `rm ~/.local/state/glm-afd/watch/latched`.
- It holds off, without counting failures, while a stack stopped by hand stays stopped (`./stop.sh` leaves a stop
  marker; `./start.sh up` clears it), while another `start.sh up`/`recover`/`probe` or `stop.sh` runs (the control
  lock), for 10 minutes after the controller boots, and in maintenance mode:
  `echo maintenance > ~/.local/state/glm-afd/watch/mode` (`echo serving > ...` resumes).
- It never reboots anything and never sends `SIGKILL`. One run at a time: a run that finds another still going (a
  recover takes a few minutes) exits at once.

**Telemetry**, one JSON line per run in `~/.local/state/glm-afd/watch/telemetry/YYYY-MM-DD.jsonl`:
- the checks: `/health` code and counters, `/v1/models`, the reply's TTFT and time, why a run failed, what held it off;
- per host (over SSH, as the scripts): MemAvailable, swap, memory pressure, GPU utilization, memory, power, temperature
  and clock, the stack's containers, the `mcdma-rpcd` count, new kernel Xid/NVRM/OOM lines;
- the MCDMA links up, from the connect daemons (`connect`, and `connect2` at `MCDMA_INFLIGHT=4`); the attention log's new error and watchdog lines;
- a recover's rc, duration and output tail; an hourly 256-token decode bench while the stack is idle (`WATCH_BENCH=0`
  turns it off).

`glm53f-afd-report [days]` rolls the records up into a markdown summary: availability over the runs that were not on
hold, probe TTFT percentiles, the failed checks, the recoveries, decode drift, and per host the memory floor, swap,
pressure, GPU power and temperature. Use it to spot a slow leak before it bites, or to tune `PARALLEL`, `CACHE_GIB` and
`MIN_FREE_GIB_EXPERT`.

Settings come from `.env` (`WATCH_ENV_FILE` for another file); exported variables win. `STATE_DIR` moves the state
directory (default `$CTL_STATE/watch`); `WATCH_BASE` points it at another URL; `WATCH_RECOVER` replaces
`./start.sh recover` (it gets `GLM_AFD_CALLER=watch`, and must keep its exit codes: 0 recovered or healthy, 2 stopped by
hand, 3 needs you, 4 up but no reply, 75 another control command is running). `.env.example` lists the `WATCH_*` knobs.
