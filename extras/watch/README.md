# extras/watch: keep-alive and telemetry

`glm53f-afd-watch` is meant to run every 2 minutes from cron on the controller box. It reads the repo's `.env`.

```cron
*/2 * * * * /path/to/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/extras/watch/glm53f-afd-watch >> $HOME/.glm53f-afd-watch/alerts.log 2>&1
@reboot sleep 240 && /path/to/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/start.sh >> $HOME/.glm53f-afd-watch/boot.log 2>&1
```

It prints only when it acts or needs you, so `alerts.log` (or cron's mail) is your alert channel. You can also pipe it to Discord, ntfy or similar.

**Keep-alive**
- Each run checks `/health` and a real completion.
- After 2 consecutive failures it runs `./start.sh recover`.
- A circuit breaker allows at most 3 recoveries in 6 hours, then latches: it stops acting but keeps collecting data. Clear it with `rm ~/.glm53f-afd-watch/latched`.
- It never reboots anything and never hard-kills MPS.
- `echo maintenance > ~/.glm53f-afd-watch/mode` pauses it; `echo serving > …` resumes.

**Telemetry** is written to `~/.glm53f-afd-watch/telemetry/YYYY-MM-DD.jsonl`, one line per run:
- per host: MemAvailable, swap, PSI, GPU util/mem/power/temperature/clock, MPS server count, unit states, new kernel Xid/NVRM/OOM lines;
- from the coordinator log: drafter acceptance and tokens per window, slot-pressure evictions, host-cache stores and restores, error lines;
- probe TTFT, and an hourly 256-token decode benchmark (skipped while the lane is busy).

`glm53f-afd-report [days]` rolls these up into a markdown summary: availability, TTFT percentiles, decode drift, drafter acceptance, and memory, power and temperature extremes per host. Use it to tune slots, context, drafter settings, or to decide whether a third or fourth Spark is worth it.

Set `STATE_DIR` to move the state directory.
