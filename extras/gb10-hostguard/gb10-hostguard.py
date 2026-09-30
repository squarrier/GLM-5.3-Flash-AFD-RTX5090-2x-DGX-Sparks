#!/usr/bin/env python3
"""gb10-hostguard: on-box protector for GPU hosts running big models (DGX Spark / GB10, or any CUDA box).

Why: on GB10 unified memory, CUDA allocations are NOT charged to the container
cgroup, so Docker --memory caps cannot stop a model from starving the host.
A starved GB10 livelocks: ping answers, sshd never completes a banner, and only
a physical power cycle recovers it. This daemon keeps the box reachable by
killing GPU workloads early, without forking anything at the critical moment.

Rules (all SIGKILL / cgroup.kill, no exec on the hot path):
  1. Memory floor: MemAvailable < HARD_GIB once, or < KILL_GIB for KILL_SAMPLES
     consecutive 1 s samples, or PSI memory "full avg10" > PSI_FULL_KILL for
     KILL_SAMPLES samples -> kill every GPU tenant (newest first, re-check between).
  2. Single tenant: more than one GPU tenant (distinct cgroups holding
     /dev/nvidia*) -> kill the newest tenant(s) until one remains.
  3. Everything is logged to /var/log/gb10-hostguard/ (persistent across reboot),
     plus a 5 s sample log so a future hang leaves forensics behind.

A "tenant" is the cgroup of a process holding an open /dev/nvidia* fd. Only
cgroups under system.slice/docker-*.scope and user sessions
are eligible for killing; anything in PROTECT_PATTERNS is never touched.
Config: /etc/default/gb10-hostguard (KEY=VALUE).
"""
import ctypes, os, re, sys, time, json

CONF = {
    "KILL_GIB": "3.0", "HARD_GIB": "1.5", "WARN_GIB": "8.0", "KILL_SAMPLES": "3",
    "PSI_FULL_KILL": "25", "INTERVAL": "1.0", "TENANT_SCAN_EVERY": "5",
    "SAMPLE_LOG_EVERY": "5", "ENFORCE_SINGLE_TENANT": "1", "DRY_RUN": "0",
    "PROTECT_PATTERNS": "^(sshd?|systemd.*|nvidia-persiste.*|nvidia-smi|nvidia_gpu_expo.*|dcgm.*|dockerd|containerd.*|Xorg|gb10-hostguard|nvidia-cuda-mps.*)$",
    "TENANT_MIN_AGE_S": "20", "PSI_SUSTAIN_S": "60",
}
_HZ = os.sysconf("SC_CLK_TCK")


def _uptime_ticks():
    return float(open("/proc/uptime").read().split()[0]) * _HZ
LOGDIR = "/var/log/gb10-hostguard"
CG = "/sys/fs/cgroup"


def load_conf():
    try:
        for line in open("/etc/default/gb10-hostguard"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                CONF[k.strip()] = v.strip().strip('"')
    except FileNotFoundError:
        pass


def lock_memory():
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.mlockall(3)  # MCL_CURRENT | MCL_FUTURE
    except Exception:
        pass


_events = None
_samples = None


def log(kind, **kw):
    global _events
    kw.update(ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), kind=kind)
    line = json.dumps(kw, sort_keys=True)
    try:
        if _events is None:
            _events = open(os.path.join(LOGDIR, "events.log"), "a", buffering=1)
        _events.write(line + "\n")
        os.fsync(_events.fileno())
    except Exception:
        pass
    print(line, flush=True)


def sample_log(d):
    global _samples
    try:
        if _samples is None or _samples.tell() > 20_000_000:
            path = os.path.join(LOGDIR, "samples.log")
            if _samples is not None:
                _samples.close()
                os.replace(path, path + ".1")
            _samples = open(path, "a", buffering=1)
        _samples.write(json.dumps(d, sort_keys=True) + "\n")
    except Exception:
        pass


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            out[k] = int(v.split()[0])
    return out


def psi_full_avg10():
    try:
        for line in open("/proc/pressure/memory"):
            if line.startswith("full"):
                return float(re.search(r"avg10=([\d.]+)", line).group(1))
    except Exception:
        pass
    return 0.0


def dstate_count():
    n = 0
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                with open(f"/proc/{pid}/stat") as f:
                    if f.read().rsplit(")", 1)[1].split()[0] == "D":
                        n += 1
            except Exception:
                pass
    return n


def cgroup_of(pid):
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            for line in f:
                if line.startswith("0::"):
                    return line[3:].strip()
    except Exception:
        return None


def start_time(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except Exception:
        return 0


def comm(pid):
    try:
        return open(f"/proc/{pid}/comm").read().strip()
    except Exception:
        return "?"


def gpu_tenants():
    """cgroup -> {pids, oldest start, comms} for processes holding /dev/nvidia*."""
    tenants = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        fddir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fddir)
        except Exception:
            continue
        hit = False
        for fd in fds:
            try:
                if os.readlink(f"{fddir}/{fd}").startswith("/dev/nvidia"):
                    hit = True
                    break
            except Exception:
                pass
        if not hit:
            continue
        cg = cgroup_of(pid)
        if not cg:
            continue
        c = comm(pid)
        if re.search(CONF["PROTECT_PATTERNS"], c):
            continue
        t = tenants.setdefault(cg, {"pids": [], "start": 1 << 62, "comms": set()})
        t["pids"].append(int(pid))
        t["start"] = min(t["start"], start_time(pid))
        t["comms"].add(c)
    return tenants


def killable(cg):
    return ("/docker-" in cg or "/user.slice/" in cg or "/session-" in cg)


def kill_tenant(cg, t, reason):
    if not killable(cg):
        log("skip_unkillable", cgroup=cg, reason=reason, comms=sorted(t["comms"]))
        return False
    if CONF["DRY_RUN"] == "1":
        log("dry_run_kill", cgroup=cg, reason=reason, pids=t["pids"], comms=sorted(t["comms"]))
        return True
    killed = False
    # Whole-container kill when the cgroup is a docker scope.
    if "/docker-" in cg:
        try:
            with open(f"{CG}{cg}/cgroup.kill", "w") as f:
                f.write("1")
            killed = True
        except Exception as e:
            log("cgroup_kill_failed", cgroup=cg, err=str(e))
    if not killed:
        for pid in t["pids"]:
            try:
                os.kill(pid, 9)
                killed = True
            except Exception:
                pass
    log("killed", cgroup=cg, reason=reason, pids=t["pids"], comms=sorted(t["comms"]))
    return killed


def main():
    load_conf()
    os.makedirs(LOGDIR, exist_ok=True)
    lock_memory()
    try:
        os.nice(-15)
    except Exception:
        pass
    kill_kib = float(CONF["KILL_GIB"]) * 1048576
    hard_kib = float(CONF["HARD_GIB"]) * 1048576
    warn_kib = float(CONF["WARN_GIB"]) * 1048576
    need = int(CONF["KILL_SAMPLES"])
    psi_kill = float(CONF["PSI_FULL_KILL"])
    interval = float(CONF["INTERVAL"])
    scan_every = int(CONF["TENANT_SCAN_EVERY"])
    sample_every = int(CONF["SAMPLE_LOG_EVERY"])
    log("start", conf={k: CONF[k] for k in sorted(CONF)})
    bad = 0
    psi_since = None
    tick = 0
    tenants = {}
    warned = False
    while True:
        tick += 1
        mi = meminfo()
        avail = mi.get("MemAvailable", 0)
        psi = psi_full_avg10()
        if tick % scan_every == 0 or bad:
            tenants = gpu_tenants()
        if tick % sample_every == 0:
            sample_log({"ts": int(time.time()), "avail_gib": round(avail / 1048576, 2),
                        "psi_full_avg10": psi, "dstate": dstate_count(), "swap_free_gib": round(mi.get("SwapFree", 0) / 1048576, 2),
                        "tenants": {cg: sorted(t["comms"]) for cg, t in tenants.items()}})
        # Rule 2: single GPU tenant. Only long-lived tenants count (ignores nvidia-smi / exporters).
        now_ticks = _uptime_ticks()
        lasting = {cg: t for cg, t in tenants.items()
                   if now_ticks - t["start"] >= float(CONF["TENANT_MIN_AGE_S"]) * _HZ}
        if CONF["ENFORCE_SINGLE_TENANT"] == "1" and len(lasting) > 1:
            order = sorted(lasting.items(), key=lambda kv: kv[1]["start"])
            for cg, t in order[1:]:
                kill_tenant(cg, t, f"multi_tenant({len(lasting)})")
            tenants = {order[0][0]: order[0][1]}
        # Rule 1: memory floor.
        if avail < warn_kib and not warned:
            log("warn_low_mem", avail_gib=round(avail / 1048576, 2), psi_full_avg10=psi)
            warned = True
        elif avail > warn_kib * 1.2:
            warned = False
        # PSI alone must stay above the line for PSI_SUSTAIN_S seconds (model loads spike briefly).
        psi_since = (psi_since or time.monotonic()) if psi > psi_kill else None
        psi_trip = psi_since is not None and time.monotonic() - psi_since >= float(CONF["PSI_SUSTAIN_S"])
        if avail < kill_kib or psi_trip:
            reason = f"mem avail={avail/1048576:.2f}GiB psi_full_avg10={psi}" + (" (psi sustained)" if psi_trip else "")
            tenants = gpu_tenants()
            victims = sorted(tenants.items(), key=lambda kv: kv[1]["start"], reverse=True)
            if not victims:
                log("trip_no_tenant", reason=reason)
            for cg, t in victims:
                kill_tenant(cg, t, reason)
                time.sleep(2)
                if meminfo().get("MemAvailable", 0) > kill_kib * 2:
                    break
            psi_since = None
        # Poll 4x faster whenever memory is below the warning line (GB10 can lose ~7 GiB/s).
        time.sleep(interval / 4 if avail < warn_kib * 2 else interval)


if __name__ == "__main__":
    main()
