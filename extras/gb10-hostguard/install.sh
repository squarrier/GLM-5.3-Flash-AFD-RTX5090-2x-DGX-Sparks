#!/usr/bin/env bash
# Install gb10-hostguard + host hardening. Usage: sudo ./install.sh gb10|rtx
#   gb10: DGX Spark (unified memory; floors 6/4 GiB)   rtx: discrete-GPU host (floors 10/5 GiB)
# Idempotent. Writes only gb10-hostguard files, sshd/journald OOM drop-ins, one sysctl file and a
# systemd watchdog drop-in. Read the README section before installing: it enables kernel.panic on
# soft lockup and the hardware watchdog, so a hung box reboots itself instead of waiting for you.
set -euo pipefail
ROLE="${1:?role gb10|rtx}"
HERE="$(cd "$(dirname "$0")" && pwd)"

install -m 755 "$HERE/gb10-hostguard.py" /usr/local/sbin/gb10-hostguard

case "$ROLE" in
  gb10) KILL=6.0; HARD=4.0; WARN=12.0 ;;
  rtx)  KILL=10.0; HARD=5.0; WARN=24.0 ;;
  *) echo "unknown role $ROLE" >&2; exit 2 ;;
esac
cat > /etc/default/gb10-hostguard <<EOF
# gb10-hostguard thresholds ($ROLE). GiB of MemAvailable.
KILL_GIB=$KILL
HARD_GIB=$HARD
WARN_GIB=$WARN
KILL_SAMPLES=3
PSI_FULL_KILL=25
ENFORCE_SINGLE_TENANT=1
DRY_RUN=0
EOF

cat > /etc/systemd/system/gb10-hostguard.service <<'EOF'
[Unit]
Description=GPU host guard (memory floor + single GPU tenant)
After=local-fs.target
StartLimitIntervalSec=0

[Service]
ExecStart=/usr/bin/python3 -u /usr/local/sbin/gb10-hostguard
Restart=always
RestartSec=2
OOMScoreAdjust=-1000
Nice=-15
MemoryMin=64M
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF

# Keep sshd and journald alive and reachable when memory is tight.
for u in ssh.service systemd-journald.service; do
  mkdir -p "/etc/systemd/system/$u.d"
  cat > "/etc/systemd/system/$u.d/gb10-hostguard-protect.conf" <<'EOF'
[Service]
OOMScoreAdjust=-1000
MemoryMin=128M
EOF
done

# Kernel: bigger free reserve and earlier kswapd so reclaim never has to stall
# userspace; reboot on kernel lockup instead of hanging forever.
cat > /etc/sysctl.d/90-gb10-hostguard.conf <<'EOF'
vm.min_free_kbytes = 1048576
vm.watermark_scale_factor = 200
kernel.panic = 30
kernel.panic_on_oops = 1
kernel.softlockup_panic = 1
EOF
if [ "$ROLE" = gb10 ]; then
  echo "vm.compaction_proactiveness = 0" >> /etc/sysctl.d/90-gb10-hostguard.conf
fi
sysctl -q --system

# Hardware (GB10: SBSA) or software (x86 without one: softdog) watchdog driven by systemd.
if [ ! -e /dev/watchdog ]; then
  echo softdog > /etc/modules-load.d/gb10-hostguard-softdog.conf
  modprobe softdog || true
fi
mkdir -p /etc/systemd/system.conf.d
cat > /etc/systemd/system.conf.d/gb10-hostguard-watchdog.conf <<'EOF'
[Manager]
RuntimeWatchdogSec=60
RebootWatchdogSec=5min
EOF

# Persistent journal so a hang leaves evidence across the reboot.
mkdir -p /var/log/journal /var/log/gb10-hostguard
systemd-tmpfiles --create --prefix /var/log/journal >/dev/null 2>&1 || true
journalctl --flush >/dev/null 2>&1 || true

systemctl daemon-reload
systemctl daemon-reexec
systemctl restart systemd-journald
systemctl try-restart ssh.service || true
systemctl enable --now gb10-hostguard.service
systemctl restart gb10-hostguard.service
sleep 2

echo "--- verify"
systemctl is-active gb10-hostguard.service
systemctl show -p OOMScoreAdjust ssh.service systemd-journald.service gb10-hostguard.service | tr '\n' ' '; echo
systemctl show -p RuntimeWatchdogUSec; ls -l /dev/watchdog* 2>&1 | head -2
sysctl vm.min_free_kbytes vm.watermark_scale_factor kernel.panic kernel.softlockup_panic
ls -d /var/log/journal && journalctl --disk-usage
tail -1 /var/log/gb10-hostguard/events.log
echo INSTALL_OK
