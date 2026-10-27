"""Phase 7 memory watchdog: if available RAM drops below the threshold, kills the faz7 process holding the most memory.
The killed job is checkpointed, so it resumes where it left off in the runner's second pass.
Writes a memory/GPU summary every 5 minutes. Usage: python3 faz7_bekci.py  (in the background)"""
import os
import signal
import subprocess
import time

ESIK_GB = float(os.environ.get("BEKCI_ESIK_GB", "12"))
LOG = os.path.expanduser("~/work/IoT_Botnet/faz7_ciktilar/logs/bekci.log")


def kullanilabilir_gb():
    """Remaining memory according to the container cgroup limit (anon + kernel; excluding reclaimable page cache).
    Even though the machine has 122 GB, the container limit is lower (memory.max ~ 92 GB); /proc/meminfo does not show this."""
    try:
        sinir = int(open("/sys/fs/cgroup/memory.max").read())
        st = dict(l.split() for l in open("/sys/fs/cgroup/memory.stat"))
        kullanilan = int(st["anon"]) + int(st.get("kernel", 0)) + int(st.get("shmem", 0))
        return (sinir - kullanilan) / 1e9
    except (OSError, ValueError, KeyError):
        for s in open("/proc/meminfo"):
            if s.startswith("MemAvailable"):
                return int(s.split()[1]) / 1e6


def faz7_surecleri():
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
            if "python" in cmd and ("faz7_" in cmd or "faz8_" in cmd) and "bekci" not in cmd:
                rss = int(open(f"/proc/{pid}/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
                out.append((rss, int(pid), cmd.strip()))
        except OSError:
            pass
    return sorted(out, reverse=True)


def yaz(s):
    with open(LOG, "a") as fh:
        fh.write(time.strftime("%H:%M:%S ") + s + "\n")


son = 0
while True:
    gb = kullanilabilir_gb()
    if gb < ESIK_GB:
        ss = faz7_surecleri()
        if ss:
            rss, pid, cmd = ss[0]
            yaz(f"UYARI kullanılabilir {gb:.1f} GB < {ESIK_GB}; sonlandırılıyor pid={pid} rss={rss:.1f} GB: {cmd}")
            try:
                os.kill(pid, signal.SIGTERM); time.sleep(10); os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    if time.time() - son > 300:
        son = time.time()
        try:
            g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
                               capture_output=True, text=True).stdout.strip()
        except OSError:
            g = "?"
        ss = faz7_surecleri()
        yaz(f"RAM boş {gb:.1f} GB | GPU {g} | süreç {len(ss)} | en büyük " +
            "; ".join(f"{r:.1f}GB {c.split()[1] if len(c.split()) > 1 else c}" for r, _, c in ss[:3]))
    time.sleep(10)
