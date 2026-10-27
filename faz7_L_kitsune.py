"""Phase 7 L - source of the timestamp artifact: the Kitsune/AfterImage extractor (claim 10, open question init_time).

1) Do the HH_jit_*_mean columns in the published Kitsune Network Attack Dataset (UCI 516; Mirsky et al. 2018) feature
   files contain ~1.5e9 (absolute time) values, and in which rows (the first packet of a new flow?)
2) From the first N packets of the same pcap, extract features with the published Kitsune-py (init_time = t) and with a
   one-line change (init_time = 0); which one matches the published features?
Usage: python3 faz7_L_kitsune.py [N]
"""
import glob
import gzip
import sys
import time

from faz7_ortak import *  # noqa: F401,F403
from faz7_ortak import log
import numpy as np

KIT = os.path.expanduser("~/work/IoT_Botnet/kaynak_kod/Kitsune-py")
VERI = os.path.expanduser("~/work/IoT_Botnet/veri/kitsune/ac")
sys.path.insert(0, KIT)
np.Inf = np.inf   # Kitsune-py uses NumPy 1 syntax (without modifying the submodule)


def cikarici(init_sifir):
    import AfterImage as af
    import netStat as ns
    if init_sifir:
        def update(self, ID, t, v, Lambda=1, isTypeDiff=False):
            incS = self.register(ID, Lambda, 0, isTypeDiff)   # the only difference: init_time = 0
            incS.insert(v, t)
            return incS
    else:
        update = af.incStatDB.update.__wrapped__ if hasattr(af.incStatDB.update, "__wrapped__") else None
    st = ns.netStat(np.nan, 100000000, 100000000)
    if init_sifir:
        import types
        for db in (st.HT_jit, st.HT_MI, st.HT_H, st.HT_Hp):
            db.update = types.MethodType(update, db)
    return st


def paketler(pcap, N):
    from scapy.all import ARP, ICMP, IP, IPv6, TCP, UDP, PcapReader
    acik = gzip.open(pcap, "rb") if pcap.endswith(".gz") else open(pcap, "rb")
    for i, packet in enumerate(PcapReader(acik)):
        if i >= N:
            break
        IPtype = np.nan
        if packet.haslayer(IP):
            srcIP, dstIP, IPtype = packet[IP].src, packet[IP].dst, 0
        elif packet.haslayer(IPv6):
            srcIP, dstIP, IPtype = packet[IPv6].src, packet[IPv6].dst, 1
        else:
            srcIP = dstIP = ""
        if packet.haslayer(TCP):
            sp, dp = str(packet[TCP].sport), str(packet[TCP].dport)
        elif packet.haslayer(UDP):
            sp, dp = str(packet[UDP].sport), str(packet[UDP].dport)
        else:
            sp = dp = ""
        sm, dm = packet.src, packet.dst
        if sp == "":
            if packet.haslayer(ARP):
                sp = dp = "arp"; srcIP, dstIP, IPtype = packet[ARP].psrc, packet[ARP].pdst, 0
            elif packet.haslayer(ICMP):
                sp = dp = "icmp"; IPtype = 0
            elif srcIP + sp + dstIP + dp == "":
                srcIP, dstIP = packet.src, packet.dst
        yield IPtype, sm, dm, srcIP, sp, dstIP, dp, len(packet), float(packet.time)


def eslestir(kp_basliklar, ozn):
    """Kitsune-py header -> published (N-BaIoT layout, 115) column index and transform (1D 'std' is 'variance' in the published data)."""
    import re
    out = []
    for j, h in enumerate(kp_basliklar):
        m = re.match(r"(MI_dir|HH_jit|HpHp|HH)_([\d.]+)_([a-z]+)", h)
        grup, lam, stat = m.groups()
        kare = stat == "std" and grup in ("MI_dir", "HH_jit")
        ad = f"{grup}_L{lam}_{'variance' if kare else stat}"
        if ad in ozn:
            out.append((j, ozn.index(ad), kare))
    return out


def main(N):
    import netStat as ns
    from faz7_ortak import yukle
    _, _, ozn = yukle()                      # published Kitsune files have the same 115-column layout as N-BaIoT
    kp = ns.netStat().getNetStatHeaders()
    es = eslestir(kp, ozn)
    jm = [ozn.index(c) for c in ozn if c.startswith("HH_jit_") and c.endswith("_mean")]
    jv = [ozn.index(c) for c in ozn if c.startswith("HH_jit_") and c.endswith("_variance")]
    log("matched columns:", len(es), "/", len(kp))
    rows = []
    for d in sorted(glob.glob(f"{VERI}/*/")):
        ad = os.path.basename(d.rstrip("/"))
        fx = sorted(glob.glob(f"{d}*dataset.csv*")); fp = sorted(glob.glob(f"{d}*_pcap.pcap*"))
        if not fx:
            continue
        Xp = pd.read_csv(fx[0], header=None, nrows=max(N, 1_000_000)).to_numpy(np.float64)
        if Xp.shape[1] == 116:               # in the Mirai file, a leading row-number column
            Xp = Xp[:, 1:]
        r = {"saldiri": ad, "satir_okunan": len(Xp), "sutun": Xp.shape[1],
             "jit_mean_zd_satir_%": 100 * float((Xp[:, jm] > 1e9).any(1).mean()),
             "jit_var_buyuk_satir_%": 100 * float((Xp[:, jv] > 1e15).any(1).mean()),
             "baska_sutunda_zd": int((Xp[:, [j for j in range(115) if j not in jm + jv]] > 1e9).any(1).sum()),
             "ilk_paket_w1_ve_zd_%": 100 * float(((Xp[:, jm[0] - 1] < 1.5) & (Xp[:, jm[0]] > 1e9)).mean()),
             "zd_ilk_utc": pd.to_datetime(Xp[:, jm[0]][Xp[:, jm[0]] > 1e9].min(), unit="s").isoformat() if (Xp[:, jm[0]] > 1e9).any() else ""}
        if fp:
            for sifir in (False, True):
                t0 = time.time()
                st = cikarici(sifir)
                Xk = np.array([st.updateGetStats(*p) for p in paketler(fp[0], N)])
                n = min(len(Xk), len(Xp))
                k = "init0" if sifir else "initt"
                A = np.stack([Xk[:n, a] ** 2 if kare else Xk[:n, a] for a, _, kare in es], 1)
                B = Xp[:n, [b for _, b, _ in es]]
                yakin = np.isclose(A, B, rtol=1e-5, atol=1e-6)
                jcols = [i for i, (_, b, _) in enumerate(es) if b in jm]
                r[f"{k}_paket"] = len(Xk)
                r[f"{k}_tum_sutun_uyum_satir_%"] = 100 * float(yakin.all(1).mean())
                r[f"{k}_hucre_uyum_%"] = 100 * float(yakin.mean())
                r[f"{k}_jit_mean_uyum_satir_%"] = 100 * float(yakin[:, jcols].all(1).mean())
                r[f"{k}_jit_mean_zd_satir_%"] = 100 * float((B[:, jcols] > 1e9).any(1).mean()) if sifir else \
                    100 * float((A[:, jcols] > 1e9).any(1).mean())
                r[f"{k}_sure_s"] = round(time.time() - t0, 1)
                log(ad, k, {kk: vv for kk, vv in r.items() if kk.startswith(k)})
        rows.append(r); log(r)
        pd.DataFrame(rows).to_csv(f"{TAB}/L_kitsune.csv", index=False)
    log("L done")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200_000)
