"""
IoT-23 inventory extraction.

Usage:
    cd ~/work/IoT_Botnet/veri
    python3 iot23_envanter.py

Scans the first ORNEK rows of each scenario and answers these questions:
  - How many distinct source IPs are there? (is the device identity meaningful)
  - How is the benign / malicious balance?
  - Is there any IP on which a clean baseline can be built?
"""

import os
import glob
import json
import collections

ORNEK = 300_000          # rows to scan per scenario
MIN_BENIGN = 1000        # minimum benign flows needed for a clean baseline
MIN_ORAN = 0.5           # and the benign ratio must exceed this threshold


def parcala(satir, n_alan):
    """Splits a Zeek line into fields.

    The last three fields (tunnel_parents, label, detailed-label) are
    separated by spaces instead of tabs in some versions.
    """
    p = satir.rstrip("\n").split("\t")
    if len(p) >= n_alan:
        return p
    return p[:-1] + p[-1].split()


def senaryo_tara(yol):
    alanlar = None
    n = 0
    ipler = collections.Counter()
    etiket = collections.Counter()
    detay = collections.Counter()
    ip_etiket = collections.defaultdict(collections.Counter)
    hatali = 0

    with open(yol, errors="replace") as fh:
        for satir in fh:
            if satir.startswith("#"):
                if satir.startswith("#fields"):
                    ham = satir.strip().split("\t")[1:]
                    alanlar = []
                    for a in ham:
                        alanlar.extend(a.split())
                continue
            if alanlar is None:
                continue
            if n >= ORNEK:
                break
            n += 1
            p = parcala(satir, len(alanlar))
            try:
                ip = p[alanlar.index("id.orig_h")]
                lb = p[alanlar.index("label")]
                dl = p[alanlar.index("detailed-label")]
            except (ValueError, IndexError):
                hatali += 1
                continue
            ipler[ip] += 1
            etiket[lb] += 1
            detay[dl] += 1
            ip_etiket[ip][lb] += 1

    temiz = []
    for ip, c in ip_etiket.items():
        b = c.get("Benign", 0)
        if b >= MIN_BENIGN and b / max(sum(c.values()), 1) > MIN_ORAN:
            temiz.append((ip, b, sum(c.values())))
    temiz.sort(key=lambda x: -x[1])

    return {
        "satir": n,
        "hatali": hatali,
        "ip_sayisi": len(ipler),
        "etiket": dict(etiket),
        "detay": dict(detay),
        "temiz_ip": temiz,
        "en_yogun_ip": ipler.most_common(3),
        "alanlar": alanlar,
    }


def main():
    dosyalar = sorted(glob.glob("iot23/**/conn.log.labeled", recursive=True))
    if not dosyalar:
        raise SystemExit("conn.log.labeled bulunamadi. Dogru klasorde misiniz?")

    print(f"{len(dosyalar)} scenarios being scanned "
          f"(at most {ORNEK:,} rows per scenario)\n")

    basliklar = ("Senaryo", "Satir", "IP", "Benign", "Malicious", "Temiz")
    print("{:<34}{:>10}{:>7}{:>11}{:>11}{:>7}".format(*basliklar))
    print("-" * 80)

    genel_detay = collections.Counter()
    genel_etiket = collections.Counter()
    ozet = {}
    toplam_hatali = 0

    for f in dosyalar:
        ad = f.split(os.sep)[-3]
        r = senaryo_tara(f)
        ozet[ad] = r
        toplam_hatali += r["hatali"]
        genel_detay.update(r["detay"])
        genel_etiket.update(r["etiket"])

        print("{:<34}{:>10,}{:>7}{:>11,}{:>11,}{:>7}".format(
            ad[:34],
            r["satir"],
            r["ip_sayisi"],
            r["etiket"].get("Benign", 0),
            r["etiket"].get("Malicious", 0),
            len(r["temiz_ip"]),
        ))

    print(f"\nUnparseable rows: {toplam_hatali:,}")

    print("\nOverall label distribution:")
    for k, v in genel_etiket.most_common():
        print(f"  {v:>12,}  {k}")

    print("\nDetailed label distribution:")
    for k, v in genel_detay.most_common(20):
        print(f"  {v:>12,}  {k}")

    print("\n" + "=" * 70)
    print("  IPs ON WHICH A CLEAN BASELINE CAN BE BUILT")
    print(f"  (>= {MIN_BENIGN} benign flows and benign ratio > %{int(MIN_ORAN*100)})")
    print("=" * 70)
    bulundu = False
    for ad, r in ozet.items():
        if r["temiz_ip"]:
            bulundu = True
            print(f"\n{ad}")
            for ip, b, t in r["temiz_ip"][:5]:
                print(f"    {ip:<18} benign={b:>8,}  toplam={t:>8,}  "
                      f"(%{100*b/t:.1f})")
    if not bulundu:
        print("\n  NONE.")
        print("  No per-device clean baseline can be built in IoT-23.")
        print("  -> The personalization analysis (RQ1-RQ4) cannot be")
        print("     replicated on this dataset; it falls to a supporting role.")

    # show the field list of the first scenario
    ilk = next(iter(ozet.values()))
    print(f"\nZeek fields ({len(ilk['alanlar'])}):")
    print("  " + ", ".join(ilk["alanlar"]))

    cikti = {
        ad: {k: v for k, v in r.items() if k != "alanlar"}
        for ad, r in ozet.items()
    }
    with open("iot23_envanter.json", "w") as fh:
        json.dump(cikti, fh, indent=2, ensure_ascii=False, default=str)
    print("\nDetailed output: iot23_envanter.json")


if __name__ == "__main__":
    main()
