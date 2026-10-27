"""
N-BaIoT dataset audit: duplication, overlap across classes and across devices.

Usage:
    cd ~/work/IoT_Botnet/veri
    python3 nbaiot_denetim.py

Output:
    CSV tables and ozet.json in the nbaiot_denetim/ folder
"""

import os
import glob
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd

KOK = "nbaiot"
CIKTI = "nbaiot_denetim"
os.makedirs(CIKTI, exist_ok=True)


def bilgi(yol):
    p = os.path.relpath(yol, KOK).split(os.sep)
    if len(p) == 2:
        return p[0], "benign", "benign"
    return p[0], p[1].replace("_attacks", ""), os.path.splitext(p[-1])[0]


def satir_ozetleri(df):
    """64-bit digest (hash) for each row. Equal rows give equal digests."""
    return pd.util.hash_pandas_object(df, index=False).to_numpy()


def main():
    dosyalar = sorted(glob.glob(f"{KOK}/**/*.csv", recursive=True))
    dosyalar = [f for f in dosyalar if "demonstrate" not in f]
    print(f"{len(dosyalar)} files being audited\n")

    t0 = time.time()
    ozetler = {}          # (device, family, type) -> set of unique hashes
    satirlar = []

    for f in dosyalar:
        cihaz, aile, tur = bilgi(f)
        df = pd.read_csv(f, dtype="float32")
        h = satir_ozetleri(df)
        tekil, adet = np.unique(h, return_counts=True)
        ozetler[(cihaz, aile, tur)] = set(tekil.tolist())
        satirlar.append({
            "cihaz": cihaz,
            "aile": aile,
            "tur": tur,
            "satir": len(df),
            "tekil": len(tekil),
            "tekillik_%": round(100 * len(tekil) / len(df), 3),
            "en_sik_tekrar": int(adet.max()),
            "en_sik_pay_%": round(100 * adet.max() / len(df), 3),
        })
        print(f"  {cihaz[:30]:<32}{aile:<8}{tur:<10}"
              f"{len(df):>9,} rows  {len(tekil):>8,} unique  "
              f"(%{100*len(tekil)/len(df):.2f})")

    print(f"\nReading time: {time.time()-t0:.0f} s")

    # ---------------------------------------------------------------
    # 1. Per-file duplication
    # ---------------------------------------------------------------
    tablo1 = pd.DataFrame(satirlar)
    tablo1.to_csv(f"{CIKTI}/tablo1_dosya_yinelenme.csv", index=False)

    print("\n" + "=" * 72)
    print("  1. MEAN UNIQUENESS PER ATTACK TYPE")
    print("=" * 72)
    oz = (tablo1.groupby(["aile", "tur"])
          .agg(cihaz=("cihaz", "nunique"),
               toplam_satir=("satir", "sum"),
               toplam_tekil=("tekil", "sum"),
               ort_tekillik=("tekillik_%", "mean"))
          .reset_index()
          .sort_values("ort_tekillik"))
    oz["ort_tekillik"] = oz.ort_tekillik.round(3)
    print(oz.to_string(index=False))
    oz.to_csv(f"{CIKTI}/tablo2_tur_ozeti.csv", index=False)

    sorunlu = oz[oz.ort_tekillik < 10]
    print(f"\nNumber of types with uniqueness below 10%: {len(sorunlu)}")
    for _, r in sorunlu.iterrows():
        print(f"  {r.aile}:{r.tur}  -> {r.toplam_satir:,} rows, "
              f"only {r.toplam_tekil:,} unique")

    # ---------------------------------------------------------------
    # 2. Overlap across types on the same device
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  2. OVERLAP ACROSS TYPES ON THE SAME DEVICE")
    print("=" * 72)
    cakisma = []
    cihazlar = sorted({k[0] for k in ozetler})
    for c in cihazlar:
        anahtarlar = [k for k in ozetler if k[0] == c]
        for i, a in enumerate(anahtarlar):
            for b in anahtarlar[i + 1:]:
                A, B = ozetler[a], ozetler[b]
                ortak = len(A & B)
                if ortak == 0:
                    continue
                cakisma.append({
                    "cihaz": c,
                    "tur_a": f"{a[1]}:{a[2]}",
                    "tur_b": f"{b[1]}:{b[2]}",
                    "ortak_tekil": ortak,
                    "a_tekil": len(A),
                    "b_tekil": len(B),
                    "a_payi_%": round(100 * ortak / len(A), 2),
                    "b_payi_%": round(100 * ortak / len(B), 2),
                })
    t3 = pd.DataFrame(cakisma)
    t3.to_csv(f"{CIKTI}/tablo3_turler_arasi_cakisma.csv", index=False)
    if t3.empty:
        print("  No overlap.")
    else:
        print(t3.sort_values("ortak_tekil", ascending=False)
              .head(25).to_string(index=False))

    # ---------------------------------------------------------------
    # 3. Are there exactly identical rows between benign and attack?
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  3. BENIGN <-> ATTACK EXACT OVERLAP  (most serious case)")
    print("=" * 72)
    ib = []
    for c in cihazlar:
        B = ozetler.get((c, "benign", "benign"), set())
        for k, S in ozetler.items():
            if k[0] != c or k[1] == "benign":
                continue
            ortak = len(B & S)
            ib.append({"cihaz": c, "tur": f"{k[1]}:{k[2]}",
                       "ortak_tekil": ortak,
                       "saldiri_tekil": len(S),
                       "saldiri_payi_%": round(100 * ortak / max(len(S), 1), 3)})
    t4 = pd.DataFrame(ib)
    t4.to_csv(f"{CIKTI}/tablo4_iyi_saldiri_cakisma.csv", index=False)
    bulunan = t4[t4.ortak_tekil > 0]
    if bulunan.empty:
        print("  There is NO exact overlap between benign and attack rows.")
    else:
        print(f"  Overlap found in {len(bulunan)} device-type pairs:\n")
        print(bulunan.sort_values("ortak_tekil", ascending=False)
              .to_string(index=False))

    # ---------------------------------------------------------------
    # 4. Overlap across devices (same attack type)
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  4. OVERLAP ACROSS DEVICES (same attack type)")
    print("=" * 72)
    ca = []
    turler = sorted({(k[1], k[2]) for k in ozetler if k[1] != "benign"})
    for aile, tur in turler:
        sahip = [c for c in cihazlar if (c, aile, tur) in ozetler]
        for i, a in enumerate(sahip):
            for b in sahip[i + 1:]:
                A = ozetler[(a, aile, tur)]
                B = ozetler[(b, aile, tur)]
                ortak = len(A & B)
                if ortak:
                    ca.append({"tur": f"{aile}:{tur}", "cihaz_a": a,
                               "cihaz_b": b, "ortak_tekil": ortak})
    t5 = pd.DataFrame(ca)
    t5.to_csv(f"{CIKTI}/tablo5_cihazlar_arasi_cakisma.csv", index=False)
    if t5.empty:
        print("  No overlap across devices.")
    else:
        print(t5.groupby("tur").ortak_tekil.agg(["count", "sum"])
              .to_string())

    # ---------------------------------------------------------------
    # 5. Effective sample size
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  5. EFFECTIVE SAMPLE SIZE")
    print("=" * 72)
    sal = tablo1[tablo1.aile != "benign"]
    iyi = tablo1[tablo1.aile == "benign"]
    print(f"  Attack  : {sal.satir.sum():>12,} rows    ->   "
          f"{sal.tekil.sum():>10,} unique  "
          f"(%{100*sal.tekil.sum()/sal.satir.sum():.2f})")
    print(f"  Benign  : {iyi.satir.sum():>12,} rows    ->   "
          f"{iyi.tekil.sum():>10,} unique  "
          f"(%{100*iyi.tekil.sum()/iyi.satir.sum():.2f})")

    ozet = {
        "dosya_sayisi": len(dosyalar),
        "toplam_satir": int(tablo1.satir.sum()),
        "toplam_tekil": int(tablo1.tekil.sum()),
        "saldiri_tekillik_%": round(100 * sal.tekil.sum() / sal.satir.sum(), 3),
        "iyi_huylu_tekillik_%": round(100 * iyi.tekil.sum() / iyi.satir.sum(), 3),
        "tekillik_10_alti_turler": [f"{r.aile}:{r.tur}" for _, r in sorunlu.iterrows()],
        "turler_arasi_cakisma_cifti": int(len(t3)),
        "iyi_saldiri_cakisma_cifti": int(len(bulunan)),
        "cihazlar_arasi_cakisma_cifti": int(len(t5)),
    }
    with open(f"{CIKTI}/ozet.json", "w") as fh:
        json.dump(ozet, fh, indent=2, ensure_ascii=False)
    print("\n" + json.dumps(ozet, indent=2, ensure_ascii=False))
    print(f"\nAll outputs: {CIKTI}/")


if __name__ == "__main__":
    main()
