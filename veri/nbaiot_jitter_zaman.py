"""N-BaIoT: HH_jit_*_mean özniteliklerindeki zaman damgası artefaktı.

Bazı satırlarda HH_jit_*_mean değeri ~1.5e9, yani Unix zaman damgası (saniye).
Her (cihaz, saldırı türü) dosyası için şunları raporlar:
  - zd_pay_%       : HH_jit_L5_mean > 1e9 olan satırların payı
  - baskin_zd      : en sık değer ve UTC karşılığı
  - baskin_pay_%   : en sık satır vektörünün dosyadaki payı (float64, 115 öznitelik)

Ek olarak gafgyt tcp/udp baskın vektörlerinin farkını yazar.
Çıktı: veri/nbaiot_denetim/tablo6_jitter_zaman.csv, tablo7_tcp_udp_baskin_fark.csv
Eğitim yok; yalnızca CSV okuma.
"""
import datetime as dt
import pathlib

import numpy as np
import pandas as pd

KOK = pathlib.Path(__file__).resolve().parent
VERI, CIKTI = KOK / "nbaiot", KOK / "nbaiot_denetim"
C = "HH_jit_L5_mean"


def utc(x):
    return dt.datetime.fromtimestamp(x, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if x > 1e9 else ""


def main():
    satirlar, farklar = [], []
    for cihaz_dir in sorted(p for p in VERI.iterdir() if p.is_dir()):
        dosyalar = [("benign", "benign", cihaz_dir / "benign_traffic.csv")]
        for aile in ("gafgyt", "mirai"):
            dosyalar += [(aile, f.stem, f) for f in sorted((cihaz_dir / f"{aile}_attacks").rglob("*.csv"))]
        baskin = {}
        for aile, tur, f in dosyalar:
            if not f.exists():
                continue
            df = pd.read_csv(f)
            s = df[C]
            vc = df.value_counts()
            baskin[(aile, tur)] = np.array(vc.index[0], float)
            mod = s.mode()[0]
            satirlar.append({
                "cihaz": cihaz_dir.name, "aile": aile, "tur": tur, "satir": len(df),
                "zd_pay_%": round(100 * (s > 1e9).mean(), 2),
                "baskin_zd": round(mod, 2), "baskin_zd_utc": utc(mod),
                "baskin_pay_%": round(100 * vc.iloc[0] / len(df), 3),
            })
        if ("gafgyt", "tcp") in baskin and ("gafgyt", "udp") in baskin:
            a, b = baskin[("gafgyt", "tcp")], baskin[("gafgyt", "udp")]
            i = np.flatnonzero(a != b)
            farklar.append({
                "cihaz": cihaz_dir.name, "farkli_oznitelik": len(i),
                "oznitelikler": ";".join(df.columns[i]),
                "max_goreli_fark": float((np.abs(a - b) / np.maximum(np.abs(a), np.abs(b)).clip(1e-12)).max()),
                "tcp_zd_utc": utc(a[df.columns.get_loc(C)]), "udp_zd_utc": utc(b[df.columns.get_loc(C)]),
            })
    pd.DataFrame(satirlar).to_csv(CIKTI / "tablo6_jitter_zaman.csv", index=False)
    pd.DataFrame(farklar).to_csv(CIKTI / "tablo7_tcp_udp_baskin_fark.csv", index=False)
    print(pd.DataFrame(farklar).to_string(index=False))


if __name__ == "__main__":
    main()
