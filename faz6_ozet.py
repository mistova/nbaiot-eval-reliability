"""Collects the Faz 6 results into a single markdown report: faz6_ciktilar/FAZ6_OZET.md"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/work/pylibs"))

import numpy as np
import pandas as pd

OUT = os.path.expanduser("~/work/IoT_Botnet/faz6_ciktilar")
TAB = f"{OUT}/tables"
parca = []


def oku(ad):
    yol = f"{TAB}/{ad}"
    return pd.read_csv(yol) if os.path.exists(yol) else None


def bas(baslik, df, ondalik=4):
    parca.append(f"\n## {baslik}\n")
    parca.append("_(tablo yok)_\n" if df is None else df.round(ondalik).to_markdown() + "\n")


def ms(g, cols):
    """mean ± std text"""
    m, s = g[cols].mean(), g[cols].std()
    return pd.DataFrame({c: [f"{m[c]:.4f} ± {s[c]:.4f}" if pd.notna(s[c]) else f"{m[c]:.4f}"] for c in cols})


# A
a1 = oku("A1_dosya_tekillik.csv")
if a1 is not None:
    t = a1[a1.aile != "benign"].groupby(["aile", "tur"])[["satir", "tekil_float64", "tekil_float32", "tekil_float64_zdsiz"]].sum()
    for p in ("float64", "float32", "float64_zdsiz"):
        t[f"tekillik_{p}_%"] = 100 * t[f"tekil_{p}"] / t.satir
    bas("A1 — Saldırı türü tekilliği (üç hash tanımı)", t, 3)
a5 = oku("A5_sizinti.csv")
if a5 is not None:
    bas("A5 — Bölme sızıntısı (%)", a5.groupby(["hassasiyet", "bolme"])[[c for c in a5.columns if c.startswith("sizinti")]].mean(), 2)
a3 = oku("A3_cihazlar_arasi.csv")
if a3 is not None:
    bas("A3 — Cihazlar arası tekil vektör ortaklığı (ortak / min, %)",
        a3.groupby("sinif")[[c for c in a3.columns if c.startswith("pay_")]].agg(["min", "mean"]), 1)

# B
for ad, b in (("B2_kural_dedektor.csv", "B2 — 'HH_jit_*_mean > 1e9' kuralının sınıf başına işaretleme oranı (%)"),
              ("B3_zd_tarihleri.csv", "B3 — Zaman damgası tarih aralıkları")):
    bas(b, oku(ad), 2)
b4 = oku("B4_yalniz_zd_ozn.csv")
if b4 is not None:
    bas("B4 — Yalnızca 5 HH_jit_*_mean özniteliğiyle öğrenme (3 tohum ort.)", b4.groupby("hedef").mean(numeric_only=True).drop(columns="seed"))

# C
c = oku("C_faysal_tekrar.csv")
if c is not None:
    cols = ["dogruluk", "tcp_geri", "udp_geri", "tcp_to_udp", "diger_9_min_geri"]
    if "secilen_HHjit_sayisi" in c:
        cols.append("secilen_HHjit_sayisi")
    bas("C — Faysal 2022 tekrarı (5 tohum; ortalama)", c.groupby(["veri", "oznitelik", "model"])[cols].mean())
    bas("C — Faysal 2022 tekrarı (5 tohum; std)", c.groupby(["veri", "oznitelik", "model"])[cols[:4]].std())

# D
d = oku("D_protokoller.csv")
if d is not None:
    ana = ["MCC_ikili", "F1_makro_cok", "yanlis_alarm", "yakalama_gafgyt", "yakalama_mirai",
           "tur_dogrulugu_gafgyt", "tur_dogrulugu_mirai"]
    zor = [c for c in d.columns if c.startswith(("geri_", "ikili_yakalama_"))]
    bas("D — Protokol × model × öznitelik seti (3 tohum ve katların ortalaması)",
        d.groupby(["bolme", "oznitelik", "model"])[ana].mean())
    bas("D — Zor türler: çok sınıflı geri çağırma ve ikili yakalama",
        d.groupby(["bolme", "oznitelik", "model"])[zor].mean())
    bas("D — Tohumlar arası std (katlar önce tohum içinde ortalanır)",
        d.groupby(["bolme", "oznitelik", "model", "seed"])[["MCC_ikili", "F1_makro_cok", "tur_dogrulugu_mirai"]]
         .mean().groupby(level=[0, 1, 2]).std())
    s5 = d[d.bolme == "S5"]
    bas("D — S5 küme başına (Mirai türü doğruluğu; ort. ± tohum std)",
        s5.groupby(["kat", "oznitelik", "model"]).tur_dogrulugu_mirai.agg(["mean", "std"]))
    if "test_n" in d:
        bas("D — Arındırılmış bölmelerde kalan test satırı (ortalama)",
            d.groupby(["bolme"]).test_n.mean().to_frame(), 0)

# E
e = oku("E_tek_sinif.csv")
if e is not None:
    cols = [c for c in ("TPR_gafgyt:tcp", "TPR_gafgyt:udp", "AUC_gafgyt:tcp", "AUC_gafgyt:udp", "TPR_kolay", "FPR",
                        "ilk_paket_esigi_gecer") if c in e]
    bas("E — Tek sınıflı rejim (yapılandırma × öznitelik × eşik stratejisi)", e.groupby(["yapilandirma", "oznitelik", "strateji"])[cols].mean())
    e["TPR_zor"] = e[["TPR_gafgyt:tcp", "TPR_gafgyt:udp"]].mean(axis=1)
    e["uc"] = (e.TPR_zor < 0.05) | (e.TPR_zor > 0.95)
    bas("E — Zor türlerde uç (ikili) sonuç payı ve tohumlar arası kararsız cihaz sayısı",
        e.groupby(["yapilandirma", "oznitelik", "strateji"]).apply(
            lambda g: pd.Series({"uc_pay": g.uc.mean(),
                                 "kararsiz_cihaz": int((g.groupby("cihaz").TPR_zor.agg(lambda v: (v > 0.5).mean())
                                                        .between(0.01, 0.99)).sum())})))
    q = e[e.strateji == "q99_esik"]
    if "ilk_paket_esigi_gecer" in q:
        uyum = ((q["TPR_gafgyt:tcp"] > 0.5) == (q.ilk_paket_esigi_gecer > 0.5)).mean()
        parca.append(f"\n**İlk-paket hipotezi (q99):** tcp TPR>0,5 ile 'iyi huylu ilk-paket satırları eşiği geçer' "
                     f"kararının örtüşme oranı: **{uyum:.3f}** (n={len(q)})\n")
    prim = q.pivot_table(index=["oznitelik", "cihaz", "tohum"], columns="yapilandirma", values="TPR_zor")
    if {"D-D", "G-G"} <= set(prim.columns):
        prim["prim"] = prim["D-D"] - prim["G-G"]
        bas("E — Kişiselleştirme primi D-D − G-G (zor türler, q99): tohum başına ortalama",
            prim.groupby(["oznitelik", "tohum"]).prim.mean().unstack())
        bas("E — Kişiselleştirme primi: cihaz × tohum (hepsi_115)",
            prim.loc["hepsi_115"].prim.unstack() if "hepsi_115" in prim.index.get_level_values(0) else None)

# G
for ad, b in (("G1_cic2023_etiket_tekillik.csv", "G1 — CIC-IoT2023 etiket başına tekillik"),
              ("G2_cic2023_sizinti.csv", "G2 — CIC-IoT2023 rastgele bölme sızıntısı"),
              ("G3_cic2023_ozet.csv", "G3 — CIC-IoT2023 özet")):
    bas(b, oku(ad), 3)

open(f"{OUT}/FAZ6_OZET.md", "w", encoding="utf-8").write("# Faz 6 — sonuç özeti (otomatik)\n" + "".join(parca))
print("written:", f"{OUT}/FAZ6_OZET.md")
