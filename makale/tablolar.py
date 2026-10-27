"""Paper tables: generates LaTeX bodies from the result files (to avoid manual transcription errors).
Output: makale/tablolar/*.tex (tabular rows only; caption and frame are in main.tex)."""
import os

import numpy as np
import pandas as pd

K = os.path.expanduser("~/work/IoT_Botnet")
OUT = f"{K}/makale/tablolar"
os.makedirs(OUT, exist_ok=True)
VERI = {"nbaiot": "N-BaIoT", "nbaiot_temiz": "N-BaIoT, timestamp-free", "botiot": "Bot-IoT (5\\%)", "iot23": "IoT-23",
        "cicids2017": "CICIDS2017", "cicids2017_engelen": "CICIDS2017 (Engelen)", "cic2023": "CIC-IoT2023",
        "kitsune": "Kitsune", "medbiot": "MedBIoT"}
SIRA = ["nbaiot", "nbaiot_temiz", "botiot", "iot23", "cicids2017", "cicids2017_engelen", "cic2023", "kitsune", "medbiot"]


def yaz(ad, satirlar):
    open(f"{OUT}/{ad}.tex", "w").write("\n".join(satirlar) + "\n")
    print(ad, len(satirlar), "rows")


def f(x, n=3):
    return "--" if pd.isna(x) else f"{x:.{n}f}"


# ---------------------------------------------------------------- uniqueness (gafgyt tcp/udp, three definitions)
a1 = pd.read_csv(f"{K}/faz6_ciktilar/tables/A1_dosya_tekillik.csv")
s = []
for tur in ("tcp", "udp"):
    g = a1[(a1.aile == "gafgyt") & (a1.tur == tur)]
    n = g.satir.sum()
    s.append(f"gafgyt {tur} & {n:,} & {g.tekil_float64.sum():,} ({100 * g.tekil_float64.sum() / n:.1f}\\%) & "
             f"{g.tekil_float32.sum():,} ({100 * g.tekil_float32.sum() / n:.2f}\\%) & "
             f"{g.tekil_float64_zdsiz.sum():,} ({100 * g.tekil_float64_zdsiz.sum() / n:.2f}\\%) \\\\")
b = a1[a1.aile == "benign"]
s.append(f"benign & {b.satir.sum():,} & {b.tekil_float64.sum():,} ({100 * b.tekil_float64.sum() / b.satir.sum():.1f}\\%) & "
         f"{b.tekil_float32.sum():,} ({100 * b.tekil_float32.sum() / b.satir.sum():.1f}\\%) & "
         f"{b.tekil_float64_zdsiz.sum():,} ({100 * b.tekil_float64_zdsiz.sum() / b.satir.sum():.1f}\\%) \\\\")
diger = a1[(a1.aile != "benign") & ~((a1.aile == "gafgyt") & a1.tur.isin(["tcp", "udp"]))]
s.append(f"other 8 attacks & {diger.satir.sum():,} & \\multicolumn{{3}}{{c}}{{100\\% unique under all three definitions}} \\\\")
yaz("t_tekillik", [x.replace(",", "{,}") if False else x for x in s])

# ---------------------------------------------------------------- share of timestamped rows
b1 = pd.read_csv(f"{K}/faz6_ciktilar/tables/B1_zd_sutunlar.csv").set_index("oznitelik")
m = b1.loc[[i for i in b1.index if i.endswith("_mean")]].max()
v = b1.loc[[i for i in b1.index if i.endswith("_variance")]].max()
u = pd.read_csv(f"{K}/faz6_ciktilar/tables/B1b_zd_birlesim.csv").set_index("s")
s = []
for c in b1.columns:
    s.append(f"{c.replace(':', ' ')} & {u.loc[c, 'm']:.2f} & {u.loc[c, 'v']:.2f} & {u.loc[c, 'u']:.2f} \\\\")
yaz("t_zamandamgasi", s)

# ---------------------------------------------------------------- cross-device overlap (content definition)
a3 = pd.read_csv(f"{K}/faz6_ciktilar/tables/A3_cihazlar_arasi.csv")
KUME = {"Danmini_Doorbell": 1, "Philips_B120N10_Baby_Monitor": 1, "Ecobee_Thermostat": 2,
        "SimpleHome_XCS7_1002_WHT_Security_Camera": 2, "SimpleHome_XCS7_1003_WHT_Security_Camera": 2,
        "Provision_PT_737E_Security_Camera": 3, "Provision_PT_838_Security_Camera": 3}
s = []
for sinif, g in a3.groupby("sinif"):
    x = g["pay_float64_zdsiz_%"]
    if sinif.startswith("gafgyt"):
        s.append(f"{sinif.replace(':', ' ')} & {len(g)} & {x.min():.1f} & {x.mean():.1f} & -- \\\\")
    else:
        ayni = g.apply(lambda r: KUME[r.cihaz_a] == KUME[r.cihaz_b], axis=1)
        s.append(f"{sinif.replace(':', ' ')} & {len(g)} & {x[ayni].min():.1f}--{x[ayni].max():.1f} & -- & {x[~ayni].max():.1f} \\\\")
yaz("t_cihazlar", s)

# ---------------------------------------------------------------- profile
p = pd.read_csv(f"{K}/faz8_ciktilar/tables/faz8_profil.csv")
s = []
for vv in SIRA:
    g = p[p.veri == vv]
    h = g[g.protokol == "P0"].iloc[0]
    fk = g[g.protokol == "P2"] if (g.protokol == "P2").any() else g[g.protokol == "P1"]
    k = fk.iloc[0]
    gs = h.kappa * (1 - h.tau_e_01)
    s.append(f"{VERI[vv]} & {h.tekrar_payi:.2f} & {h.kappa:.3f} & {h.tau0:.2f} & {h.tau_e:.2f} & {h.tau_e_01:.2f} & "
             f"{gs:.3f} & {k.protokol} & {k.tau_e_01:.2f} & {h.tau_e_01 - k.tau_e_01:.2f} & {h.F1:.3f} & {h.sisme:+.3f} \\\\")
yaz("t_profil", s)

# ---------------------------------------------------------------- validity (ε candidates, selected rows)
A = pd.read_csv(f"{K}/faz8_ciktilar/tables/faz8_eps_adaylar.csv")
A["e"] = A.aday.str.extract(r"=\s*([0-9.e-]+)").astype(float)
s = []
for aile, e in (("sabit", 0.001), ("sabit", 0.01), ("sabit", 0.0562), ("sabit", 0.1), ("sabit", 0.178), ("sabit", 0.316),
                ("uyarlamalı", 10.0), ("uyarlamalı", 1.0)):
    r = A[(A.aile == aile) & (np.isclose(A.e, e, rtol=0.02))].iloc[0]
    sira = int((A["TOPSIS_eşit"] > r["TOPSIS_eşit"]).sum()) + 1
    ad = f"$\\varepsilon={e:g}$" if aile == "sabit" else f"$\\varepsilon={e:g}\\,s$"
    s.append(f"{ad} & {r.C1_arasi:.2f} & {r.C2_ici:.2f} & {r.C3_en_kotu_LODO:.2f} & {r.C4_yayilim:.3f} & "
             f"{r.C5_kararsizlik:.3f} & {r['TOPSIS_eşit']:.3f} & {sira} \\\\")
yaz("t_eps", s)

# ---------------------------------------------------------------- corruption summary
d = pd.read_csv(f"{K}/faz8_ciktilar/tables/faz8c_bozma.csv")
KOS = [("taban", 0, "none"), ("gurultu", 0.01, "noise $\\sigma=0.01$"), ("gurultu", 0.05, "noise $\\sigma=0.05$"),
       ("nicem", 2, "2 significant digits"), ("seyrelt", 16, "every 16th row"), ("boyut", 0.25, "25\\% of features")]
s = []
for tur, dz, ad in KOS:
    hucre = []
    for vv in ("medbiot", "cicids2017_engelen", "nbaiot_temiz", "kitsune"):
        r = d[(d.veri == vv) & (d.bozma == tur) & np.isclose(d.duzey, dz)].iloc[0]
        if pd.isna(r.eps_yildiz) or r.AUC < 0.6:
            hucre.append("n.i.")
        else:
            e = r.eps_yildiz
            hucre.append("$<10^{-5}$" if e < 1e-5 + 1e-12 else f"{e:.2g}")
    s.append(f"{ad} & " + " & ".join(hucre) + " \\\\")
yaz("t_bozma", s)
