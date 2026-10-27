"""Collects the Faz 7 results into a single markdown report: faz7_ciktilar/FAZ7_OZET.md (+ combined CSVs under tables/)."""
import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.expanduser("~/work/pylibs"))

import pandas as pd

OUT = os.path.expanduser("~/work/IoT_Botnet/faz7_ciktilar")
TAB, CK = f"{OUT}/tables", f"{OUT}/checkpoint"
parca = []


def ck(desen):
    rows = []
    for f in sorted(glob.glob(f"{CK}/{desen}.pkl")):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows)


def oku(ad):
    yol = f"{TAB}/{ad}"
    return pd.read_csv(yol) if os.path.exists(yol) else None


def bas(baslik, df, ondalik=4, aciklama=""):
    parca.append(f"\n## {baslik}\n")
    if aciklama:
        parca.append(aciklama + "\n\n")
    parca.append("_(tablo yok)_\n" if df is None or len(df) == 0 else df.round(ondalik).to_markdown() + "\n")


def tohum_ort(df, anahtar, cols):
    """Folds are first averaged within each seed; then mean and std across seeds."""
    g = df.groupby(anahtar + ["seed"])[cols].mean()
    return g.groupby(level=list(range(len(anahtar)))).mean(), g.groupby(level=list(range(len(anahtar)))).std()


# Q + J — protocols × 7 models × 3 feature sets × 5 seeds
Q, J = ck("Q_*"), ck("J_*")
A = pd.concat([Q, J], ignore_index=True)
if len(A):
    A.to_csv(f"{TAB}/QJ_protokoller.csv", index=False)
    sira = ["S1", "S1f", "S2", "S3", "S3f", "S4", "S4f", "S5", "S6", "S6f"]
    for met, ad in (("F1_makro_cok", "çok sınıflı makro F1"), ("MCC_ikili", "ikili MCC"),
                    ("geri_gafgyt:tcp", "gafgyt:tcp çok sınıflı geri çağırma")):
        m, s = tohum_ort(A, ["bolme", "oznitelik", "model"], [met])
        t = m[met].unstack("model").reindex(sira, level=0)
        bas(f"Q+J — {ad} (5 tohum ortalaması)", t, 3,
            "S6 = 500 satırlık blok ayrık bölme (Arumugam 2026); 'f' = içerik ikizi eğitimde olan test satırları atılmış.")
        if met == "F1_makro_cok":
            bas(f"Q+J — {ad}: tohumlar arası std", s[met].unstack("model").reindex(sira, level=0), 3)
    if "test_sizinti" in Q:
        bas("Q — testte içerik ikizi eğitimde bulunan satır payı", Q.groupby("bolme").test_sizinti.mean().to_frame(), 3)

# H — Arumugam reproduction
H = ck("H_*")
if len(H):
    H.drop(columns=["secilen"], errors="ignore").to_csv(f"{TAB}/H_arumugam.csv", index=False)
    H[H.kayit == "secim"][["seed", "secilen_zd_sayisi", "secilen_HHjit_sayisi", "secilen"]].to_csv(f"{TAB}/H_secilen_oznitelikler.csv", index=False)
    s = H[H.kayit == "sizinti"]
    bas("H — Arumugam: örneklem tekrarı ve tutulan test sızıntısı (11 tohum)",
        s[["seed", "tekrar_satir_ornek", "tekrar_grup_ornek", "test_egitimde_tam_%", "test_egitimde_icerik_%",
           "icerik_sizinti_gafgyt:tcp_%", "icerik_sizinti_gafgyt:udp_%"]].set_index("seed").sort_index(), 2)
    t = H[H.kayit.str.startswith("test|")]
    bas("H — tutulan test (tüm / içerik ikizi atılmış); ortalama ve std",
        t.groupby(["model", "oznitelik"])[["tum_MCC", "temiz_MCC", "tum_F1", "temiz_F1", "tum_geri_gafgyt:tcp"]].agg(["mean", "std"]), 4)
    l = H[H.kayit.str.startswith("lodo|")]
    bas("H — bir-cihaz-dışarıda (RF probu)", l.groupby(["oznitelik", "cihaz"])[["MCC", "MCC_temiz", "icerik_sizinti_%"]].mean(), 4)
    z = H[H.kayit.str.startswith("zeroday|")]
    bas("H — 'sıfırıncı gün' (bir alt tür dışarıda): işaretlenme oranı ve en sık tahmin",
        z.groupby(["oznitelik", "alt_tur"]).agg(isaretlenen=("isaretlenen_%", "mean"),
                                               en_sik=("en_cok_tahmin", lambda x: x.value_counts().index[0])), 2)

# I — memorization
I = oku("I_ezber.csv")
bas("I — 1-NN ezber taban çizgisi ve yakın tekrar (tam eğitim kümesi, 2 tohum)",
    None if I is None else I.groupby(["bolme", "oznitelik"])[
        ["icerik_tekrar_%", "yakin_tekrar_0.01_%", "yakin_tekrar_0.1_%", "NN1_F1", "NN1_dogruluk"]].mean(), 3)

# K — shortcut
K = oku("K_kisayol.csv")
if K is not None:
    k1 = K[K.kayit.str.startswith("K1")]
    bas("K1 — tek öznitelikle 11 sınıf: en iyi 10 (S1 ve S6f)",
        pd.concat([k1[k1.bolme == b].nlargest(10, "F1_makro")[["bolme", "oznitelik", "F1_makro", "MCC", "geri_gafgyt:tcp"]]
                   for b in ("S1", "S6f")]).set_index(["bolme", "oznitelik"]), 3)
    k2 = K[K.kayit.str.startswith("K2")]
    bas("K2 — öznitelik grupları (makro F1)", k2.pivot_table(index=["tip", "grup"], columns=["bolme", "kip"], values="F1_makro"), 3)
    k3 = K[K.kayit.str.startswith("K3")]
    bas("K3 — saldırı satırlarından cihaz tanınabilirliği (doğruluk; şans = çoğunluk sınıfı)",
        k3.groupby(["sinif", "oznitelik"])[["dogruluk", "sans_cogunluk", "cihaz_sayisi"]].mean(), 3)

# P — Meidan
P = oku("P_meidan.csv")
if P is not None:
    c = [x for x in ("FPR_pencere", "FPR_ornek", "TPR_ornek_tum", "TPR_ornek_gafgyt:tcp", "TPR_tekil_gafgyt:tcp", "ws",
                     "tst_iyi_trn_icerik_%") if x in P]
    bas("P — Meidan 2018 protokolü (9 cihaz × 3 tohum)", P.groupby(["oznitelik", "onisleme"])[c].mean(), 4)
    bas("P — gafgyt:tcp TPR, cihaz × (öznitelik, ön işleme); tohumlar '/' ile",
        P.pivot_table(index="cihaz", columns=["oznitelik", "onisleme"], values="TPR_ornek_gafgyt:tcp",
                      aggfunc=lambda v: "/".join(f"{x:.2f}" for x in v)), 2)

# L — Kitsune
bas("L — Kitsune (UCI) yayımlanmış öznitelikler ve init_time varyantları", oku("L_kitsune.csv"), 4)

# N, O, R
for ad, b in (("N_iot23_ozet.csv", "N — IoT-23 özet"), ("O_cic2023_ozet.csv", "O — CIC-IoT2023 özet"),
              ("O_cic2023_celiskili_ciftler.csv", "O — CIC-IoT2023 çelişkili etiket çiftleri"),
              ("R_bilesenler.csv", "R — değerlendirme güvenilirliği metriği için aday bileşenler (veri setleri arası)")):
    d = oku(ad)
    if d is not None and ad == "O_cic2023_celiskili_ciftler.csv":
        d = d.head(15)
    bas(b, d, 4)
n = oku("N_iot23_ayrinti.csv")
if n is not None and "senaryo" in n:
    bas("N — IoT-23 bir-senaryo-dışarıda: senaryo satırlarının diğer senaryolarda bulunma payı",
        n[n.senaryo.notna()].pivot_table(index="senaryo", columns="vektor", values="diger_senaryoda_%"), 2)

open(f"{OUT}/FAZ7_OZET.md", "w", encoding="utf-8").write("# Faz 7 — sonuç özeti (otomatik)\n" + "".join(parca))
print("written:", f"{OUT}/FAZ7_OZET.md")
