"""Faz 7 H — Arumugam & Ramalingam 2026 (Research Square, rs-11143924) yeniden üretimi ve sızıntı denetimi.

Yazarların kodu (kaynak_kod/Code_NBaIoT, stage1–stage8) birebir izlenir:
  stage1  her dosyadan rastgele 6 blok × 500 satır (default_rng(seed)), cihazlar sıralı; benign, mirai, gafgyt
  stage2  GroupShuffleSplit(test 0,2) blok kimliğiyle; korelasyon tüm örneklemde
  stage3  MI (40k alt örneklem, k=3) + RF önemi sıra toplamı, |r|>=0,98 budama, ilk 40 öznitelik
  stage4  HistGB(max_iter=300, max_depth=10, balanced), RobustScaler; LR(max_iter=500, balanced)
  stage8  RF probu (200 ağaç, derinlik 16, balanced): bir-cihaz-dışarıda ve bir-alt-tür-dışarıda
Bizim eklediklerimiz: test satırlarının eğitimde içerik olarak bulunma oranı; zaman damgası sütunları
çıkarıldığında ve sızıntılı test satırları atıldığında aynı ölçüler; LR için işaretli log dönüşümü; çok tohum.
Kullanım: python3 faz7_H_arumugam.py <tohum> [<tohum> ...]
"""
import sys
import time

from faz7_ortak import *  # noqa: F401,F403
from faz7_ortak import ck_oku, ck_yaz, icerik_hash, isaretli_log, log, satir_hash, yukle, zd_ozn
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

N_JOBS = int(os.environ.get("FAZ7_NJOBS", "4"))


def orneklem(M, seed):
    """stage1_build_sample.py: dosya sırası ve rng çağrıları yazarların koduyla aynı."""
    rng = np.random.default_rng(seed)
    idx, blok = [], []
    for c in sorted(M.cihaz.unique()):
        dosyalar = [("benign", "benign")]
        for a in ("mirai", "gafgyt"):
            dosyalar += [(a, t) for t in sorted(M.tur[(M.cihaz == c) & (M.aile == a)].unique())]
        for a, t in dosyalar:
            satir = np.flatnonzero(((M.cihaz == c) & (M.aile == a) & (M.tur == t)).to_numpy())
            if len(satir) == 0:
                continue
            satir = satir[np.argsort(M.satir_no.to_numpy()[satir])]
            nb = len(satir) // 500
            sec = np.arange(nb) if nb <= 6 else np.sort(rng.choice(np.arange(nb), size=6, replace=False))
            for b in sec:
                idx.append(satir[b * 500:(b + 1) * 500])
                blok += [f"{c}|{a}|{t}|{b}"] * 500
    return np.concatenate(idx), np.array(blok)


def oznitelik_sec(Xtv, ytv, Xtum, ozn, seed):
    """stage3_feature_selection.py (LightGBM diz noktası taraması yerine makaledeki K*=40 kullanılır)."""
    rng = np.random.default_rng(seed)
    alt = rng.choice(len(Xtv), size=min(40000, len(Xtv)), replace=False)
    mi = mutual_info_classif(Xtv[alt], ytv[alt], random_state=seed, n_neighbors=3)
    rf = RandomForestClassifier(n_estimators=150, max_depth=14, n_jobs=N_JOBS, random_state=seed,
                                class_weight="balanced").fit(Xtv[alt], ytv[alt])
    sira = pd.DataFrame({"f": range(len(ozn)), "mi": pd.Series(mi).rank(ascending=False),
                         "rf": pd.Series(rf.feature_importances_).rank(ascending=False)})
    sira = sira.assign(t=sira.mi + sira.rf).sort_values("t")
    with np.errstate(all="ignore"):
        kor = np.corrcoef(Xtum, rowvar=False)   # yazarlar korelasyonu test dahil tüm örneklemde hesaplıyor
    tut = []
    for f in sira.f:
        if not any(abs(kor[f, k]) >= 0.98 for k in tut):
            tut.append(int(f))
    return tut[:40]


def olc(y, p, siniflar, on):
    r = {f"{on}MCC": matthews_corrcoef(y, p), f"{on}F1": f1_score(y, p, average="macro")}
    for s in ("gafgyt:tcp", "gafgyt:udp", "gafgyt:scan", "mirai:scan"):
        m = y == s
        r[f"{on}geri_{s}"] = float((p[m] == s).mean()) if m.any() else np.nan
    return r


def main(seed):
    X, M, ozn = yukle()
    Hc = icerik_hash(X, ozn)                    # içerik (zaman damgası hariç)
    Hf = satir_hash(X, "float64")               # tam satır
    zi = set(zd_ozn(ozn))
    ad = f"H_{seed}"
    SON = ck_oku(ad)
    tamam = {r["kayit"] for r in SON}

    def ekle(r):
        SON.append(r); ck_yaz(ad, SON); tamam.add(r["kayit"]); log(seed, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    idx, blok = orneklem(M, seed)
    y = M.sinif.to_numpy()[idx]
    ybin = (y != "benign").astype(int)
    siniflar = sorted(set(y))
    Xs = np.asarray(X[idx])
    tv, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(Xs, ybin, groups=blok))
    assert not set(blok[tv]) & set(blok[te])

    # 1) sızıntı denetimi: test satırı eğitimde birebir var mı (tam satır / içerik)
    if "sizinti" not in tamam:
        r = {"kayit": "sizinti", "seed": seed, "n": len(idx), "n_test": len(te),
             "tekrar_satir_ornek": int(pd.Series(Hf[idx]).duplicated().sum()),
             "tekrar_grup_ornek": int((pd.Series(Hf[idx]).value_counts() > 1).sum()),
             "test_egitimde_tam_%": 100 * np.isin(Hf[idx][te], Hf[idx][tv]).mean(),
             "test_egitimde_icerik_%": 100 * np.isin(Hc[idx][te], Hc[idx][tv]).mean()}
        for s in siniflar:
            m = y[te] == s
            r[f"icerik_sizinti_{s}_%"] = 100 * np.isin(Hc[idx][te][m], Hc[idx][tv]).mean()
        ekle(r)

    # 2) öznitelik seçimi (yazarlarınki) ve zaman damgası sütunlarının seçilip seçilmediği
    if "secim" not in tamam:
        sec = oznitelik_sec(Xs[tv], y[tv], Xs, ozn, seed)
        ekle({"kayit": "secim", "seed": seed, "secilen": [ozn[i] for i in sec],
              "secilen_zd_sayisi": sum(i in zi for i in sec),
              "secilen_HHjit_sayisi": sum(ozn[i].startswith("HH_jit") for i in sec)})
    sec = [ozn.index(c) for c in next(r for r in SON if r["kayit"] == "secim")["secilen"]]
    setler = {"sec40": sec, "sec40_zdsiz": [i for i in sec if i not in zi],
              "hepsi_115": list(range(115)), "zd_yok_110": [i for i in range(115) if i not in zi]}
    temiz = ~np.isin(Hc[idx][te], Hc[idx][tv])   # içerik olarak eğitimde bulunmayan test satırları

    # 3) HistGB ve LR, tutulan test kümesi: tam ve arındırılmış
    for oad, cols in setler.items():
        for mad in ("HistGB", "LR", "LR_log"):
            k = f"test|{mad}|{oad}"
            if k in tamam or (mad != "HistGB" and oad not in ("sec40", "sec40_zdsiz")):
                continue
            t0 = time.time()
            A, B = Xs[tv][:, cols], Xs[te][:, cols]
            if mad == "LR_log":
                A, B = isaretli_log(A), isaretli_log(B)
            if mad == "HistGB":
                mdl = HistGradientBoostingClassifier(max_iter=300, max_depth=10, random_state=seed, class_weight="balanced")
            else:
                mdl = LogisticRegression(max_iter=500, random_state=seed, class_weight="balanced")
            sc = RobustScaler().fit(A)
            mdl.fit(sc.transform(A), y[tv])
            p = mdl.predict(sc.transform(B))
            r = {"kayit": k, "seed": seed, "model": mad, "oznitelik": oad, "n_ozn": len(cols),
                 "test_n": len(te), "temiz_n": int(temiz.sum())}
            r.update(olc(y[te], p, siniflar, "tum_"))
            r.update(olc(y[te][temiz], p[temiz], siniflar, "temiz_"))
            r["sure_s"] = round(time.time() - t0, 1)
            ekle(r)

    # 4) stage8: bir-cihaz-dışarıda (RF probu, tüm örneklem) ve bir-alt-tür-dışarıda
    def rf():
        return RandomForestClassifier(n_estimators=200, max_depth=16, n_jobs=N_JOBS, random_state=seed, class_weight="balanced")
    cihaz = M.cihaz.to_numpy()[idx]
    for oad in ("sec40", "sec40_zdsiz", "zd_yok_110"):
        cols = setler[oad]
        sc = RobustScaler().fit(Xs[tv][:, cols])
        Z = sc.transform(Xs[:, cols])
        for c in sorted(set(cihaz)):
            k = f"lodo|{oad}|{c}"
            if k in tamam:
                continue
            m = cihaz == c
            p = rf().fit(Z[~m], y[~m]).predict(Z[m])
            sz = np.isin(Hc[idx][m], Hc[idx][~m])
            r = {"kayit": k, "seed": seed, "oznitelik": oad, "cihaz": c, "test_n": int(m.sum()),
                 "icerik_sizinti_%": 100 * sz.mean(),
                 "MCC": matthews_corrcoef(y[m], p),
                 "MCC_temiz": matthews_corrcoef(y[m][~sz], p[~sz]) if (~sz).sum() > 10 else np.nan}
            ekle(r)
        for s in [s for s in siniflar if s != "benign"]:
            k = f"zeroday|{oad}|{s}"
            if k in tamam:
                continue
            m = y == s
            p = rf().fit(Z[~m], y[~m]).predict(Z[m])
            ekle({"kayit": k, "seed": seed, "oznitelik": oad, "alt_tur": s, "test_n": int(m.sum()),
                  "isaretlenen_%": 100 * float((p != "benign").mean()),
                  "en_cok_tahmin": pd.Series(p).value_counts().index[0]})
    log(f"H tohum {seed} bitti")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(int(s))
