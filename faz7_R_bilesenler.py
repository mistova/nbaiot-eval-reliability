"""Faz 7 R — değerlendirme güvenilirliği metriği (Bölüm 11.5) için ADAY BİLEŞENLERİN veri setleri arasında ölçümü.
Metrik burada önerilmiyor; yalnızca bileşenler aynı tanımla her veri setinde ölçülüyor.

Bileşenler (tümü üstveri sütunları — kimlik, IP, port, zaman — çıkarılmış öznitelik vektöründe):
  tekrar_payi          1 − tekil/satır
  celiskili_satir_%    farklı etiketlerde de görünen vektörleri taşıyan satırlar
  dogruluk_tavani      çelişkili vektörlerde çoğunluk etiketiyle ulaşılabilecek en yüksek doğruluk (Li 2026)
  rastgele_ikiz_%      rastgele %70/30 bölmede test satırının eğitimde birebir bulunma oranı (3 tohum)
  varlik_disi_ikiz_%   bir varlık (cihaz/gün/senaryo/yakalama) dışarıda bırakıldığında satırlarının geri kalanda bulunma oranı
  NN1_F1_rastgele / NN1_F1_temiz   1-NN makro F1: rastgele test / eğitimde ikizi olmayan test (ezber payı)
  HDF_max_F1           tek öznitelikle en yüksek makro F1 (Flood 2024 HDF_C; GPU XGB, 100k alt örneklem)
Kullanım: python3 faz7_R_bilesenler.py <veri_seti> [...]   veri setleri: nbaiot cic2023 cicids2017 cicids2017_engelen botiot medbiot kitsune
"""
import glob
import sys
import zipfile

from faz7_ortak import *  # noqa: F401,F403
from faz7_ortak import ck_oku, ck_yaz, log, yukle, zd_ozn
import numpy as np
from sklearn.metrics import f1_score

from faz7_gpu_modeller import Onisleme, en_yakin

VERI = os.path.expanduser("~/work/IoT_Botnet/veri")
METAVERI = {"flow id", "source ip", "src ip", "destination ip", "dst ip", "source port", "src port",
            "destination port", "dst port", "timestamp", "pkseqid", "stime", "ltime", "saddr", "daddr", "sport",
            "dport", "seq", "attempted category", "ts", "uid"}


def _sayisal(df):
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan).fillna(-1).to_numpy(np.float64)


def yukle_veri(ad):
    """Döndürür: X (float64), y (str etiket), varlik (str veya None), oznitelik adları."""
    global ON_HASH, EK
    if ad == "nbaiot":
        X, M, ozn = yukle()
        return X, M.sinif.to_numpy(), M.cihaz.to_numpy(), ozn
    if ad == "nbaiot_zdsiz":
        X, M, ozn = yukle()
        zi = set(zd_ozn(ozn)); cols = [j for j in range(len(ozn)) if j not in zi]
        return np.asarray(X)[:, cols], M.sinif.to_numpy(), M.cihaz.to_numpy(), [ozn[j] for j in cols]
    if ad == "cic2023":
        parca, ys = [], []
        for f in sorted(glob.glob(f"{VERI}/cic2023/merged/MERGED_CSV/*.csv")):
            df = pd.read_csv(f, engine="pyarrow", on_bad_lines="skip")
            ys.append(df.pop("Label").astype(str).to_numpy()); parca.append(_sayisal(df)); ozn = list(df.columns)
        return np.concatenate(parca), np.concatenate(ys), None, ozn
    if ad in ("cicids2017", "cicids2017_engelen"):
        if ad == "cicids2017":
            dosyalar = [(os.path.basename(f), lambda f=f: pd.read_csv(f, engine="pyarrow"))
                        for f in sorted(glob.glob(f"{VERI}/cicids2017/orijinal/*.csv"))]
        else:
            z = zipfile.ZipFile(f"{VERI}/cicids2017/engelen_dataset.zip")
            dosyalar = [(os.path.basename(n), lambda n=n: pd.read_csv(z.open(n), low_memory=False))
                        for n in sorted(z.namelist()) if n.lower().endswith(".csv")]
        parca, ys, vs = [], [], []
        for isim, oku in dosyalar:
            df = oku(); df.columns = [c.strip() for c in df.columns]
            lab = [c for c in df.columns if c.lower() == "label"][0]
            ys.append(df.pop(lab).astype(str).str.strip().to_numpy())
            df = df[[c for c in df.columns if c.lower() not in METAVERI]]
            parca.append(_sayisal(df)); vs.append(np.full(len(df), isim.split("-")[0].split(".")[0], dtype=object))
            ozn = list(df.columns)
            log(ad, isim, len(df))
        return np.concatenate(parca), np.concatenate(ys), np.concatenate(vs), ozn
    if ad == "botiot":
        parca, ys = [], []
        for f in sorted(glob.glob(f"{VERI}/botiot/bes_yuzde/*.csv")):
            df = pd.read_csv(f, low_memory=False)
            ys.append((df["category"].astype(str) + ":" + df["subcategory"].astype(str)).to_numpy())
            df = df.drop(columns=[c for c in df.columns if c.lower() in METAVERI or c in ("attack", "category", "subcategory")])
            df = df.select_dtypes(include=[np.number])
            parca.append(_sayisal(df)); ozn = list(df.columns)
        return np.concatenate(parca), np.concatenate(ys), None, ozn
    if ad == "medbiot":
        parca, ys, vs, hs, zd = [], [], [], [], []
        for f in sorted(glob.glob(f"{VERI}/medbiot/fine-grained/structured_dataset/*/*.csv")):
            b = os.path.basename(f)[:-4].split("_")          # ör. mirai_mal_CC_fan, bashlite_leg_lock
            aile, tur = b[0], b[1]
            cihaz = b[-1]
            etiket = "normal" if tur == "leg" else f"{aile}_{'_'.join(b[2:-1]) or 'mal'}"
            df = pd.read_csv(f, engine="pyarrow")
            df = df[[c for c in df.columns if c.lower() not in METAVERI]].select_dtypes(include=[np.number])
            x = _sayisal(df); del df
            hs.append(hashle(x))                    # hash tam duyarlıkta; bellek için float32 saklanır
            zd.append(int((x[:, [j for j, c in enumerate(ozn_) if c.startswith("HH_jit") and "_mean" in c]] > 1e9).any(1).sum())
                      if (ozn_ := list(pd.read_csv(f, nrows=0).columns)) else 0)
            parca.append(x.astype(np.float32)); del x
            ys.append(np.full(len(parca[-1]), etiket, dtype=object))
            vs.append(np.full(len(parca[-1]), cihaz, dtype=object)); ozn = ozn_
            log(ad, b, len(parca[-1]), "zd satır:", zd[-1])
        ON_HASH = np.concatenate(hs); EK = {"jit_mean_zd_satir": int(sum(zd))}
        return np.concatenate(parca), np.concatenate(ys), np.concatenate(vs), ozn
    if ad == "kitsune":
        parca, ys, vs, hs = [], [], [], []
        for d in sorted(glob.glob(f"{VERI}/kitsune/ac/*/")):
            fx = glob.glob(f"{d}*dataset.csv*"); fl = glob.glob(f"{d}*labels.csv*")
            if not fx or not fl:
                continue
            Xd = pd.read_csv(fx[0], header=None, engine="pyarrow").to_numpy(np.float64)
            if Xd.shape[1] == 116:                 # Mirai dosyasında sıra numarası sütunu
                Xd = Xd[:, 1:]
            yl = pd.read_csv(fl[0]).iloc[:, -1].to_numpy()
            n = min(len(Xd), len(yl)); isim = os.path.basename(d.rstrip("/"))
            hs.append(hashle(Xd[:n])); parca.append(Xd[:n].astype(np.float32)); del Xd
            ys.append(np.where(yl[:n] > 0, isim, "benign").astype(object))
            vs.append(np.full(n, isim, dtype=object)); log(ad, isim, n, "saldırı payı", float((yl[:n] > 0).mean()))
        ON_HASH = np.concatenate(hs)
        X = np.concatenate(parca)
        return X, np.concatenate(ys), np.concatenate(vs), [f"f{j}" for j in range(X.shape[1])]
    raise ValueError(ad)


ON_HASH, EK = None, {}


def hashle(X):
    h = np.zeros(len(X), dtype=np.uint64)
    for j in range(X.shape[1]):
        h = (h * np.uint64(1099511628211)) ^ pd.util.hash_array(np.ascontiguousarray(X[:, j]))
    return h


def main(ad):
    SON = ck_oku("R")
    if any(r["veri"] == ad for r in SON):
        log(ad, "zaten var"); return
    X, y, v, ozn = yukle_veri(ad)
    n = len(y)
    log(ad, "yüklendi", X.shape)
    H = ON_HASH if ON_HASH is not None and len(ON_HASH) == n else hashle(X)
    r = {"veri": ad, "satir": n, "oznitelik": X.shape[1], "sinif": len(set(y)), **EK}
    u = np.unique(H)
    r["tekrar_payi"] = 1 - len(u) / n
    sy = pd.DataFrame({"h": H, "y": pd.Categorical(y)}).groupby(["h", "y"], observed=True).size().rename("n").reset_index()
    cok = sy.groupby("h").y.transform("nunique") > 1
    r["celiskili_satir_%"] = 100 * sy.n[cok].sum() / n
    r["dogruluk_tavani"] = float(sy.groupby("h").n.max().sum() / n)
    del sy
    iz = []
    for seed in (42, 7, 123):
        p = np.random.RandomState(seed).permutation(n); k = int(0.7 * n)
        iz.append(100 * np.isin(H[p[k:]], np.unique(H[p[:k]])).mean())
    r["rastgele_ikiz_%"] = float(np.mean(iz)); r["rastgele_ikiz_std"] = float(np.std(iz))
    if v is not None:
        vals = []
        for e in np.unique(v):
            m = v == e
            vals.append(100 * np.isin(H[m], np.unique(H[~m])).mean())
        r["varlik_sayisi"] = len(vals)
        r["varlik_disi_ikiz_%"] = float(np.mean(vals)); r["varlik_disi_ikiz_max_%"] = float(np.max(vals))
    # 1-NN ezber (rastgele bölme, eğitim en çok 2M, test 20k)
    rs = np.random.RandomState(42)
    p = rs.permutation(n); k = int(0.7 * n)
    tr, te = p[:k][:2_000_000], p[k:]
    te_s = rs.choice(te, min(20_000, len(te)), replace=False)
    ikiz = np.isin(H[te_s], np.unique(H[tr]))
    on = Onisleme().fit(X[rs.choice(tr, min(len(tr), 300_000), replace=False)])
    _, I = en_yakin(on(X[tr]), on(X[te_s]), k=1, parca_te=2048, parca_tr=500_000)
    pr = y[tr][I[:, 0]]
    r["NN1_F1_rastgele"] = f1_score(y[te_s], pr, average="macro")
    r["NN1_dogruluk_rastgele"] = float((pr == y[te_s]).mean())
    r["NN1_F1_temiz"] = f1_score(y[te_s][~ikiz], pr[~ikiz], average="macro") if (~ikiz).sum() > 100 else np.nan
    r["test_ikiz_orneklem_%"] = 100 * ikiz.mean()
    # HDF_C: tek öznitelikle en yüksek makro F1
    from xgboost import XGBClassifier
    alt = rs.choice(n, min(n, 140_000), replace=False)
    sin = np.array(sorted(set(y))); yk = np.searchsorted(sin, y[alt])
    a, b = alt[:100_000], np.arange(100_000, len(alt))
    en, en_ad = -1, None
    for j in range(X.shape[1]):
        xa = X[alt, j:j + 1].astype(np.float32)
        if np.unique(xa[:5000]).size < 2:
            continue
        var = np.unique(yk[:100_000]); har = np.full(len(sin), -1); har[var] = np.arange(len(var))
        m = XGBClassifier(n_estimators=50, max_depth=6, tree_method="hist", device="cuda").fit(xa[:100_000], har[yk[:100_000]])
        f = f1_score(yk[100_000:], var[m.predict(xa[100_000:]).astype(int)], average="macro")
        if f > en:
            en, en_ad = f, ozn[j]
    r["HDF_max_F1"] = float(en); r["HDF_oznitelik"] = en_ad
    SON.append(r); ck_yaz("R", SON)
    pd.DataFrame(SON).to_csv(f"{TAB}/R_bilesenler.csv", index=False)
    log(ad, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.items()})


if __name__ == "__main__":
    for a in sys.argv[1:]:
        try:
            main(a)
        except Exception as e:  # bir veri setindeki hata diğerlerini durdurmasın
            log(a, "HATA", repr(e))
