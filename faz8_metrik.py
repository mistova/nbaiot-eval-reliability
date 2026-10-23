"""Faz 8 — değerlendirme güvenilirliği metriği (Bölüm 11.5): profil, tek sayı, tehlike/kalan risk ve geçerlilik sınaması.

Metrik bir (veri seti d, protokol π, öznitelik seti φ) üçlüsü için tanımlanır.
Çekirdek profil (modelden bağımsız):
  τ0  birebir ikiz payı      test satırlarının eğitimde birebir aynı vektöre sahip olma oranı
  τε  yakın ikiz payı        işaretli log + z-skoru uzayında en yakın eğitim komşusu ≤ ε·√boyut (ε = 0,01)
  κ   çelişki tavanı         aynı vektöre farklı etiket veren kayıtlar yüzünden ulaşılabilecek en yüksek doğruluk (Li 2026)
Ekler:
  σk  kısayol eki            tek öznitelikle en yüksek makro F1 / tüm özniteliklerle makro F1 (Flood 2024 HDF_C'nin göreli hali)
  σs  çalışma noktası eki    tohumlar arası makro F1 standart sapması (3 tohum)
  ν   ezber tanısı           1-NN makro F1 / model makro F1 (çekirdekte değil: kolay veri ile ezberi ayıramıyor, Faz 7 R)
Tek sayı (alternatif):  GS = κ · (1 − τε)   — "test sonucunun ikizlere dayanmayan ve ilke olarak ulaşılabilir kısmı"
Tehlike H(d, φ) = profil(π0 rastgele); kalan risk = profil(farkında protokol); giderilebilirlik = H − kalan.

Protokoller:
  P0  rastgele %70/30
  P1  tekrar grubu ayrık (birebir aynı vektörler aynı tarafta)            → τ0 = 0 yapısal olarak
  P2  varlık dışarıda (cihaz / gün / senaryo / yakalama), varlık başına
  P3  P2 + yakın ikizleri testten at                                       → farkında kullanımın en sıkı hali
  P4  P0 + yakın ikizleri testten at
Geçerlilik: şişme(d, π, φ) = F1(d, π, φ) − F1_ref(d), F1_ref = temiz öznitelik setinde P3 (varlık yoksa P4).
Modeller: XGBoost (cuda), eğitim ≤ 200k, test örneklemi ≤ 20k, 3 tohum. Görülmemiş sınıflar (P2/P3) testten çıkarılır.
Kullanım: python3 faz8_metrik.py <veri_seti> [...]
"""
import glob
import multiprocessing as mp
import pickle
import sys
import time

from faz7_ortak import *  # noqa: F401,F403
from faz7_ortak import ck_oku, ck_yaz, log, yukle, zd_ozn
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

import faz7_R_bilesenler as R
from faz7_gpu_modeller import Onisleme, en_yakin

EPS = (0.001, 0.01, 0.1)       # ana ε = 0,01; diğerleri duyarlılık
N_TR, N_TE, TOHUM = 200_000, 20_000, (42, 7, 123)
IOT23_SENARYO_UST = 300_000

OUT8 = os.path.expanduser("~/work/IoT_Botnet/faz8_ciktilar")
TAB8 = f"{OUT8}/tables"
os.makedirs(TAB8, exist_ok=True)
os.makedirs(f"{OUT8}/checkpoint", exist_ok=True)


# ------------------------------------------------------------------ IoT-23 öznitelik yükleyicisi
SAYISAL = ["duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
KATEGORIK = ["proto", "service", "conn_state", "history"]


def _iot23_oku(yol):
    import zlib
    rs = np.random.RandomState(zlib.crc32(yol.split("/")[-3].encode()))   # senaryo adına bağlı, süreçten bağımsız
    p_kabul = min(1.0, IOT23_SENARYO_UST / max(1, os.path.getsize(yol) / 140))
    alanlar, X, Y = None, [], []
    with open(yol, errors="replace") as fh:
        for s in fh:
            if s[0] == "#":
                if s.startswith("#fields"):
                    alanlar = []
                    for a in s.strip().split("\t")[1:]:
                        alanlar.extend(a.split())
                    isy = [alanlar.index(a) for a in SAYISAL]; ik = [alanlar.index(a) for a in KATEGORIK]
                    il, idt = alanlar.index("label"), alanlar.index("detailed-label")
                continue
            if p_kabul < 1 and rs.random_sample() > p_kabul:
                continue
            p = s.rstrip("\n").split("\t")
            if len(p) < len(alanlar):
                p = p[:-1] + p[-1].split()
            if len(p) < len(alanlar):
                continue
            X.append([float(p[i]) if p[i] not in ("-", "") else -1.0 for i in isy] + [p[i] for i in ik])
            Y.append("benign" if p[il].lower() == "benign" else (p[idt] if p[idt] != "-" else "malicious"))
            if len(X) >= IOT23_SENARYO_UST:
                break
    sen = yol.split("/")[-3]
    return sen, X, Y


def yukle_iot23():
    dosyalar = sorted(glob.glob(os.path.expanduser("~/work/IoT_Botnet/veri/iot23/opt/**/conn.log.labeled"), recursive=True))
    parca, ys, vs = [], [], []
    with mp.Pool(4) as pool:
        for sen, X, Y in pool.imap_unordered(_iot23_oku, dosyalar):
            if not X:
                continue
            df = pd.DataFrame(X, columns=SAYISAL + KATEGORIK)
            for c in KATEGORIK:   # kararlı sayısal kod (dosyalar arası tutarlı)
                df[c] = (pd.util.hash_array(df[c].to_numpy(object)) % 1_000_003).astype(np.float64)
            parca.append(df.to_numpy(np.float64)); ys.append(np.array(Y, dtype=object))
            vs.append(np.full(len(Y), sen, dtype=object)); log("iot23", sen, len(Y))
    return np.concatenate(parca), np.concatenate(ys), np.concatenate(vs), SAYISAL + KATEGORIK


def veri(ad):
    if ad == "iot23":
        return yukle_iot23()
    if ad == "nbaiot_temiz":          # zaman damgası taşıyan 15 HH_jit sütununun tamamı çıkarılmış (Faz 7 L)
        X, M, ozn = yukle()
        cols = [j for j, c in enumerate(ozn) if not c.startswith("HH_jit_")]
        return np.asarray(X)[:, cols], M.sinif.to_numpy(), M.cihaz.to_numpy(), [ozn[j] for j in cols]
    R.ON_HASH = None; R.EK = {}
    return R.yukle_veri(ad)


# ------------------------------------------------------------------ yardımcılar
def tavan(H, y):
    sy = pd.DataFrame({"h": H, "y": pd.Categorical(y)}).groupby(["h", "y"], observed=True).size().rename("n").reset_index()
    return float(sy.groupby("h").n.max().sum() / sy.n.sum())


def xgb_f1(Xtr, ytr, Xte, yte, seed):
    from xgboost import XGBClassifier
    sin = np.unique(ytr); ki = {s: i for i, s in enumerate(sin)}
    m = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, tree_method="hist", device="cuda", random_state=seed)
    m.fit(Xtr.astype(np.float32), np.array([ki[s] for s in ytr]))
    p = sin[m.predict(Xte.astype(np.float32)).astype(int)]
    return p


def olc_model(y, p, ben):
    yb, pb = (y != ben).astype(int), (p != ben).astype(int)
    return {"F1": f1_score(y, p, average="macro"), "dogruluk": float((y == p).mean()),
            "MCC_ikili": matthews_corrcoef(yb, pb) if len(set(yb)) > 1 else np.nan}


def benign_adi(y):
    for a in ("benign", "BENIGN", "normal"):
        if (y == a).any():
            return a
    return pd.Series(y).value_counts().index[0]


def kisayol(X, y, seed=42):
    """Tek öznitelikle en yüksek makro F1 (100k eğitim / 40k test, rastgele)."""
    from xgboost import XGBClassifier
    rs = np.random.RandomState(seed)
    alt = rs.choice(len(y), min(len(y), 140_000), replace=False)
    a, b = alt[:100_000], alt[100_000:]
    sin = np.unique(y[a]); ki = {s: i for i, s in enumerate(sin)}
    ya = np.array([ki[s] for s in y[a]])
    en, en_j = -1.0, -1
    for j in range(X.shape[1]):
        xa = np.asarray(X[a, j:j + 1], dtype=np.float32)
        if np.unique(xa[:5000]).size < 2:
            continue
        m = XGBClassifier(n_estimators=50, max_depth=6, tree_method="hist", device="cuda").fit(xa, ya)
        f = f1_score(y[b], sin[m.predict(np.asarray(X[b, j:j + 1], dtype=np.float32)).astype(int)], average="macro")
        if f > en:
            en, en_j = f, j
    return en, en_j


# ------------------------------------------------------------------ ana döngü
def bolmeler8(n, H, v):
    """(kayit, protokol, tr, te, kat) — faz8_metrik ve faz8_eps aynı bölmeleri kullanır."""
    rs = np.random.RandomState(42)
    p = rs.permutation(n); k = int(0.7 * n)
    yield "P0", "P0", p[:k], p[k:], "-"
    yield "P4", "P4", p[:k], p[k:], "-"
    # P1: tekrar grupları ayrık
    ug = np.unique(H); test_g = rs.permutation(ug)[: int(0.3 * len(ug))]
    m = np.isin(H, test_g)
    yield "P1", "P1", np.flatnonzero(~m), np.flatnonzero(m), "-"
    if v is not None:
        for e in sorted(set(v)):
            m = v == e
            if m.sum() < 100 or (~m).sum() < 100:
                continue
            yield f"P2|{e}", "P2", np.flatnonzero(~m), np.flatnonzero(m), e
            yield f"P3|{e}", "P3", np.flatnonzero(~m), np.flatnonzero(m), e


def main(ad):
    ck = f"faz8_{ad}"
    yol = f"{OUT8}/checkpoint/{ck}.pkl"
    SON = pickle.load(open(yol, "rb")) if os.path.exists(yol) else []
    tamam = {r["kayit"] for r in SON}

    def ekle(r):
        SON.append(r); pickle.dump(SON, open(yol + ".tmp", "wb")); os.replace(yol + ".tmp", yol); tamam.add(r["kayit"])
        log(ad, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    t0 = time.time()
    X, y, v, ozn = veri(ad)
    y = np.asarray(y).astype(str)
    n = len(y)
    log(ad, "yüklendi", X.shape, f"{time.time() - t0:.0f}s")
    # MedBIoT/Kitsune yükleyicisi hash'i float64 duyarlılıkta önceden hesaplar (X bellekte float32 tutulur)
    H = R.ON_HASH if R.ON_HASH is not None and len(R.ON_HASH) == n else R.hashle(np.asarray(X, dtype=np.float64))
    ben = benign_adi(y)
    if "veri" not in tamam:
        u = len(np.unique(H))
        ekle({"kayit": "veri", "veri": ad, "satir": n, "oznitelik": X.shape[1], "sinif": len(set(y)),
              "varlik": int(len(set(v))) if v is not None else 0, "tekrar_payi": 1 - u / n, "kappa": tavan(H, y)})
    if "kisayol" not in tamam:
        f, j = kisayol(X, y)
        ekle({"kayit": "kisayol", "veri": ad, "HDF_max_F1": f, "HDF_oznitelik": ozn[j] if j >= 0 else ""})

    def bolme_degerlendir(kayit, protokol, tr, te, kat="-", yakin_filtre=False, tohumlar=TOHUM):
        if kayit in tamam or len(te) == 0 or len(tr) == 0:
            return
        rs = np.random.RandomState(0)
        te_s = te if len(te) <= N_TE else rs.choice(te, N_TE, replace=False)
        tr_nn = tr if len(tr) <= 2_000_000 else rs.choice(tr, 2_000_000, replace=False)
        on = Onisleme().fit(np.asarray(X[rs.choice(tr, min(len(tr), 300_000), replace=False)], dtype=np.float64))
        Z = np.concatenate([on(np.asarray(X[tr_nn[i:i + 500_000]], dtype=np.float64)) for i in range(0, len(tr_nn), 500_000)])
        D, I = en_yakin(Z, on(np.asarray(X[te_s], dtype=np.float64)), k=1, parca_te=2048, parca_tr=500_000)
        del Z
        d = D[:, 0] / np.sqrt(X.shape[1])
        ikiz0 = np.isin(H[te_s], np.unique(H[tr]))
        r = {"kayit": kayit, "veri": ad, "protokol": protokol, "kat": kat, "n_tr": len(tr), "n_te": len(te), "n_te_ornek": len(te_s),
             "tau0": float(ikiz0.mean())}
        for e in EPS:
            r[f"tau_{e:g}"] = float((d <= e).mean())
        nn = y[tr_nn][I[:, 0]]
        if yakin_filtre:                     # P3/P4: yakın ikizler testten atılır
            m = d > 0.01
            te_s, nn, d, ikiz0 = te_s[m], nn[m], d[m], ikiz0[m]
            r["n_te_filtre"] = int(m.sum())
        gor = np.isin(y[te_s], np.unique(y[tr]))   # görülmemiş sınıflar dışarıda
        te_s, nn = te_s[gor], nn[gor]
        r["gorulmemis_sinif_payi"] = float(1 - gor.mean()) if len(gor) else np.nan
        if len(te_s) < 50:
            ekle(r); return
        r["NN1_F1"] = f1_score(y[te_s], nn, average="macro")
        f1s = []
        for s in tohumlar:
            rs2 = np.random.RandomState(s)
            tr_m = tr if len(tr) <= N_TR else rs2.choice(tr, N_TR, replace=False)
            p = xgb_f1(np.asarray(X[tr_m]), y[tr_m], np.asarray(X[te_s]), y[te_s], s)
            o = olc_model(y[te_s], p, ben)
            f1s.append(o["F1"])
            for k2, v2 in o.items():
                r.setdefault(f"{k2}_tohum", []).append(v2)
        r["F1"] = float(np.mean(f1s)); r["F1_std"] = float(np.std(f1s))
        r["MCC_ikili"] = float(np.nanmean(r["MCC_ikili_tohum"])); r["dogruluk"] = float(np.mean(r["dogruluk_tohum"]))
        r["nu"] = r["NN1_F1"] / r["F1"] if r["F1"] > 0 else np.nan
        ekle(r)

    for kayit, protokol, tr, te, kat in bolmeler8(n, H, v):
        bolme_degerlendir(kayit, protokol, tr, te, kat=kat, yakin_filtre=protokol in ("P3", "P4"),
                          tohumlar=TOHUM[:1] if protokol in ("P2", "P3") else TOHUM)
    pd.DataFrame([{k2: v2 for k2, v2 in r.items() if not isinstance(v2, list)} for r in SON]).to_csv(f"{TAB8}/faz8_{ad}.csv", index=False)
    log(ad, "bitti", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        try:
            main(a)
        except Exception as e:
            import traceback
            log(a, "HATA", repr(e)); traceback.print_exc()
