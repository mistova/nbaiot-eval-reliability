"""Faz 7 N — IoT-23 (23 senaryo, conn.log.labeled) tekrar / çelişki / sızıntı denetimi (eğitim yok).

Öznitelik vektörü, IoT-23 çalışmalarında tipik kullanılan akış alanlarıdır (zaman, uid, IP'ler hariç):
  akis      : proto service duration orig_bytes resp_bytes conn_state missed_bytes history orig_pkts orig_ip_bytes resp_pkts resp_ip_bytes
  akis_port : akis + id.resp_p (hedef port; D'hooge 2022'nin üstveri uyarısı)
Ölçüler: etiket ve ayrıntılı etiket başına tekillik; aynı vektörün iyi ve kötü huylu etiket taşıması;
rastgele %70/30 bölmede test-eğitim ikiz payı; bir-senaryo-dışarıda: senaryo satırlarının diğer senaryolarda
bulunma payı; Li 2026 doğruluk tavanı.
Kullanım: python3 faz7_N_iot23.py   (senaryolar 4 süreçte paralel okunur)
"""
import glob
import multiprocessing as mp

from faz7_ortak import *  # noqa: F401,F403
from faz7_ortak import log
import numpy as np

AKIS = ["proto", "service", "duration", "orig_bytes", "resp_bytes", "conn_state", "missed_bytes", "history",
        "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
PARCA = 2_000_000


def oku(yol):
    alanlar, n = None, 0
    LKOD, DKOD = {}, {}   # etiket → küçük tamsayı (süreçler arası taşınan veri küçük kalsın)
    hA, hP, lab, det = [], [], [], []
    bufA, bufP, bufL, bufD = [], [], [], []

    def bosalt():
        if bufA:
            hA.append(pd.util.hash_array(np.array(bufA, dtype=object)))
            hP.append(pd.util.hash_array(np.array(bufP, dtype=object)))
            lab.append(np.array([LKOD.setdefault(x, len(LKOD)) for x in bufL], dtype=np.int16))
            det.append(np.array([DKOD.setdefault(x, len(DKOD)) for x in bufD], dtype=np.int16))
            bufA.clear(); bufP.clear(); bufL.clear(); bufD.clear()

    with open(yol, errors="replace") as fh:
        for s in fh:
            if s[0] == "#":
                if s.startswith("#fields"):
                    alanlar = []
                    for a in s.strip().split("\t")[1:]:
                        alanlar.extend(a.split())
                    ia = [alanlar.index(a) for a in AKIS]
                    ip, il, idt = alanlar.index("id.resp_p"), alanlar.index("label"), alanlar.index("detailed-label")
                continue
            p = s.rstrip("\n").split("\t")
            if len(p) < len(alanlar):
                p = p[:-1] + p[-1].split()
            if len(p) < len(alanlar):
                continue
            a = "\t".join(p[i] for i in ia)
            bufA.append(a); bufP.append(a + "\t" + p[ip])
            bufL.append(p[il].lower()); bufD.append(p[idt])
            n += 1
            if len(bufA) >= PARCA:
                bosalt()
    bosalt()
    sen = yol.split("/")[-3]
    ters = lambda d: np.array(sorted(d, key=d.get), dtype=object)
    return (sen, np.concatenate(hA), np.concatenate(hP),
            pd.Categorical.from_codes(np.concatenate(lab), ters(LKOD)),
            pd.Categorical.from_codes(np.concatenate(det), ters(DKOD)))


def main():
    dosyalar = sorted(glob.glob(os.path.expanduser("~/work/IoT_Botnet/veri/iot23/opt/**/conn.log.labeled"), recursive=True))
    # büyük dosyalar önce
    dosyalar.sort(key=os.path.getsize, reverse=True)
    S, HA, HP, L, D = [], [], [], [], []
    with mp.Pool(4) as pool:
        for sen, ha, hp, la, de in pool.imap_unordered(oku, dosyalar):
            S.append(pd.Categorical.from_codes(np.zeros(len(ha), np.int8), [sen]))
            HA.append(ha); HP.append(hp); L.append(la); D.append(de)
            log(sen, len(ha))
    from pandas.api.types import union_categoricals
    sen, lab, det = union_categoricals(S), union_categoricals(L), union_categoricals(D)
    del S, L, D
    labarr = np.asarray(lab.astype(str))   # yalnızca 'benign' / 'malicious' (ve küçük varyantlar)
    ozet, satirlar = {}, []
    for vad, H in (("akis", np.concatenate(HA)), ("akis_port", np.concatenate(HP))):
        n = len(H)
        ozet[f"{vad}_satir"] = n
        ozet[f"{vad}_tekil_%"] = 100 * len(np.unique(H)) / n
        df = pd.DataFrame({"h": H, "y": lab})
        sy = df.groupby(["h", "y"], observed=True).size().rename("n").reset_index()
        cok = sy.groupby("h").y.transform("nunique") > 1
        ozet[f"{vad}_celiskili_satir_%"] = 100 * sy.n[cok].sum() / n
        ozet[f"{vad}_dogruluk_tavani"] = float(sy.groupby("h").n.max().sum() / n)
        dk = det.codes
        for ci, g in enumerate(det.categories):
            gi = np.flatnonzero(dk == ci)
            if len(gi) == 0:
                continue
            satirlar.append({"vektor": vad, "ayrintili_etiket": g, "satir": len(gi),
                             "tekillik_%": 100 * len(np.unique(H[gi])) / len(gi)})
        for seed in (42, 7, 123):
            p = np.random.RandomState(seed).permutation(n); k = int(0.7 * n)
            ici = np.isin(H[p[k:]], np.unique(H[p[:k]]))
            ozet[f"{vad}_rastgele_sizinti_%_s{seed}"] = 100 * ici.mean()
            yt = labarr[p[k:]]
            for e in ("benign", "malicious"):
                ozet[f"{vad}_rastgele_sizinti_{e}_%_s{seed}"] = 100 * ici[yt == e].mean()
        # bir-senaryo-dışarıda: senaryonun satırlarının ne kadarı başka senaryoda da var
        sk = sen.codes
        for si, s in enumerate(sen.categories):
            m = sk == si
            diger = np.unique(H[~m])
            ici = np.isin(H[m], diger)
            r = {"vektor": vad, "senaryo": s, "satir": int(m.sum()), "diger_senaryoda_%": 100 * ici.mean()}
            yt = labarr[m]
            for e in ("benign", "malicious"):
                r[f"{e}_satir"] = int((yt == e).sum())
                r[f"{e}_diger_senaryoda_%"] = 100 * ici[yt == e].mean() if (yt == e).any() else np.nan
            satirlar.append(r)
        log(vad, "tamam")
    pd.Series(ozet).to_csv(f"{TAB}/N_iot23_ozet.csv")
    pd.DataFrame(satirlar).to_csv(f"{TAB}/N_iot23_ayrinti.csv", index=False)
    print(pd.Series(ozet).round(3).to_string())
    log("N bitti")


if __name__ == "__main__":
    main()
