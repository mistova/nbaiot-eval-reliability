"""Faz 7 O — CIC-IoT2023: aynı öznitelik vektörünü taşıyan farklı etiketler hangileri (Faz 6 G'nin açık sorusu).
Ayrıca 11.5 metriği bileşenleri: tekrar payı, rastgele bölmede test-eğitim ikiz payı, 1-NN değil yalnızca hash.
Kullanım: python3 faz7_O_cic_celiski.py"""
import glob
import itertools

from faz7_ortak import *  # noqa: F401,F403
from faz7_ortak import log
import numpy as np


def main():
    kok = os.path.expanduser("~/work/IoT_Botnet/veri/cic2023/merged/MERGED_CSV")
    hs, ls = [], []
    for f in sorted(glob.glob(f"{kok}/*.csv")):
        df = pd.read_csv(f, engine="pyarrow", on_bad_lines="skip")
        y = df.pop("Label").astype(str).to_numpy()
        h = np.zeros(len(df), dtype=np.uint64)
        for c in df.columns:
            h = (h * np.uint64(1099511628211)) ^ pd.util.hash_array(df[c].to_numpy(np.float64))
        hs.append(h); ls.append(y)
        log(os.path.basename(f), len(df))
    H, Y = np.concatenate(hs), np.concatenate(ls)
    d = pd.DataFrame({"h": H, "y": pd.Categorical(Y)})
    say = d.groupby(["h", "y"], observed=True).size().rename("n").reset_index()
    cok = say.groupby("h").y.transform("nunique")
    c = say[cok > 1]
    log("çelişkili vektör:", c.h.nunique(), "satır:", int(c.n.sum()))
    # etiket çiftleri: her çelişkili vektör için etiketlerin tüm ikilileri; satır ağırlığı = o vektörü taşıyan satırlar
    ciftler = {}
    for h, g in c.groupby("h"):
        labs = sorted(g.y.astype(str)); n = int(g.n.sum())
        for a, b in itertools.combinations(labs, 2):
            k = (a, b)
            v = ciftler.setdefault(k, [0, 0]); v[0] += 1; v[1] += n
    t = pd.DataFrame([(a, b, v[0], v[1]) for (a, b), v in ciftler.items()],
                     columns=["etiket_a", "etiket_b", "ortak_vektor", "satir"]).sort_values("satir", ascending=False)
    t.to_csv(f"{TAB}/O_cic2023_celiskili_ciftler.csv", index=False)
    print(t.head(40).to_string(index=False))
    # etiket başına: satırlarının ne kadarı başka etiketle paylaşılan vektörde
    pay = say.assign(celiskili=cok > 1).groupby("y", observed=True).apply(
        lambda g: pd.Series({"satir": g.n.sum(), "celiskili_satir_%": 100 * g.n[g.celiskili].sum() / g.n.sum()}))
    pay.sort_values("celiskili_satir_%", ascending=False).to_csv(f"{TAB}/O_cic2023_etiket_celiski_payi.csv")
    print(pay.sort_values("celiskili_satir_%", ascending=False).round(2).to_string())
    # Li 2026 tavanı: çelişkili vektörlerde çoğunluk etiketi tahmini ile ulaşılabilecek en yüksek doğruluk
    tavan = say.groupby("h").n.max().sum() / say.n.sum()
    pd.Series({"satir": int(say.n.sum()), "tekil_vektor": int(say.h.nunique()),
               "celiskili_vektor": int(c.h.nunique()), "dogruluk_tavani": float(tavan)}).to_csv(f"{TAB}/O_cic2023_ozet.csv")
    log("O bitti; doğruluk tavanı", tavan)


if __name__ == "__main__":
    main()
