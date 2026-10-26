"""Makale şekilleri (J Comput Virol Hack Tech, Springer). Girdi: faz çıktı tabloları. Çıktı: makale/figures/Fig*.pdf
Sütun genişliği 84 mm, tam genişlik 174 mm; yazı Nimbus Sans (Helvetica eşdeğeri); renk: dataviz referans paleti
ilk üç yuva (tüm çiftlerde CVD doğrulamalı), her seride ayrıca işaretçi/yazı ile ikincil kodlama."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

K = os.path.expanduser("~/work/IoT_Botnet")
OUT = f"{K}/makale/figures"
MM = 1 / 25.4
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # kategorik yuvalar 1–3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "Nimbus Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": False, "grid.color": GRID,
    "grid.linewidth": 0.5, "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42,
})
KISA = {"Danmini_Doorbell": "Danmini", "Ecobee_Thermostat": "Ecobee", "Ennio_Doorbell": "Ennio",
        "Philips_B120N10_Baby_Monitor": "Philips", "Provision_PT_737E_Security_Camera": "Provision 737E",
        "Provision_PT_838_Security_Camera": "Provision 838", "Samsung_SNH_1011_N_Webcam": "Samsung",
        "SimpleHome_XCS7_1002_WHT_Security_Camera": "SimpleHome 1002",
        "SimpleHome_XCS7_1003_WHT_Security_Camera": "SimpleHome 1003"}
KUME = {"Danmini": "C1", "Philips": "C1", "Ecobee": "C2", "SimpleHome 1002": "C2", "SimpleHome 1003": "C2",
        "Provision 737E": "C3", "Provision 838": "C3"}
VERI = {"nbaiot": "N-BaIoT", "nbaiot_temiz": "N-BaIoT (no timestamp)", "botiot": "Bot-IoT", "iot23": "IoT-23",
        "cicids2017": "CICIDS2017", "cicids2017_engelen": "CICIDS2017 (Engelen)", "cic2023": "CIC-IoT2023",
        "kitsune": "Kitsune", "medbiot": "MedBIoT"}


def kaydet(fig, ad):
    fig.savefig(f"{OUT}/{ad}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{OUT}/{ad}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1: kayıt oturumları (zaman damgasından)
def fig1():
    b = pd.read_csv(f"{K}/faz6_ciktilar/tables/B3_zd_tarihleri.csv", parse_dates=["ilk", "son"])
    b["cihaz"] = b.cihaz.map(KISA)
    fig, (a, c) = plt.subplots(1, 2, figsize=(174 * MM, 62 * MM), gridspec_kw={"wspace": 0.42})
    for ax, aile, renk_f, baslik in ((a, "gafgyt", lambda d: S1, "(a) BASHLITE (gafgyt), 20 Sep 2017"),
                                     (c, "mirai", lambda d: {"C1": S1, "C2": S2, "C3": S3}[KUME[d]],
                                      "(b) Mirai, 10 Oct 2017")):
        g = b[b.sinif.str.startswith(aile)]
        cihazlar = sorted(g.cihaz.unique(), key=lambda d: (KUME.get(d, ""), d))
        for i, d in enumerate(cihazlar):
            for _, r in g[g.cihaz == d].iterrows():
                if r.son <= r.ilk:
                    continue
                ax.barh(i, (r.son - r.ilk) / pd.Timedelta(days=1), left=mdates.date2num(r.ilk), height=0.62,
                        color=renk_f(d), edgecolor="white", linewidth=0.5)
        # tür adları: ilk cihazın aralığının üstünde
        ilk = g[g.cihaz == cihazlar[0]]
        for _, r in ilk.iterrows():
            ax.text(mdates.date2num(r.ilk + (r.son - r.ilk) / 2), -0.75, r.sinif.split(":")[1],
                    ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_yticks(range(len(cihazlar)), cihazlar)
        ax.set_ylim(len(cihazlar) - 0.4, -1.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel("Decoded capture time (UTC)")
        ax.set_title(baslik, loc="left")
        ax.grid(axis="x")
    c.yaxis.set_tick_params(length=0)
    for k, r in (("C1", S1), ("C2", S2), ("C3", S3)):
        c.barh(np.nan, 0, color=r, label=f"Mirai cluster {k}")
    c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handlelength=1.0)
    kaydet(fig, "Fig1")


# ---------------------------------------------------------------- Fig. 2: protokol sızıntısı ve arındırma etkisi
def fig2():
    q = pd.read_csv(f"{K}/faz7_ciktilar/tables/QJ_protokoller.csv")
    siz = q.groupby("bolme").test_sizinti.mean()
    fig, (a, c) = plt.subplots(1, 2, figsize=(174 * MM, 58 * MM), gridspec_kw={"wspace": 0.45, "width_ratios": [1, 1.1]})
    ad = {"S1": "Random", "S3": "Temporal", "S5": "Mirai cluster out", "S6": "Block-disjoint", "S4": "Leave-one-device-out"}
    sira = ["S1", "S3", "S4", "S5", "S6"]
    y = np.arange(len(sira))
    a.barh(y, [siz[s] * 100 for s in sira], color=S1, height=0.6)
    for i, s in enumerate(sira):
        a.text(siz[s] * 100 + 1.5, i, f"{siz[s] * 100:.1f}%", va="center", fontsize=6.5, color=INK)
    a.set_yticks(y, [ad[s] for s in sira]); a.invert_yaxis(); a.set_xlim(0, 112)
    a.set_xlabel("Test rows whose content is in training (%)")
    a.set_title("(a) Content leakage by split protocol", loc="left"); a.grid(axis="x")
    t = q[q.oznitelik == "HHjit_hepsi_yok_100"].groupby(["bolme", "model"]).F1_makro_cok.mean()
    modeller = ["RF", "HistGB", "XGB", "kNN1", "kNN5", "MLP", "LR"]
    for i, s in enumerate(["S1", "S6", "S4"]):
        ham = t[s].reindex(modeller).values; ar = t[s + "f"].reindex(modeller).values
        for j in range(len(modeller)):
            c.plot([ham[j], ar[j]], [i - 0.25 + j * 0.083] * 2, color=GRID, lw=1.0, zorder=1)
        yy = i - 0.25 + np.arange(len(modeller)) * 0.083
        c.scatter(ham, yy, s=12, color=S1, marker="o", zorder=2, label="as split" if i == 0 else None)
        c.scatter(ar, yy, s=12, color=S2, marker="D", zorder=2, label="leak-filtered" if i == 0 else None)
    c.set_yticks(range(3), ["Random", "Block-\ndisjoint", "Leave-one-\ndevice-out"]); c.invert_yaxis()
    c.set_xlabel("Macro F1, timestamp-free features (7 models)")
    c.set_title("(b) Effect of removing leaked test rows", loc="left"); c.grid(axis="x")
    c.legend(loc="upper left", ncol=1, handletextpad=0.3)
    kaydet(fig, "Fig2")


# ---------------------------------------------------------------- Fig. 3: tehlike → kalan risk
def fig3():
    p = pd.read_csv(f"{K}/faz8_ciktilar/tables/faz8_profil.csv")
    rows = []
    for v, g in p.groupby("veri"):
        h = g[g.protokol == "P0"].iloc[0]
        f = g[g.protokol == "P2"] if (g.protokol == "P2").any() else g[g.protokol == "P1"]
        rows.append((VERI[v], h.tau_e_01, f.tau_e_01.iloc[0], f.protokol.iloc[0]))
    r = pd.DataFrame(rows, columns=["v", "h", "k", "p"]).sort_values("k")
    fig, ax = plt.subplots(figsize=(84 * MM, 62 * MM))
    y = np.arange(len(r))
    for i, (_, x) in enumerate(r.iterrows()):
        ax.plot([x.k, x.h], [i, i], color=GRID, lw=1.6, zorder=1)
    ax.scatter(r.h, y, s=16, color=S1, marker="o", zorder=2, label="hazard: random split (P0)")
    ax.scatter(r.k, y, s=18, color=S2, marker="D", zorder=2, label="residual: aware split (P2, or P1)")
    ax.set_yticks(y, [f"{v}{'*' if p == 'P1' else ''}" for v, p in zip(r.v, r.p)])
    ax.set_xlabel(r"Near-twin share $\tau_{\varepsilon}$ ($\varepsilon = 0.1$)")
    ax.set_xlim(0.4, 1.02); ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1, handletextpad=0.3)
    kaydet(fig, "Fig3")


# ---------------------------------------------------------------- Fig. 4: ε seçimi
def fig4():
    a = pd.read_csv(f"{K}/faz8_ciktilar/tables/faz8_eps_adaylar.csv")
    s = a[a.aile == "sabit"].copy()
    s["e"] = s.aday.str.extract(r"=\s*([0-9.e-]+)").astype(float)
    s = s.sort_values("e")
    fig, ax = plt.subplots(figsize=(84 * MM, 55 * MM))
    ax.plot(s.e, s.C2_ici, color=S1, lw=2, marker="o", ms=3.5, label="within dataset (protocol comparison)")
    ax.plot(s.e, s.C1_arasi, color=S2, lw=2, marker="D", ms=3.2, label="across datasets (dataset ranking)")
    ax.axhline(0, color=INK2, lw=0.6)
    for e, t in ((0.01, "pre-set"), (0.1, "selected")):
        ax.axvline(e, color=INK2, lw=0.6, ls=(0, (3, 2)))
        ax.text(e * 1.08, -0.47, t, fontsize=6.5, color=INK2, va="bottom")
    ax.set_xscale("log"); ax.set_ylim(-0.5, 1.0)
    ax.set_xlabel(r"$\varepsilon$ (signed-log + z-score space, per $\sqrt{d}$)")
    ax.set_ylabel(r"Spearman $\rho(\tau_{\varepsilon}$, inflation)")
    ax.grid(axis="y"); ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.17), ncol=1)
    kaydet(fig, "Fig4")


# ---------------------------------------------------------------- Fig. 5: gürültü → ε*
def fig5():
    d = pd.read_csv(f"{K}/faz8_ciktilar/tables/faz8c_bozma.csv")
    fig, ax = plt.subplots(figsize=(84 * MM, 55 * MM))
    for v, r, m in (("medbiot", S1, "o"), ("cicids2017_engelen", S2, "D"), ("nbaiot_temiz", S3, "s")):
        g = d[(d.veri == v) & (d.bozma == "gurultu")].sort_values("duzey")
        ax.errorbar(g.duzey, g.eps_yildiz, yerr=[g.eps_yildiz - g.eps_alt, g.eps_ust - g.eps_yildiz],
                    color=r, marker=m, ms=4, lw=1.6, capsize=2, label=VERI[v])
    x = np.array([8e-4, 0.07]); ax.plot(x, x, color=INK2, lw=0.7, ls=(0, (3, 2)))
    ax.text(0.05, 0.062, r"$\varepsilon^{*}=\sigma$", fontsize=6.5, color=INK2, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Added relative noise $\sigma$"); ax.set_ylabel(r"Row-level optimal $\varepsilon^{*}$")
    ax.axhline(1.2e-5, color=GRID, lw=0.6)
    ax.text(1.35e-3, 1.4e-5, r"grid floor: $\varepsilon^{*} < 10^{-5}$", fontsize=6.3, color=INK2, va="bottom")
    ax.grid(); ax.legend(loc="lower right")
    kaydet(fig, "Fig5")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5):
        f(); print(f.__name__, "tamam")
