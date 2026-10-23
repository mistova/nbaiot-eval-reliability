"""Faz 7 GPU modelleri (torch + XGBoost cuda): MLP, softmax regresyon (LR), kNN; ve ortak ön işleme.
Girdi her zaman işaretli log1p + eğitim istatistikleriyle standartlaştırma (zaman damgası ve 1e21 büyüklükleri için)."""
import numpy as np
import torch

CIHAZ = "cuda"


class Onisleme:
    def fit(self, X):
        Z = np.sign(X) * np.log1p(np.abs(X))
        self.m, self.s = Z.mean(0), Z.std(0) + 1e-6
        return self

    def __call__(self, X):
        return ((np.sign(X) * np.log1p(np.abs(X)) - self.m) / self.s).astype(np.float32)


def _tensor(a):
    return torch.as_tensor(a, device=CIHAZ)


def sinir_agi(Xtr, ytr, n_sinif, gizli=(256, 128), epok=15, seed=0, lr=1e-3, bs=4096):
    """gizli=() → çok terimli lojistik regresyon. Döndürür: tahmin fonksiyonu (sınıf indeksleri)."""
    torch.manual_seed(seed)
    katman, d = [], Xtr.shape[1]
    for h in gizli:
        katman += [torch.nn.Linear(d, h), torch.nn.ReLU()]
        d = h
    katman.append(torch.nn.Linear(d, n_sinif))
    net = torch.nn.Sequential(*katman).to(CIHAZ)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    X, y = _tensor(Xtr), _tensor(ytr.astype(np.int64))
    g = torch.Generator(device=CIHAZ).manual_seed(seed)
    for _ in range(epok):
        p = torch.randperm(len(X), device=CIHAZ, generator=g)
        for i in range(0, len(X), bs):
            j = p[i:i + bs]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(X[j]), y[j]).backward()
            opt.step()
    net.eval()

    def tahmin(Xte):
        out = []
        with torch.no_grad():
            for i in range(0, len(Xte), 65536):
                out.append(net(_tensor(Xte[i:i + 65536])).argmax(1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, int)
    return tahmin


def en_yakin(Xtr, Xte, k=5, parca_te=4096, parca_tr=400_000):
    """Her test satırı için en yakın k eğitim satırının (öklid uzaklığı, indeks). Tamamen GPU'da, parçalı."""
    Tr = _tensor(Xtr)
    trn = (Tr * Tr).sum(1)
    D_all, I_all = [], []
    for i in range(0, len(Xte), parca_te):
        Q = _tensor(Xte[i:i + parca_te]); qn = (Q * Q).sum(1, keepdim=True)
        best_d = torch.full((len(Q), k), float("inf"), device=CIHAZ)
        best_i = torch.zeros((len(Q), k), dtype=torch.long, device=CIHAZ)
        for j in range(0, len(Tr), parca_tr):
            d = (qn + trn[j:j + parca_tr][None] - 2 * Q @ Tr[j:j + parca_tr].T).clamp_min_(0)
            kk = min(k, d.shape[1])
            dv, di = d.topk(kk, dim=1, largest=False)
            cd, ci = torch.cat([best_d, dv], 1), torch.cat([best_i, di + j], 1)
            o = cd.topk(k, dim=1, largest=False).indices
            best_d, best_i = cd.gather(1, o), ci.gather(1, o)
        D_all.append(best_d.sqrt().cpu().numpy()); I_all.append(best_i.cpu().numpy())
    return np.concatenate(D_all), np.concatenate(I_all)


def knn_tahmin(I, ytr, k):
    """Çoğunluk oyu (eşitlikte en yakın komşu)."""
    lab = ytr[I[:, :k]]
    if k == 1:
        return lab[:, 0]
    out = np.empty(len(lab), dtype=lab.dtype)
    for n, row in enumerate(lab):
        v, c = np.unique(row, return_counts=True)
        out[n] = row[0] if (c == c.max()).sum() > 1 and row[0] in v[c == c.max()] else v[c.argmax()]
    return out


def xgb_gpu(Xtr, ytr, seed, n_sinif):
    from xgboost import XGBClassifier
    m = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, tree_method="hist", device="cuda",
                      random_state=seed, objective="multi:softmax" if n_sinif > 2 else "binary:logistic")
    m.fit(Xtr, ytr)
    return lambda X: m.predict(X)
