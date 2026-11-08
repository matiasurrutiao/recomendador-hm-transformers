"""
Precalcula, solo con los datos de ENTRENAMIENTO (cada secuencia sin sus 2 últimos ítems), las estructuras
que usa el dashboard, con los mismos hiperparámetros elegidos en validación en el notebook 1:

  modelos/itemknn_vecinos.npz   similitud coseno ítem–ítem, 100 vecinos por prenda (λ = 0,8 se aplica al recomendar)
  modelos/reglas.parquet        reglas del Certamen 2: FP-Growth, max_len 3, min_conteo 5, confianza ≥ 0,1, 1 consecuente

Uso (desde la raíz del repositorio):  python scripts/precalcular_app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

RAIZ = Path(__file__).resolve().parents[1]
MIN_CONTEO, MIN_CONF, TOPN = 5, 0.1, 100


def main():
    seq = pd.read_parquet(RAIZ / "data" / "secuencias.parquet")
    n_items = int(pd.read_csv(RAIZ / "data" / "mapa_items.csv").item_idx.max())
    train = [np.asarray(s, dtype=np.int64)[:-2] for s in seq["items"]]

    filas = np.repeat(np.arange(len(train)), [len(h) for h in train])
    X = sp.csr_matrix((np.ones(len(filas), np.float32), (filas, np.concatenate(train))), shape=(len(train), n_items + 1))
    X.sum_duplicates(); X.data[:] = 1
    pop = np.asarray(X.sum(0)).ravel().astype(np.float32)

    # Ítem–ítem coseno: 100 vecinos por prenda
    CO = (X.T @ X).tocsr(); CO.setdiag(0); CO.eliminate_zeros()
    c = CO.tocoo()
    M = sp.csr_matrix(((c.data / np.sqrt(pop[c.row] * pop[c.col])).astype(np.float32), (c.row, c.col)), shape=CO.shape)
    fs, cs, vs = [], [], []
    for i in range(M.shape[0]):
        a, b = M.indptr[i], M.indptr[i + 1]
        if a == b: continue
        v, cc = M.data[a:b], M.indices[a:b]
        if b - a > TOPN:
            k = np.argpartition(-v, TOPN)[:TOPN]; v, cc = v[k], cc[k]
        fs.append(np.full(len(cc), i)); cs.append(cc); vs.append(v)
    SIM = sp.csr_matrix((np.concatenate(vs), (np.concatenate(fs), np.concatenate(cs))), shape=M.shape)
    sp.save_npz(RAIZ / "modelos" / "itemknn_vecinos.npz", SIM)
    print(f"ítem–ítem: {SIM.nnz:,} similitudes")

    # Reglas de asociación (método del Certamen 2)
    from mlxtend.frequent_patterns import association_rules, fpgrowth
    Xi = X[:, 1:]; vivas = np.asarray(Xi.sum(0)).ravel() > 0
    canastas = pd.DataFrame.sparse.from_spmatrix(Xi[:, vivas].astype(bool),
                                                 columns=[str(i) for i in np.arange(1, n_items + 1)[vivas]])
    frec = fpgrowth(canastas, min_support=MIN_CONTEO / len(canastas), use_colnames=True, max_len=3)
    R = association_rules(frec, metric="confidence", min_threshold=MIN_CONF)
    R = R[R["consequents"].apply(len) == 1].copy()
    out = pd.DataFrame({
        "antecedente_1": R["antecedents"].apply(lambda a: sorted(int(x) for x in a)[0]).values,
        "antecedente_2": R["antecedents"].apply(lambda a: (sorted(int(x) for x in a) + [0])[1]).values,
        "producto": R["consequents"].apply(lambda z: int(next(iter(z)))).values,
        "soporte": R["support"].values, "confianza": R["confidence"].values, "lift": R["lift"].values,
    })
    out.to_parquet(RAIZ / "modelos" / "reglas.parquet", index=False)
    print(f"reglas: {len(out):,} ({(out.antecedente_2 > 0).sum():,} con antecedente de 2 prendas)")


if __name__ == "__main__":
    main()
