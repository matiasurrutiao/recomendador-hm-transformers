"""Protocolo de evaluación compartido con el notebook 1: leave-last-out, ranking completo, exclusión de lo ya comprado.
Aquí se evalúa a partir de la lista FINAL que ve la clienta (incluido el re-ranking de diversidad) y se agregan
métricas de diversidad y novedad para la innovación."""
import numpy as np
import pandas as pd

KS = (5, 10, 20)


def evaluar_listas(rec, modelo, hists, targets, tareas=None, diversidad=None, candidatos=50, n=100, bs=512):
    """Devuelve (métricas, rangos, top10). El rango del ítem real se busca en la lista final de `n`;
    si no aparece, se usa su posición en el orden completo del modelo (solo afecta al MRR, en < 0,01)."""
    rangos, top10 = np.empty(len(hists)), []
    for i in range(0, len(hists), bs):
        H = hists[i:i + bs]
        T = None if tareas is None else tareas[i:i + bs]
        sc = rec.puntajes(modelo, H, tareas=T)
        for j, h in enumerate(H):
            lista = rec.ordenar(sc[j], h, n=n, max_por_producto=diversidad, candidatos=candidatos)
            t = targets[i + j]
            pos = np.where(lista == t)[0]
            if len(pos):
                rangos[i + j] = pos[0] + 1
            elif t in set(h.tolist()):
                rangos[i + j] = np.inf                   # recompra: excluida para todos
            else:
                fila = np.asarray(sc[j], dtype=np.float64).copy(); fila[0] = -np.inf; fila[h] = -np.inf
                rangos[i + j] = 1 + (fila > fila[t]).sum()
            top10.append(lista[:10].copy())
    top10 = np.array(top10)
    m = {f"HR@{k}": np.mean(rangos <= k) for k in KS}
    m |= {f"NDCG@{k}": np.mean(np.where(rangos <= k, 1 / np.log2(rangos + 1), 0)) for k in KS}
    m["MRR"] = np.mean(1 / rangos)
    m |= diversidad_listas(rec, top10)
    return m, rangos, top10


def diversidad_listas(rec, top10):
    prod = rec.producto[top10]
    tipos = rec.mapa.product_type_name.reindex(top10.ravel()).to_numpy(dtype=object).reshape(top10.shape)
    n_usuarias = len(rec.train)
    return {
        "Cobertura@10": len(np.unique(top10)) / rec.n_items,
        "Listas con variantes repetidas": np.mean([len(set(p)) < len(p) for p in prod]),
        "Prendas distintas@10": np.mean([len(set(p)) for p in prod]),
        "Tipos distintos@10": np.mean([len(set(t)) for t in tipos]),
        "Novedad@10 (bits)": float(np.mean(-np.log2(np.maximum(rec.pop[top10], 1) / n_usuarias))),
    }


def tareas_de(fechas_hist_ultima, fechas_objetivo):
    """'carrito' si el ítem a predecir es del mismo día que la última compra del historial (sesión abierta)."""
    return np.where(np.asarray(fechas_hist_ultima) == np.asarray(fechas_objetivo), "carrito", "visita")
