"""
Recomendadores del proyecto, usados igual por el dashboard (src/app.py) y por el notebook 2.

Todo se ajusta SOLO con entrenamiento (cada secuencia sin sus 2 últimos ítems), con los hiperparámetros
elegidos en validación en el notebook 1. Ningún recomendador incluye prendas que la clienta ya compró.

Modelos:
  - SASRec (Transformer, modelos/sasrec.pt)
  - Ítem–ítem coseno con peso λ^(L−j) a las últimas compras (λ = 0,8)
  - Reglas de asociación del Certamen 2 (orden por lift, relleno con más vendidos)
  - Popularidad reciente (últimos 7 días de entrenamiento)
  - Híbrido SASRec + ítem–ítem con diversidad (innovación del grupo): ver `puntajes_hibrido` y `ordenar`
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

try:
    from .modelo_sasrec import MAXLEN, cargar_sasrec, lote
except ImportError:                      # cuando se importa como módulo suelto (streamlit run src/app.py)
    from modelo_sasrec import MAXLEN, cargar_sasrec, lote

LAMBDA_KNN = 0.8          # elegido en validación (notebook 1, sección 5.3)
DIAS_RECIENTE = 7         # ventana de "populares ahora" para el catálogo y el arranque en frío

# Innovación: configuración elegida en VALIDACIÓN (notebook 2, sección 2). El enrutamiento por tarea
# (carrito → ítem–ítem, próxima visita → SASRec) se probó y no superó a la fusión pareja, así que ambas tareas
# usan la misma fusión por rangos recíprocos con peso 1 para cada modelo.
HIBRIDO = {
    "carrito": "sasrec",       # modelo principal cuando la clienta sigue comprando el mismo día
    "visita": "sasrec",        # modelo principal cuando vuelve otro día
    "peso_secundario": 1.0,    # peso del otro modelo en la fusión (1 = fusión pareja SASRec + ítem–ítem)
    "max_por_producto": 1,     # diversidad: a lo sumo una variante de color por prenda en la lista
    "candidatos": 50,          # la diversidad re-ordena dentro de los 50 mejores de la fusión
}

NOMBRES_MODELO = {
    "sasrec": "SASRec (Transformer)",
    "hibrido": "Híbrido SASRec + ítem–ítem con diversidad",
    "knn": "Ítem–ítem coseno",
    "reglas": "Reglas de asociación",
    "pop": "Popularidad reciente",
}


class Recomendador:
    def __init__(self, raiz):
        raiz = Path(raiz)
        self.mapa = pd.read_csv(raiz / "data" / "mapa_items.csv").set_index("item_idx").sort_index()
        self.n_items = int(self.mapa.index.max())
        seq = pd.read_parquet(raiz / "data" / "secuencias.parquet")
        self.S = [np.asarray(x, dtype=np.int64) for x in seq["items"]]
        self.F = [np.asarray(x, dtype="datetime64[D]") for x in seq["fechas"]]
        self.train = [s[:-2] for s in self.S]
        self.producto = np.zeros(self.n_items + 1, dtype=np.int64)
        self.producto[self.mapa.index.values] = self.mapa.product_code.values

        # Popularidad (solo entrenamiento)
        X = self._binaria(self.train)
        self.pop = np.asarray(X.sum(0)).ravel().astype(np.float32)
        fechas_train = np.concatenate([f[:-2] for f in self.F])
        items_train = np.concatenate(self.train)
        self.fecha_ref = fechas_train.max()
        recientes = items_train[fechas_train > self.fecha_ref - np.timedelta64(DIAS_RECIENTE, "D")]
        self.pop_reciente = np.bincount(recientes, minlength=self.n_items + 1).astype(np.float32)
        self.desempate = 1e-6 * self.pop / self.pop.max()
        self.rank_pop = np.empty(self.n_items + 1, dtype=np.int64)       # 1 = la prenda más vendida
        self.rank_pop[np.argsort(-self.pop)] = np.arange(1, self.n_items + 2)

        # Ítem–ítem y reglas (precalculados por scripts/precalcular_app.py)
        self.SIM = sp.load_npz(raiz / "modelos" / "itemknn_vecinos.npz").tocsr()
        self.reglas = pd.read_parquet(raiz / "modelos" / "reglas.parquet")
        r1 = self.reglas[self.reglas.antecedente_2 == 0]
        r2 = self.reglas[self.reglas.antecedente_2 > 0]
        val1 = (r1.lift + 1e-6 * r1.confianza).values
        self.L1 = sp.csr_matrix((val1, (r1.antecedente_1.values, r1.producto.values)),
                                shape=(self.n_items + 1, self.n_items + 1))
        self.a2 = r2[["antecedente_1", "antecedente_2"]].values.astype(np.int64)
        self.c2 = r2.producto.values.astype(np.int64)
        self.v2 = (r2.lift + 1e-6 * r2.confianza).values

        # SASRec
        self.sasrec, self.cfg_sasrec = cargar_sasrec(raiz / "modelos" / "sasrec.pt", self.n_items)
        self.parametros_sasrec = sum(p.numel() for p in self.sasrec.parameters())

    # ------------------------------------------------------------------ utilidades
    def _binaria(self, hists):
        filas = np.repeat(np.arange(len(hists)), [len(h) for h in hists])
        cols = np.concatenate(hists) if len(filas) else np.zeros(0, np.int64)
        X = sp.csr_matrix((np.ones(len(filas), np.float32), (filas, cols)), shape=(len(hists), self.n_items + 1))
        X.sum_duplicates(); X.data[:] = 1
        return X

    # ------------------------------------------------------------------ puntajes en lote (usuarias × prendas)
    @torch.no_grad()
    def puntajes_sasrec(self, H):
        sc = self.sasrec.puntajes(lote(H)).numpy() + self.desempate
        vacias = np.array([len(h) == 0 for h in H])
        if vacias.any():                                   # arranque en frío: sin historial, SASRec no tiene qué mirar
            sc[vacias] = self.pop_reciente + self.desempate
        return sc

    def puntajes_knn(self, H, lam=LAMBDA_KNN):
        filas = np.repeat(np.arange(len(H)), [len(h) for h in H])
        if len(filas) == 0:
            return np.tile(self.pop_reciente + self.desempate, (len(H), 1))
        w = np.concatenate([lam ** np.arange(len(h) - 1, -1, -1) for h in H]).astype(np.float32)
        W = sp.csr_matrix((w, (filas, np.concatenate(H))), shape=(len(H), self.n_items + 1))
        return (W @ self.SIM).toarray() + self.desempate

    def puntajes_reglas(self, H):
        out = np.zeros((len(H), self.n_items + 1))
        for i, h in enumerate(H):
            if len(h) == 0: continue
            sub = self.L1[np.unique(h)]
            if sub.nnz: out[i] = sub.max(axis=0).toarray().ravel()
        if len(self.a2) and any(len(h) for h in H):
            Xh = self._binaria(H).tocsc()
            act = Xh[:, self.a2[:, 0]].multiply(Xh[:, self.a2[:, 1]]).tocoo()
            np.maximum.at(out, (act.row, self.c2[act.col]), self.v2[act.col])
        return out + 1e-3 * self.pop / self.pop.max()      # sin regla activa → más vendidos

    def puntajes_pop(self, n_filas):
        return np.tile(self.pop_reciente + self.desempate, (n_filas, 1))

    def puntajes(self, modelo, H, tareas=None):
        if modelo == "sasrec": return self.puntajes_sasrec(H)
        if modelo == "knn": return self.puntajes_knn(H)
        if modelo == "reglas": return self.puntajes_reglas(H)
        if modelo == "pop": return self.puntajes_pop(len(H))
        if modelo == "hibrido": return self.puntajes_hibrido(H, tareas)
        raise ValueError(modelo)

    # ------------------------------------------------------------------ innovación: híbrido por tarea
    @staticmethod
    def _rangos_inversos(sc, k=60, profundidad=200):
        """Fusión por rangos recíprocos: 1 / (k + rango) para las `profundidad` mejores de cada fila, 0 el resto."""
        out = np.zeros(sc.shape, dtype=np.float32)
        top = np.argpartition(-sc, profundidad, axis=1)[:, :profundidad]
        orden = np.argsort(-np.take_along_axis(sc, top, 1), axis=1)
        top = np.take_along_axis(top, orden, 1)
        np.put_along_axis(out, top, (1.0 / (k + np.arange(1, profundidad + 1, dtype=np.float32)))[None, :], axis=1)
        return out

    def puntajes_hibrido(self, H, tareas, cfg=None):
        """tareas[i] = 'carrito' si la clienta sigue comprando el mismo día de su última compra, 'visita' si vuelve otro día.
        Permite enrutar por tarea (modelo principal + el otro con peso_secundario). La configuración elegida en validación
        (HIBRIDO) usa SASRec en ambas tareas con peso_secundario = 1: es decir, fusión pareja SASRec + ítem–ítem para todas.
        El enrutamiento por tarea (carrito → ítem–ítem) se probó y se descartó (notebook 2, sección 2)."""
        cfg = cfg or HIBRIDO
        tareas = np.asarray(tareas)
        sc_s, sc_k = self.puntajes_sasrec(H), self.puntajes_knn(H)
        for sc in (sc_s, sc_k):                              # lo ya comprado no compite en la fusión
            filas = np.repeat(np.arange(len(H)), [len(h) for h in H])
            if len(filas): sc[filas, np.concatenate(H)] = -np.inf
        r_s = self._rangos_inversos(sc_s); del sc_s
        r_k = self._rangos_inversos(sc_k); del sc_k
        w = np.float32(cfg["peso_secundario"])
        principal_es_sasrec = np.where(tareas == "carrito", cfg["carrito"] == "sasrec", cfg["visita"] == "sasrec")
        fus = r_k                                            # se reutiliza la memoria
        fus[principal_es_sasrec] = r_s[principal_es_sasrec] + w * r_k[principal_es_sasrec]
        fus[~principal_es_sasrec] = r_k[~principal_es_sasrec] + w * r_s[~principal_es_sasrec]
        fus += np.float32(1e-9) * self.desempate.astype(np.float32)   # desempate final por popularidad
        return fus

    def ordenar(self, sc, hist, n=10, max_por_producto=None, candidatos=50):
        """Top-n de una fila de puntajes, sin lo ya comprado. Si max_por_producto, re-ordena con diversidad:
        recorre los `candidatos` mejores y salta las variantes de color de una prenda que ya tiene ese cupo."""
        sc = np.asarray(sc, dtype=np.float64).copy()
        sc[0] = -np.inf
        if len(hist): sc[np.asarray(hist)] = -np.inf
        orden = np.argsort(-sc)
        if not max_por_producto:
            return orden[:n].copy()                        # copia: una vista retendría el orden completo en memoria
        elegidos, cuenta = [], {}
        for it in orden[:max(candidatos, n)]:
            p = self.producto[it]
            if cuenta.get(p, 0) < max_por_producto:
                elegidos.append(it); cuenta[p] = cuenta.get(p, 0) + 1
            if len(elegidos) == n: break
        for it in orden:                                    # si faltan, se completa en orden original
            if len(elegidos) == n: break
            if it not in elegidos: elegidos.append(it)
        return np.array(elegidos)

    # ------------------------------------------------------------------ una clienta (dashboard y analista)
    def recomendar(self, hist, modelo, n=10, tarea="visita"):
        hist = np.asarray(hist, dtype=np.int64)
        sc = self.puntajes(modelo, [hist], tareas=[tarea])[0]
        diversidad = HIBRIDO["max_por_producto"] if modelo == "hibrido" else None
        top = self.ordenar(sc, hist, n, max_por_producto=diversidad, candidatos=HIBRIDO["candidatos"])
        apoyo = {}
        if modelo == "hibrido" and len(hist):                 # puesto que le dio cada modelo a cada prenda
            for m in ("sasrec", "knn"):
                orden = self.ordenar(self.puntajes(m, [hist])[0], hist, n=500)
                apoyo[m] = {int(x): k for k, x in enumerate(orden, 1)}
        if modelo == "sasrec" and len(hist):
            apoyo["att"] = self.atencion(hist)
        filas = []
        for pos, it in enumerate(top, 1):
            filas.append({"posicion": pos, "item_idx": int(it), **self.ficha(it),
                          "puntaje": float(sc[it]), "por_que": self.por_que(hist, int(it), modelo, tarea, apoyo)})
        return pd.DataFrame(filas)

    def ficha(self, it):
        r = self.mapa.loc[int(it)]
        return {"prenda": r.prod_name, "tipo": r.product_type_name, "grupo": r.product_group_name,
                "color": r.colour_group_name, "linea": r.index_group_name, "precio_rel": round(float(r.precio_mediano), 4),
                "codigo_producto": int(r.product_code), "ranking_popularidad": int(self.rank_pop[int(it)])}

    @torch.no_grad()
    def atencion(self, hist):
        """Pesos α del último bloque (promedio de cabezas; el modelo elegido tiene 1) en la fila de la última posición."""
        hist = np.asarray(hist, dtype=np.int64)
        if len(hist) == 0: return np.zeros(0)
        _, pesos = self.sasrec(lote([hist]))
        w = pesos[-1][0].mean(0)[-1].numpy()
        n = min(len(hist), MAXLEN)
        out = np.zeros(len(hist)); out[-n:] = w[-n:]
        return out

    def por_que(self, hist, it, modelo, tarea="visita", apoyo=None):
        apoyo = apoyo or {}
        if len(hist) == 0:
            return "sin historial: de las más vendidas de la última semana"
        if modelo == "hibrido":
            p = lambda m: f"n.º {apoyo[m][it]}" if it in apoyo.get(m, {}) else "fuera de su top 500"
            return f"SASRec la pone {p('sasrec')} · ítem–ítem la pone {p('knn')}"
        if modelo == "knn":
            w = LAMBDA_KNN ** np.arange(len(hist) - 1, -1, -1)
            aporte = np.array([w[j] * self.SIM[h, it] for j, h in enumerate(hist)])
            j = int(np.argmax(aporte))
            return f"parecida a «{self.mapa.loc[int(hist[j]), 'prod_name']}» (compra n.º {j + 1}, similitud {self.SIM[hist[j], it]:.2f})"
        if modelo == "reglas":
            hs = set(int(x) for x in hist)
            r = self.reglas[(self.reglas.producto == it) & self.reglas.antecedente_1.isin(hs) &
                            ((self.reglas.antecedente_2 == 0) | self.reglas.antecedente_2.isin(hs))]
            if len(r) == 0:
                return "sin regla activa: relleno con las más vendidas"
            r = r.sort_values(["lift", "confianza"], ascending=False).iloc[0]
            ant = " + ".join(f"«{self.mapa.loc[int(a), 'prod_name']}»" for a in (r.antecedente_1, r.antecedente_2) if a > 0)
            return f"regla {ant} ⇒ esta prenda (lift {r.lift:,.0f}, confianza {r.confianza:.0%})"
        if modelo == "sasrec":
            w = apoyo["att"] if "att" in apoyo else self.atencion(hist)
            j = int(np.argmax(w))
            E = self.sasrec.item.weight.detach().numpy()
            h_uni = np.unique(hist)
            cos = (E[h_uni] @ E[it]) / (np.linalg.norm(E[h_uni], axis=1) * np.linalg.norm(E[it]) + 1e-9)
            k = int(h_uni[np.argmax(cos)])
            return (f"más atención en «{self.mapa.loc[int(hist[j]), 'prod_name']}» (H{j + 1}, {w[j]:.0%}); "
                    f"compra más parecida en el modelo: «{self.mapa.loc[k, 'prod_name']}» (coseno {cos.max():.2f})")
        if modelo == "pop":
            return f"n.º {int(np.argsort(-self.pop_reciente).tolist().index(it)) + 1} en ventas de la última semana"
        return ""

    def tarea_de_historial(self, fechas, fecha_consulta):
        """'carrito' si la consulta es el mismo día de la última compra; 'visita' si es otro día."""
        if len(fechas) == 0: return "visita"
        return "carrito" if np.datetime64(fechas[-1], "D") == np.datetime64(fecha_consulta, "D") else "visita"
