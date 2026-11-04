# Generado desde notebooks/entrega1_recomendador_hm.ipynb (mismas celdas). Ejecutar desde la carpeta notebooks/.

display = print

# ---- celda 3 ----
import json, time, math, collections, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as Fn

warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
# Carpeta raíz del proyecto: funciona abriendo el notebook desde notebooks/ o desde la raíz
RAIZ = next(p for p in (Path(".."), Path(".")) if (p / "data" / "secuencias.parquet").exists())
DATA = RAIZ / "data"
MODELOS = RAIZ / "modelos"
MODELOS.mkdir(exist_ok=True)
KS = (5, 10, 20)

# Estilo de gráficos: marcas finas, grilla suave, sin bordes superiores/derechos
AZUL, NARANJO, AQUA, AMARILLO, MAGENTA, VERDE, VIOLETA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
GRIS = "#8a8984"
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
    "axes.edgecolor": GRIS, "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.prop_cycle": plt.cycler(color=[AZUL, NARANJO, AQUA, AMARILLO, MAGENTA]),
})
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
print(f"PyTorch {torch.__version__} · hilos de CPU: {torch.get_num_threads()}")

# ---- celda 5 ----
resumen = json.load(open(DATA / "resumen_preparacion.json"))
seq = pd.read_parquet(DATA / "secuencias.parquet")
mapa = pd.read_csv(DATA / "mapa_items.csv")
inter = pd.read_parquet(DATA / "interacciones.parquet")

N_ITEMS = int(mapa.item_idx.max())
N_USERS = len(seq)
S = [np.asarray(x, dtype=np.int64) for x in seq["items"]]          # secuencia de ítems por usuaria
F = [np.asarray(x, dtype="datetime64[D]") for x in seq["fechas"]]  # fecha de cada ítem
nombre = dict(zip(mapa.item_idx, mapa.prod_name + " (" + mapa.product_type_name + ", " + mapa.colour_group_name + ")"))

assert N_USERS >= 2000 and N_ITEMS >= 500 and len(inter) >= 50_000, "No cumple los mínimos"
pd.Series({
    "Filas originales": resumen["filas_originales"],
    "Tras filtro de fecha y muestra": resumen["tras_filtro_fecha_y_muestra"],
    "Duplicados del mismo día eliminados": resumen["duplicados_mismo_dia"],
    "Interacciones finales": resumen["interacciones"],
    "Usuarias": resumen["usuarios"],
    "Ítems (prendas)": resumen["items"],
}).to_frame("valor").style.format("{:,.0f}")

# ---- celda 14 ----
train = [s[:-2] for s in S]
hist_val, val_t = train, np.array([s[-2] for s in S])
hist_test, test_t = [s[:-1] for s in S], np.array([s[-1] for s in S])
fecha_val = np.array([f[-2] for f in F]); fecha_test = np.array([f[-1] for f in F])
mismo_carrito = fecha_val == fecha_test

print(f"Interacciones de entrenamiento: {sum(map(len, train)):,}")
print(f"Usuarias con validación y prueba el mismo día: {mismo_carrito.mean():.1%}")
print(f"Objetivos de prueba que son recompra: {np.mean([t in set(h) for h, t in zip(hist_test, test_t)]):.1%}")

# ---- celda 15 ----
def evaluar(score_fn, hists, targets, bs=2048):
    # Ranking completo con exclusión de lo ya comprado.
    # score_fn(lista_de_historiales) -> matriz (usuarias x N_ITEMS+1); la columna 0 es relleno.
    ranks = np.empty(len(hists)); top10 = []; t0 = time.perf_counter()
    for i in range(0, len(hists), bs):
        H = hists[i:i + bs]
        sc = np.asarray(score_fn(H), dtype=np.float64)
        sc[:, 0] = -np.inf
        filas = np.repeat(np.arange(len(H)), [len(h) for h in H])
        sc[filas, np.concatenate(H)] = -np.inf                        # excluir lo ya comprado
        st = sc[np.arange(len(H)), targets[i:i + bs]]
        mayores = (sc > st[:, None]).sum(1)
        empates = (sc == st[:, None]).sum(1) - 1
        r = 1 + mayores + empates / 2                                  # rango promedio en empates
        ranks[i:i + bs] = np.where(np.isinf(st), np.inf, r)            # recompra excluida = fallo
        top10.append(np.argpartition(-sc, 10, axis=1)[:, :10].copy())  # copia: no retener la matriz completa
    seg = time.perf_counter() - t0
    m = {f"HR@{k}": np.mean(ranks <= k) for k in KS}
    m |= {f"NDCG@{k}": np.mean(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0)) for k in KS}
    m["MRR"] = np.mean(1 / ranks)
    m["Cobertura@10"] = len(np.unique(np.concatenate(top10))) / N_ITEMS
    m["ms_por_1000_usuarias"] = 1000 * seg / len(hists) * 1000
    return m, ranks

def matriz_binaria(hists):
    filas = np.repeat(np.arange(len(hists)), [len(h) for h in hists])
    X = sp.csr_matrix((np.ones(len(filas), np.float32), (filas, np.concatenate(hists))),
                      shape=(len(hists), N_ITEMS + 1))
    X.sum_duplicates(); X.data[:] = 1
    return X

X_train = matriz_binaria(train)
pop = np.asarray(X_train.sum(0)).ravel().astype(np.float32)   # compras por prenda en entrenamiento
desempate = 1e-6 * pop / pop.max()                            # desempate por popularidad para todos los modelos
COLS = [f"HR@{k}" for k in KS] + [f"NDCG@{k}" for k in KS] + ["MRR", "Cobertura@10"]

# ---- celda 26 ----
class AtencionCausal(nn.Module):
    def __init__(self, d, cabezas, dropout):
        super().__init__()
        assert d % cabezas == 0
        self.h, self.dk = cabezas, d // cabezas
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.o = nn.Linear(d, d)                       # mezcla las cabezas
        self.drop = nn.Dropout(dropout)

    def forward(self, x, relleno):
        B, L, D = x.shape
        dividir = lambda t: t.view(B, L, self.h, self.dk).transpose(1, 2)        # (B, cabezas, L, dk)
        q, k, v = dividir(self.q(x)), dividir(self.k(x)), dividir(self.v(x))
        a = q @ k.transpose(-1, -2) / math.sqrt(self.dk)                         # (B, cabezas, L, L)
        futuro = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        a = a.masked_fill(futuro, float("-inf"))                                 # máscara causal triangular
        a = a.masked_fill(relleno[:, None, None, :], float("-inf"))              # no mirar el relleno
        pesos = torch.softmax(a, dim=-1).nan_to_num(0.0)                         # filas de relleno quedan en 0
        salida = (self.drop(pesos) @ v).transpose(1, 2).reshape(B, L, D)
        return self.o(salida), pesos

# ---- celda 28 ----
class BloqueTransformer(nn.Module):
    def __init__(self, d, cabezas, dropout):
        super().__init__()
        self.norm1, self.atencion = nn.LayerNorm(d), AtencionCausal(d, cabezas, dropout)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, relleno):
        a, pesos = self.atencion(self.norm1(x), relleno)
        x = x + self.drop(a)                        # residual 1
        x = x + self.drop(self.ff(self.norm2(x)))   # residual 2
        return x, pesos

# ---- celda 30 ----
class SASRec(nn.Module):
    def __init__(self, n_items, d=64, bloques=2, cabezas=1, dropout=0.2, maxlen=50):
        super().__init__()
        self.d, self.maxlen = d, maxlen
        self.item = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos = nn.Embedding(maxlen, d)
        self.drop = nn.Dropout(dropout)
        self.bloques = nn.ModuleList([BloqueTransformer(d, cabezas, dropout) for _ in range(bloques)])
        self.norm = nn.LayerNorm(d)
        nn.init.normal_(self.item.weight, std=0.02); nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad(): self.item.weight[0].zero_()

    def forward(self, s):                                          # s: (B, L) índices de prenda, 0 = relleno
        relleno = s == 0
        x = self.item(s) * math.sqrt(self.d) + self.pos(torch.arange(s.shape[1], device=s.device))[None]
        x = self.drop(x) * (~relleno)[..., None]
        pesos = []
        for bloque in self.bloques:
            x, w = bloque(x, relleno)
            x = x * (~relleno)[..., None]
            pesos.append(w)
        return self.norm(x), pesos                                  # (B, L, d) y pesos de atención por bloque

    def puntajes(self, s):                                          # puntaje de todas las prendas para la última posición
        h, _ = self.forward(s)
        return h[:, -1] @ self.item.weight.T

MAXLEN = 50   # cubre al 90 % de las historias completas (p90 = 31)

def lote(seqs, maxlen=MAXLEN):
    X = np.zeros((len(seqs), maxlen), np.int64)
    for i, s in enumerate(seqs):
        s = s[-maxlen:]; X[i, maxlen - len(s):] = s               # relleno a la izquierda
    return torch.from_numpy(X)

# ---- celda 32 ----
def epoca(modelo, opt, perdida, batch):
    modelo.train()
    idx = np.random.permutation(len(train)); total, n = 0.0, 0
    for i in range(0, len(idx), batch):
        b = [train[j] for j in idx[i:i + batch]]
        X, Y = lote([s[:-1] for s in b]), lote([s[1:] for s in b])    # entrada y siguiente ítem (desplazado en 1)
        h, _ = modelo(X)
        m = Y > 0; hh, y = h[m], Y[m]                                  # solo posiciones reales
        if perdida == "ce":
            logits = hh @ modelo.item.weight.T
            logits[:, 0] = float("-inf")                               # el relleno nunca es una respuesta
            loss = Fn.cross_entropy(logits, y)
        else:
            neg = torch.randint(1, N_ITEMS + 1, y.shape)
            pos_l, neg_l = (hh * modelo.item(y)).sum(-1), (hh * modelo.item(neg)).sum(-1)
            loss = (Fn.binary_cross_entropy_with_logits(pos_l, torch.ones_like(pos_l)) +
                    Fn.binary_cross_entropy_with_logits(neg_l, torch.zeros_like(neg_l)))
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(y); n += len(y)
    return total / n

@torch.no_grad()
def validar(modelo, bs=4096):
    # HR@10 y NDCG@10 en validación (ranking completo, excluye lo comprado) para la parada temprana
    modelo.eval(); hr = nd = 0.0
    for i in range(0, len(hist_val), bs):
        H = hist_val[i:i + bs]
        sc = modelo.puntajes(lote(H)); sc[:, 0] = float("-inf")
        filas = torch.from_numpy(np.repeat(np.arange(len(H)), [len(x) for x in H]))
        sc[filas, torch.from_numpy(np.concatenate(H))] = float("-inf")
        t = torch.from_numpy(val_t[i:i + bs])
        st = sc[torch.arange(len(H)), t]
        r = 1 + (sc > st[:, None]).sum(1).float()
        r = torch.where(torch.isinf(st), torch.full_like(r, float("inf")), r)
        hr += (r <= 10).sum().item(); nd += torch.where(r <= 10, 1 / torch.log2(r + 1), torch.zeros_like(r)).sum().item()
    return hr / len(hist_val), nd / len(hist_val)

RUTA_HIST = MODELOS / "historial_sasrec.json"
HIST = json.load(open(RUTA_HIST)) if RUTA_HIST.exists() else {}

def entrenar_sasrec(clave, cfg, max_epocas=30, paciencia=3):
    ruta = MODELOS / f"sasrec_{clave}.pt"
    arq = {k: cfg[k] for k in ("d", "bloques", "cabezas", "dropout")}
    if HIST.get(clave, {}).get("terminado") and not ruta.exists():  # entrenada, pero solo se guardó la elegida
        print(f"[{clave}] entrenada antes; se conserva solo su registro (el modelo guardado es el de la configuración elegida):")
        for linea in HIST[clave]["log"]: print("   ", linea)
        return None
    if ruta.exists() and HIST.get(clave, {}).get("terminado"):       # ya entrenado: se carga (entrenar toma ~1 h en CPU)
        print(f"[{clave}] modelo ya entrenado, se carga de {ruta.name}. Registro del entrenamiento:")
        for linea in HIST[clave]["log"]: print("   ", linea)
        modelo = SASRec(N_ITEMS, maxlen=MAXLEN, **arq)
        modelo.load_state_dict(torch.load(ruta, map_location="cpu")["estado"])
        return modelo.eval()
    torch.manual_seed(SEED); np.random.seed(SEED)
    modelo = SASRec(N_ITEMS, maxlen=MAXLEN, **arq)
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"])
    curvas = {"epoca": [], "perdida": [], "hr10_val": [], "ndcg10_val": [], "segundos": []}
    mejor, mejor_ep, sin_mejora, log = -1.0, 0, 0, []
    for ep in range(1, max_epocas + 1):
        t0 = time.perf_counter()
        perdida = epoca(modelo, opt, cfg["perdida"], cfg["batch"])
        hr, nd = validar(modelo)
        seg = time.perf_counter() - t0
        for k_, v_ in zip(curvas, (ep, perdida, hr, nd, seg)): curvas[k_].append(v_)
        if nd > mejor:
            mejor, mejor_ep, sin_mejora = nd, ep, 0
            torch.save({"estado": modelo.state_dict(), "cfg": cfg, "n_items": N_ITEMS, "maxlen": MAXLEN}, ruta)
        else:
            sin_mejora += 1
        linea = f"época {ep:2d} | pérdida {perdida:.4f} | val HR@10 {hr:.4f} NDCG@10 {nd:.4f} | {seg:.0f} s"
        print(f"[{clave}] {linea}", flush=True); log.append(linea)
        HIST[clave] = {"cfg": cfg, "curvas": curvas, "log": log, "mejor_epoca": mejor_ep, "terminado": False,
                       "segundos": float(sum(curvas["segundos"]))}
        json.dump(HIST, open(RUTA_HIST, "w"), indent=1)
        if sin_mejora >= paciencia:
            log.append(f"parada temprana: {paciencia} épocas sin mejorar; se usa la época {mejor_ep}")
            print(f"[{clave}] {log[-1]}"); break
    HIST[clave]["terminado"] = True; HIST[clave]["log"] = log
    json.dump(HIST, open(RUTA_HIST, "w"), indent=1)
    modelo.load_state_dict(torch.load(ruta, map_location="cpu")["estado"])
    return modelo.eval()

# ---- celda 34 ----
CONFIGS = {
    "A": dict(d=64,  bloques=1, cabezas=1, dropout=0.2, perdida="ce",  lr=2e-3, batch=256),
    "B": dict(d=64,  bloques=2, cabezas=1, dropout=0.2, perdida="ce",  lr=2e-3, batch=256),
    "C": dict(d=128, bloques=2, cabezas=2, dropout=0.3, perdida="ce",  lr=2e-3, batch=256),
    "D": dict(d=64,  bloques=2, cabezas=1, dropout=0.2, perdida="bce", lr=2e-3, batch=256),
}
modelos_sasrec = {clave: entrenar_sasrec(clave, cfg) for clave, cfg in CONFIGS.items()}
