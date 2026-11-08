"""Arquitectura de SASRec: el mismo código del notebook 1 (sección 6), sin cambios, para cargar modelos/sasrec.pt."""
import math

import numpy as np
import torch
import torch.nn as nn

MAXLEN = 50   # largo máximo del historial que ve el modelo (cubre al 90 % de las historias)


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


class SASRec(nn.Module):
    def __init__(self, n_items, d=64, bloques=2, cabezas=1, dropout=0.2, maxlen=MAXLEN):
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


def lote(seqs, maxlen=MAXLEN):
    X = np.zeros((len(seqs), maxlen), np.int64)
    for i, s in enumerate(seqs):
        s = np.asarray(s, dtype=np.int64)[-maxlen:]
        if len(s): X[i, maxlen - len(s):] = s                     # relleno a la izquierda
    return torch.from_numpy(X)


def cargar_sasrec(ruta, n_items):
    """Carga el SASRec elegido en validación (CPU, modo evaluación)."""
    ck = torch.load(ruta, map_location="cpu")
    cfg = ck["cfg"]
    modelo = SASRec(n_items, d=cfg["d"], bloques=cfg["bloques"], cabezas=cfg["cabezas"],
                    dropout=cfg["dropout"], maxlen=ck.get("maxlen", MAXLEN))
    modelo.load_state_dict(ck["estado"])
    return modelo.eval(), cfg
