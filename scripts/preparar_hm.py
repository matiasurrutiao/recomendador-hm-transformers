"""
Preparación de la base H&M para la Entrega 1 (SASRec + líneas base).

Lee los CSV originales de Kaggle (data/raw/), filtra un período, toma una
muestra fija de clientes, limpia, aplica 5-core y genera:

  data/interacciones.parquet  una fila por (usuario, ítem, fecha), ordenada
  data/secuencias.parquet     una fila por usuario: lista de ítems y fechas
  data/mapa_items.csv         item_idx -> article_id + atributos del producto
  data/mapa_usuarios.csv      user_idx -> customer_id
  data/hm.txt                 formato SASRec.pytorch: "user item" por línea
  data/resumen_preparacion.json

Índices desde 1 (0 = relleno), como pide el enunciado y como espera SASRec.

Uso:
  python scripts/preparar_hm.py --desde 2020-03-23 --pct_clientes 10
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
OUT = RAIZ / "data"

COLS_ART = ["article_id", "product_code", "prod_name", "product_type_name",
            "product_group_name", "colour_group_name", "department_name",
            "index_group_name", "section_name", "garment_group_name"]


def encontrar(nombre):
    """Busca el archivo en data/raw/ o en subcarpetas, sin comprimir o en .zip."""
    for patron in (nombre, nombre + ".zip"):
        hallados = sorted(p for p in RAW.rglob(patron) if p.is_file())
        if hallados:
            return hallados[0]
    raise FileNotFoundError(f"No encontré {nombre} dentro de {RAW}")


def leer_transacciones(desde, pct, chunksize=2_000_000):
    """Lectura por bloques (el CSV pesa ~3,5 GB). Filtra fecha y muestra de
    clientes por hash del customer_id (determinístico: misma muestra siempre)."""
    partes, n_total = [], 0
    lector = pd.read_csv(
        encontrar("transactions_train.csv"),
        usecols=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
        dtype={"customer_id": str, "article_id": np.int32, "price": np.float32,
               "sales_channel_id": np.int8},
        chunksize=chunksize,
    )
    for i, ch in enumerate(lector):
        ch["fila"] = np.arange(n_total, n_total + len(ch), dtype=np.int64)  # orden original del archivo
        n_total += len(ch)
        ch = ch[ch.t_dat >= desde]
        if pct < 100:
            cubeta = ch.customer_id.str[-8:].apply(int, base=16) % 100
            ch = ch[cubeta < pct]
        partes.append(ch)
        print(f"  bloque {i + 1}: {n_total:,} filas leídas, {sum(len(p) for p in partes):,} conservadas", flush=True)
    df = pd.concat(partes, ignore_index=True)
    df["t_dat"] = pd.to_datetime(df.t_dat)
    df["customer_id"] = df.customer_id.astype("category")
    return df, n_total


def kcore(df, k, min_dias=1):
    """Filtro k-core iterativo: cada usuario y cada ítem con >= k interacciones.
    Si min_dias > 1, además cada usuario debe haber comprado en >= min_dias días
    distintos (se repite hasta que ambas condiciones se cumplen a la vez)."""
    it = 0
    while True:
        it += 1
        cu = df.groupby("customer_id", observed=True).article_id.transform("size")
        ci = df.groupby("article_id").customer_id.transform("size")
        m = (cu >= k) & (ci >= k)
        if min_dias > 1:
            m &= df.groupby("customer_id", observed=True).t_dat.transform("nunique") >= min_dias
        if m.all():
            return df, it
        df = df[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="2020-03-23", help="fecha mínima AAAA-MM-DD (la base termina el 2020-09-22)")
    ap.add_argument("--pct_clientes", type=int, default=10, help="porcentaje de clientes a muestrear (1-100)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min_dias", type=int, default=2,
                    help="días de compra distintos mínimos por usuario (2 = sin usuarios de un solo carrito)")
    a = ap.parse_args()
    t0 = time.time()
    res = {"parametros": vars(a)}

    print("1) Leyendo transacciones…")
    df, n_raw = leer_transacciones(a.desde, a.pct_clientes)
    res["filas_originales"] = int(n_raw)
    res["tras_filtro_fecha_y_muestra"] = int(len(df))

    print("2) Limpieza…")
    # Varias unidades del mismo artículo, mismo cliente, mismo día = una interacción
    antes = len(df)
    df = df.sort_values(["customer_id", "t_dat", "fila"]).drop_duplicates(
        ["customer_id", "article_id", "t_dat"], keep="first")
    res["duplicados_mismo_dia"] = int(antes - len(df))

    print(f"3) Filtro {a.k}-core + mínimo {a.min_dias} días de compra por usuario…")
    df, it = kcore(df, a.k, a.min_dias)
    df["customer_id"] = df.customer_id.cat.remove_unused_categories()
    res["iteraciones_kcore"] = it

    # Índices desde 1 (0 = relleno)
    usuarios = df.customer_id.cat.categories
    df["user_idx"] = df.customer_id.cat.codes.astype(np.int32) + 1
    items = np.sort(df.article_id.unique())
    mapa_item = pd.Series(np.arange(1, len(items) + 1, dtype=np.int32), index=items)
    df["item_idx"] = df.article_id.map(mapa_item).astype(np.int32)

    # Orden temporal; dentro del mismo día se respeta el orden del archivo original
    df = df.sort_values(["user_idx", "t_dat", "fila"]).reset_index(drop=True)
    df["pos"] = df.groupby("user_idx").cumcount()

    nu, ni, n = len(usuarios), len(items), len(df)
    largos = df.groupby("user_idx").size()
    mismo_dia = df.groupby("user_idx").t_dat.diff().dt.days.eq(0).mean()
    ult2 = df.groupby("user_idx").t_dat.agg(lambda s: s.iloc[-1] == s.iloc[-2])
    res["pct_usuarios_prueba_y_validacion_mismo_dia"] = float(ult2.mean())
    res.update({
        "interacciones": int(n), "usuarios": int(nu), "items": int(ni),
        "densidad": n / (nu * ni),
        "largo_secuencia": {"media": float(largos.mean()), "mediana": float(largos.median()),
                            "p90": float(largos.quantile(.9)), "max": int(largos.max())},
        "pct_pasos_mismo_dia": float(mismo_dia),
        "periodo": [str(df.t_dat.min().date()), str(df.t_dat.max().date())],
        "cumple_minimos": {"interacciones>=50000": n >= 50_000, "usuarios>=2000": nu >= 2_000,
                           "items>=500": ni >= 500},
    })

    print("4) Guardando archivos…")
    OUT.mkdir(exist_ok=True)
    df[["user_idx", "item_idx", "t_dat", "pos", "price", "sales_channel_id"]].to_parquet(
        OUT / "interacciones.parquet", index=False)
    seq = df.groupby("user_idx").agg(items=("item_idx", list), fechas=("t_dat", list)).reset_index()
    seq.to_parquet(OUT / "secuencias.parquet", index=False)
    with open(OUT / "hm.txt", "w") as f:
        for u, i in zip(df.user_idx.values, df.item_idx.values):
            f.write(f"{u} {i}\n")

    art = pd.read_csv(encontrar("articles.csv"), usecols=COLS_ART, dtype={"article_id": np.int32})
    precio = df.groupby("article_id").price.median().rename("precio_mediano")
    mi = (pd.DataFrame({"item_idx": mapa_item.values, "article_id": mapa_item.index})
          .merge(art, on="article_id", how="left").merge(precio, on="article_id", how="left"))
    mi.to_csv(OUT / "mapa_items.csv", index=False)
    pd.DataFrame({"user_idx": np.arange(1, nu + 1), "customer_id": usuarios}).to_csv(
        OUT / "mapa_usuarios.csv", index=False)
    res["items_sin_atributos"] = int(mi.product_type_name.isna().sum())
    res["segundos"] = round(time.time() - t0, 1)

    json.dump(res, open(OUT / "resumen_preparacion.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
