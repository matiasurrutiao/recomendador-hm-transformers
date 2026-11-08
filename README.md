# Entrega 1 — Recomendador secuencial para H&M con Transformers

Machine Learning no supervisado (C2) · UDD

**Integrantes:** Valentina Romero y Matías Urrutia

**Usuaria y decisión:** una clienta de H&M online que acaba de comprar algo; el recomendador decide qué 10 prendas mostrarle en "Completa tu look" o en el correo de la semana.

## Cómo ejecutar el notebook (≈ 15 minutos en CPU)

1. **Requisitos:** Python 3.10, 3.11 o 3.12 y unos 2 GB libres (para instalar PyTorch). No necesita GPU. Probado en Windows con Python 3.11.9 y en Linux con Python 3.13.
2. Abrir **la carpeta completa** en VS Code (o Jupyter) y abrir `notebooks/entrega1_recomendador_hm.ipynb`.
3. Elegir un kernel de Python y presionar **Run All**.
   - La primera celda instala las librerías (`pandas pyarrow scipy matplotlib jinja2 mlxtend scikit-learn torch`, las mismas de `requirements.txt`). También se pueden instalar antes con `pip install -r requirements.txt`.
   - **SASRec no se reentrena:** el notebook carga los modelos de `modelos/` y muestra el registro de cada época. Reentrenar las 4 configuraciones toma ≈ 1 h 45 min en CPU; para hacerlo, borrar `modelos/sasrec_*.pt` y `modelos/historial_sasrec.json` (o correr `cd notebooks` y luego `python ../scripts/entrenar_sasrec.py`).
4. Semilla fija (42): los resultados coinciden con los guardados en `data/metricas.csv`.

**En Google Colab:** agregar al inicio una celda que clone el repositorio y entre a la carpeta del notebook:

```python
!git clone https://github.com/matiasurrutiao/recomendador-hm-transformers.git
%cd recomendador-hm-transformers/notebooks
```

**Problema conocido:** si la celda de instalación responde `error: externally-managed-environment`, el kernel elegido es un Python administrado por otra herramienta (por ejemplo `uv`) que no permite instalar paquetes. Basta con elegir otro Python en el selector de kernel.

## Contenido

```
Recomendador secuencial para H&M con Transformers — Entrega 1.pdf   informe de 3 páginas
notebooks/
  entrega1_recomendador_hm.ipynb       notebook ejecutado con todas sus salidas
data/
  secuencias.parquet                   una fila por clienta: lista de prendas y fechas (índices desde 1, 0 = relleno)
                                       sin identificadores de clientas: solo user_idx (1..36.168)
  mapa_items.csv                       item_idx → article_id, nombre, tipo, color, departamento, línea, precio
  interacciones.parquet                una fila por (clienta, prenda, fecha)
  resumen_preparacion.json             números de la limpieza
  metricas.csv                         métricas de todos los modelos en validación y prueba
  hm.txt                               mismas secuencias en el formato de SASRec.pytorch ("usuario ítem")
modelos/
  sasrec.pt                            SASRec elegido en validación (configuración A, 4,7 MB)
  sasrec_A.pt … sasrec_D.pt            las 4 configuraciones entrenadas
  historial_sasrec.json                pérdida y métricas de validación por época
scripts/
  preparar_hm.py                       limpieza, muestra, 5-core y secuencias desde los CSV de Kaggle
  entrenar_sasrec.py                   el mismo código de SASRec del notebook, para entrenar fuera de Jupyter
requirements.txt
```

Los CSV originales de Kaggle (3,5 GB) no se incluyen. Para regenerar `data/` desde cero: descargar `transactions_train.csv` y `articles.csv` de *H&M Personalized Fashion Recommendations* en `data/raw/` y correr
`python scripts/preparar_hm.py --desde 2020-03-23 --pct_clientes 10 --min_dias 2`.

## Resultados en prueba (leave-last-out, ranking completo de 17.801 prendas)

| Modelo | HR@10 | HR@20 | NDCG@10 | MRR | Cobertura@10 | Entrenamiento (CPU) |
|---|---|---|---|---|---|---|
| Popularidad global | 0,87 % | 1,48 % | 0,004 | 0,004 | 0,1 % | 0 s |
| Popularidad reciente (3 días) | 1,65 % | 2,96 % | 0,008 | 0,009 | 2,8 % | 0,1 s |
| Reglas de asociación (Certamen 2) | 8,18 % | 9,58 % | 0,051 | 0,044 | 21,8 % | 6 s |
| Ítem–ítem coseno (λ = 0,8) | 8,72 % | 10,35 % | **0,060** | **0,055** | **91,4 %** | 1 s |
| **SASRec** (1 bloque, d = 64, entropía cruzada) | **8,95 %** | **11,73 %** | 0,057 | 0,052 | 67,8 % | 38 min |

## Referencias

- Kang, W.-C. & McAuley, J. (2018). *Self-Attentive Sequential Recommendation*. ICDM.
- Huang, Z. *SASRec.pytorch*. https://github.com/pmixer/SASRec.pytorch
- Klenitskiy, A. & Vasilev, A. (2023). *Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?* RecSys.
- H&M Personalized Fashion Recommendations, Kaggle (2022).
