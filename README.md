# Recomendador secuencial para H&M con Transformers — Entregas 1 y 2

Machine Learning no supervisado (C2) · UDD

**Integrantes:** Valentina Romero y Matías Urrutia

**Usuaria y decisión:** el equipo de e-commerce de una tienda de moda online decide qué 10 prendas mostrarle a una clienta que acaba de comprar, en "Completa tu look" o en el correo de la semana. Un Transformer (SASRec) recomienda a partir de la secuencia de compras, y un analista con IA explica, contrasta y advierte sobre esa lista, sin reordenarla nunca.

**Dashboard:** https://recomendador-hm-transformers-m7dbvcss94tlbt4jnzlgcq.streamlit.app

## Entrega 2 en 1 minuto

| Pieza | Dónde |
|---|---|
| Dashboard (Streamlit): simulador de historial, top-N de SASRec y del híbrido, líneas base, pesos de atención, analista con chat, métricas y vitrina | `src/app.py` |
| Analista con IA (API de Gemini, plan gratuito): contexto estructurado, análisis inicial, chat que reconsulta al modelo y control automático de honestidad | `src/analista.py` |
| Instrucciones de sistema del analista, versionadas | `src/analista_prompt.md` |
| Innovación (híbrido SASRec + ítem–ítem con diversidad), pruebas de honestidad, costo y latencia | `notebooks/entrega2_analista_e_innovacion.ipynb` |
| Recomendador compartido por la app y el notebook (SASRec, ítem–ítem, reglas, popularidad, híbrido) | `src/recomendador.py`, `src/modelo_sasrec.py`, `src/evaluacion.py` |

La app **no entrena nada**: carga `modelos/sasrec.pt` en CPU con `st.cache_resource` y usa los vecinos de ítem–ítem y las reglas ya calculados (`scripts/precalcular_app.py`).

## Cómo correr el dashboard

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

El analista necesita una clave **gratuita** de la API de Gemini, que se crea en [Google AI Studio](https://aistudio.google.com) con "Get API key". Hay dos formas de dársela, y en ninguna la clave queda en GitHub:

- **En tu computador:** copia `.streamlit/secrets.toml.ejemplo` como `.streamlit/secrets.toml` y pega ahí la clave. Ese archivo está en `.gitignore`.
- **En Streamlit Community Cloud:** pega `GEMINI_API_KEY = "..."` en *App settings → Secrets*.

Sin clave, el dashboard funciona igual: solo la pestaña del analista muestra un aviso.

**Publicar en Streamlit Community Cloud:**
1. En share.streamlit.io elige *Create app* y luego este repositorio, rama `main`.
2. Main file: `src/app.py`.
3. En *Advanced settings*: Python 3.12 y, en Secrets, la clave.

La nube usa `src/requirements.txt`, que instala PyTorch solo para CPU.

## Innovación: híbrido SASRec + ítem–ítem con diversidad

**Problema.** El 72 % de las listas de SASRec trae la misma prenda en dos o más colores, y ningún modelo de la Entrega 1 gana en todo: SASRec acierta más seguido (HR) e ítem–ítem ubica mejor el acierto (NDCG) y cubre más catálogo.

**Qué hace:**
1. Fusiona por rangos recíprocos (RRF) un modelo secuencial (SASRec) y uno de co-ocurrencia (ítem–ítem coseno).
2. Arma el top 10 con **a lo sumo una variante de color por prenda** (`product_code`), tomando de los 50 mejores candidatos.

**Cómo se eligió.** Se probaron 8 variantes en validación, entre ellas la hipótesis de enrutar carrito → ítem–ítem y próxima visita → SASRec, que se descartó. La prueba se miró una sola vez. Mismo protocolo que la Entrega 1: leave-last-out, ranking completo de 17.801 prendas, sin lo ya comprado.

| Prueba (36.168 clientas) | HR@10 | NDCG@10 | MRR | Cobertura@10 | Listas con colores repetidos | Prendas distintas en el top 10 | Novedad (bits) |
|---|---|---|---|---|---|---|---|
| SASRec | 8,95 % | 0,057 | 0,052 | 67,8 % | 72 % | 8,3 | 9,2 |
| Ítem–ítem coseno | 8,73 % | 0,060 | 0,055 | 91,4 % | 49 % | 8,9 | 11,5 |
| Fusión sin diversidad (referencia) | 11,26 % | 0,072 | 0,064 | 91,5 % | 69 % | 8,2 | 10,0 |
| **Híbrido (innovación)** | **9,63 %** | **0,065** | **0,059** | **92,8 %** | **0 %** | **10,0** | 10,2 |

El híbrido supera a SASRec y a ítem–ítem a la vez en HR@10, NDCG@10, MRR y cobertura, y elimina las listas con colores repetidos. La fusión sin diversidad acierta más, pero repite colores en 7 de cada 10 listas: esa diferencia es el costo medido de la diversidad. El dashboard muestra siempre la pestaña de **SASRec puro** al lado del híbrido.

## Analista con IA

- **Modelo:** Gemini 3.5 Flash-Lite por el plan gratuito de Google AI Studio (responde en ~1 s). Si se agota la cuota o no responde, pasa solo a Gemini 3.1 Flash-Lite. Gemini 3.8 Flash se descartó porque en el plan gratuito tardaba más de 30 s. Razonamiento en nivel bajo y temperatura 0,3.
- **Contexto estructurado (JSON):**
  - el historial con sus pesos de atención;
  - la lista del modelo con puntaje, popularidad y el "por qué" que da el propio modelo;
  - una línea base;
  - indicadores calculados por código (repeticiones, coincidencias, popularidad);
  - las métricas del modelo.

  No incluye ningún identificador de clienta. Cada prenda tiene una referencia (`H#` historial, `P#` lista, `B#` línea base, `Q#` reconsulta) y el analista solo puede hablar de ellas.
- **Regla de oro:** el analista comenta y cuestiona, pero no agrega, no quita ni reordena. Si una pregunta pide otra recomendación ("¿y si hubiera comprado P2?"), llama a la herramienta `reconsultar_modelo`: la nueva lista la calcula el recomendador.
- **Control automático de honestidad** sobre cada respuesta. Detecta:
  - referencias inexistentes;
  - prendas del catálogo que no venían en el contexto;
  - listas numeradas en otro orden que el del modelo.

  Se probó sin falsos positivos en 146 contextos reales.
- **Cinco pruebas de honestidad** documentadas en el notebook 2, más una extra de arranque en frío. **Aprobó las 6**, y la revisión manual encontró una sola imprecisión menor (exagera un peso de atención):
  1. Análisis inicial.
  2. Pedido de reordenar.
  3. Prenda que no está.
  4. Pregunta sin datos.
  5. Otra recomendación vía herramienta.
- **Costo y latencia:** cada llamada registra tokens de entrada, salida y razonamiento, latencia y costo. El costo real es US$ 0 en el plan gratuito; también se informa lo que costaría con la tarifa pagada. Resultado de las pruebas: ~2 s por respuesta y, con la tarifa pagada, ≈ US$ 0,0025 por análisis inicial (≈ US$ 2,4 por 1.000). La app tiene un tope de 20 llamadas por sesión. En el plan gratuito Google puede usar el contenido para mejorar sus productos, y por eso el contexto no lleva datos personales.

## Cómo ejecutar los notebooks (CPU)

1. **Requisitos:** Python 3.10, 3.11 o 3.12 y unos 2 GB libres para instalar PyTorch. No necesita GPU.
2. Abrir **la carpeta completa** en VS Code o Jupyter, elegir un kernel de Python y presionar **Run All**.
   - `notebooks/entrega1_recomendador_hm.ipynb` toma unos 15 minutos. **SASRec no se reentrena**: el notebook carga `modelos/` y muestra el registro de cada época. Reentrenar las 4 configuraciones toma unas 1 h 45 min; se hace con `python ../scripts/entrenar_sasrec.py` desde `notebooks/`.
   - `notebooks/entrega2_analista_e_innovacion.ipynb` toma unos 5 minutos más las llamadas al analista. Lee la clave desde `.streamlit/secrets.toml` o desde la variable `GEMINI_API_KEY`. Sin clave, las celdas de la API lo avisan y se saltan.
3. La semilla es fija (42): los resultados coinciden con `data/metricas.csv` y `data/metricas_innovacion.csv`.

**En Google Colab:** clonar el repositorio con `!git clone https://github.com/matiasurrutiao/recomendador-hm-transformers.git` y luego entrar a la carpeta con `%cd recomendador-hm-transformers/notebooks`.

**Problema conocido:** si la instalación responde `externally-managed-environment`, el kernel elegido es un Python administrado por otra herramienta (por ejemplo `uv`). Basta con elegir otro Python en el selector de kernel.

## Contenido

```
src/
  app.py                    dashboard Streamlit
  recomendador.py           SASRec, ítem–ítem, reglas, popularidad e híbrido; explicaciones "por qué"
  modelo_sasrec.py          arquitectura SASRec (la misma del notebook 1) y carga en CPU
  evaluacion.py             métricas de ranking y de diversidad
  analista.py               contexto, llamada a Gemini, herramienta de reconsulta, control de honestidad
  analista_prompt.md        instrucciones de sistema del analista (versionadas)
  requirements.txt          dependencias del dashboard (PyTorch CPU)
notebooks/
  entrega1_recomendador_hm.ipynb          Entrega 1, ejecutado
  entrega2_analista_e_innovacion.ipynb    Entrega 2, ejecutado
data/
  secuencias.parquet        una fila por clienta: prendas y fechas (sin identificadores: solo user_idx)
  mapa_items.csv            item_idx → article_id, nombre, tipo, color, departamento, línea, precio
  interacciones.parquet     una fila por (clienta, prenda, fecha)
  metricas.csv              métricas de la Entrega 1 (validación y prueba)
  metricas_innovacion.csv   métricas de la innovación (validación y prueba)
  resumen_preparacion.json, hm.txt
modelos/
  sasrec.pt                 SASRec elegido (configuración A, 4,7 MB); sasrec_A…D.pt, historial_sasrec.json
  itemknn_vecinos.npz       100 vecinos por prenda (ítem–ítem), para la app
  reglas.parquet            17.199 reglas de asociación (método del Certamen 2), para la app
scripts/
  preparar_hm.py            limpieza, muestra, 5-core y secuencias desde los CSV de Kaggle
  entrenar_sasrec.py        entrenamiento de SASRec fuera de Jupyter
  precalcular_app.py        genera itemknn_vecinos.npz y reglas.parquet
  construir_notebook2.py    genera el notebook 2
.streamlit/
  config.toml               tema del dashboard
  secrets.toml.ejemplo      plantilla de la clave (secrets.toml no se sube)
```

Los CSV originales de Kaggle (3,5 GB) no se incluyen. Para regenerar `data/`, descargar `transactions_train.csv` y `articles.csv` de *H&M Personalized Fashion Recommendations* en `data/raw/` y correr `python scripts/preparar_hm.py --desde 2020-03-23 --pct_clientes 10 --min_dias 2`.

## Resultados de la Entrega 1 (prueba, leave-last-out, 17.801 prendas)

| Modelo | HR@10 | HR@20 | NDCG@10 | MRR | Cobertura@10 | Entrenamiento (CPU) |
|---|---|---|---|---|---|---|
| Popularidad global | 0,87 % | 1,48 % | 0,004 | 0,004 | 0,1 % | 0 s |
| Popularidad reciente (3 días) | 1,65 % | 2,96 % | 0,008 | 0,009 | 2,8 % | 0,1 s |
| Reglas de asociación (Certamen 2) | 8,18 % | 9,58 % | 0,051 | 0,044 | 21,8 % | 6 s |
| Ítem–ítem coseno (λ = 0,8) | 8,72 % | 10,35 % | **0,060** | **0,055** | **91,4 %** | 1 s |
| **SASRec** (1 bloque, d = 64, entropía cruzada) | **8,95 %** | **11,73 %** | 0,057 | 0,052 | 67,8 % | 38 min |

## Declaración de uso de IA

Usamos Claude (Anthropic) como asistente de programación y redacción: estructura del código, depuración, redacción del informe y de este README. Las decisiones de diseño (dataset, protocolo, modelos, innovación) las tomó el equipo y las validamos con las métricas del notebook. El analista del dashboard usa la API de Gemini (Google) en tiempo de ejecución, y sus pruebas de honestidad están documentadas en el notebook 2.

## Referencias

- Kang, W.-C. & McAuley, J. (2018). *Self-Attentive Sequential Recommendation*. ICDM.
- Huang, Z. *SASRec.pytorch*. https://github.com/pmixer/SASRec.pytorch
- Cormack, G., Clarke, C. & Buettcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*. SIGIR.
- Klenitskiy, A. & Vasilev, A. (2023). *Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?* RecSys.
- H&M Personalized Fashion Recommendations, Kaggle (2022).
