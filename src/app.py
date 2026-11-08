"""
Dashboard «Probador» — recomendador secuencial de moda con SASRec (Transformer) y analista con IA.
Ejecutar desde la raíz del repositorio:  streamlit run src/app.py
La clave de la API de Gemini (plan gratuito de Google AI Studio) va en .streamlit/secrets.toml (local) o en Secrets de Streamlit Cloud; nunca en GitHub.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analista import MODELO_LLM, Analista, nuevo_estado          # noqa: E402
from recomendador import HIBRIDO, NOMBRES_MODELO, Recomendador   # noqa: E402

MAX_LLAMADAS_SESION = 20      # tope de llamadas al LLM por sesión (cuida la cuota en la exposición)

st.set_page_config(page_title="Probador · recomendador de moda", page_icon="🧥", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.prenda {border: 1px solid #e7e2da; border-radius: 10px; padding: 8px 10px; margin-bottom: 6px; background: #fff;}
.prenda .n {font-weight: 600; font-size: 0.92rem; color: #1d1b18;}
.prenda .m {font-size: 0.78rem; color: #6b665e;}
.prenda .w {font-size: 0.74rem; color: #8a5a2b;}
.chip {display:inline-block; width: 12px; height: 12px; border-radius: 50%; border: 1px solid #bbb; margin-right: 6px; vertical-align: -1px;}
.pos {display:inline-block; min-width: 1.6rem; font-weight: 700; color: #b5452f;}
.etiqueta {font-size: 0.72rem; text-transform: uppercase; letter-spacing: .06em; color: #8a857c;}
</style>""", unsafe_allow_html=True)

COLORES = [("off white", "#f0ece2"), ("light beige", "#eadfcb"), ("dark beige", "#b9a27f"), ("beige", "#d6c4a8"),
           ("light grey", "#d3d3d3"), ("dark grey", "#555555"), ("grey", "#9e9e9e"), ("black", "#151515"),
           ("white", "#fafafa"), ("dark blue", "#1f2a44"), ("light blue", "#a9c7e8"), ("blue", "#3a5ba0"),
           ("light pink", "#f7d3dc"), ("dark pink", "#c2185b"), ("pink", "#f2a7b8"), ("dark red", "#7b1e1e"),
           ("red", "#c0392b"), ("dark green", "#2e4a2e"), ("light green", "#b5d6a7"), ("green", "#3c8d4f"),
           ("yellow", "#f2d14c"), ("orange", "#e67e22"), ("brown", "#6d4c41"), ("purple", "#7e57c2"),
           ("lilac", "#b39ddb"), ("turquoise", "#1abc9c"), ("gold", "#c9a227"), ("silver", "#c0c0c0"),
           ("khaki", "#8a8a5c"), ("bronze", "#a97142"), ("mole", "#8b7d6b")]


def hex_color(nombre):
    n = str(nombre).lower()
    return next((h for k, h in COLORES if k in n), "#cccccc")


@st.cache_resource(show_spinner="Cargando modelos (SASRec en CPU, sin entrenar nada)…")
def cargar():
    rec = Recomendador(RAIZ)
    met = pd.read_csv(RAIZ / "data" / "metricas.csv")
    ruta_inn = RAIZ / "data" / "metricas_innovacion.csv"
    inn = pd.read_csv(ruta_inn) if ruta_inn.exists() else None
    return rec, met, inn


rec, METRICAS, INNOVACION = cargar()


def metricas_de(modelo):
    """Métricas de prueba que el analista recibe como contexto."""
    nombre = {"sasrec": "SASRec", "knn": "Ítem–ítem coseno", "reglas": "Reglas de asociación", "pop": "Popularidad reciente"}.get(modelo)
    if modelo == "hibrido" and INNOVACION is not None:
        f = INNOVACION[(INNOVACION.conjunto == "prueba") & INNOVACION.modelo.str.startswith("Híbrido")].iloc[0]
    else:
        f = METRICAS[(METRICAS.conjunto == "prueba") & (METRICAS.modelo == nombre)]
        if f.empty: return None
        f = f.iloc[0]
    return {"HitRate@10": round(float(f["HR@10"]), 4), "NDCG@10": round(float(f["NDCG@10"]), 4),
            "Cobertura@10": round(float(f["Cobertura@10"]), 3), "nota": "medido en prueba (último ítem de 36.168 clientas)"}


def miles(x):
    return f"{int(x):,}".replace(",", ".")


def tarjeta(fila, extra=""):
    pos = f"<span class='pos'>{int(fila['posicion'])}</span>" if "posicion" in fila else ""
    return (f"<div class='prenda'>{pos}<span class='chip' style='background:{hex_color(fila['color'])}'></span>"
            f"<span class='n'>{fila['prenda']}</span><br><span class='m'>{fila['tipo']} · {fila['color']} · {fila['linea']}"
            f" · popularidad n.º {miles(fila['ranking_popularidad'])}</span>{extra}</div>")


def mostrar_lista(df, con_por_que=True):
    html = ""
    for _, r in df.iterrows():
        extra = f"<br><span class='w'>{r['por_que']}</span>" if con_por_que and r.get("por_que") else ""
        html += tarjeta(r, extra)
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------- barra lateral
@st.cache_data
def ejemplos():
    """Historiales reales de prueba (todo menos su última compra). Sin identificadores: solo un número de ejemplo."""
    casos = {"Cardigans (atención en una compra)": 356, "Historial largo, trajes de baño": 220,
             "Historial corto, traje verde": 43, "Básicos de mujer": 1200, "Línea Divided": 5000}
    out = {}
    for nombre, u in casos.items():
        h = rec.S[u][:-1]; f = rec.F[u]
        out[nombre] = (h.tolist(), "carrito" if f[-1] == f[-2] else "visita", int(rec.S[u][-1]))
    return out


@st.cache_data
def opciones_catalogo():
    m = rec.mapa
    orden = np.argsort(rec.rank_pop[m.index.values])
    return {f"{m.prod_name.iat[i]} · {m.product_type_name.iat[i]} · {m.colour_group_name.iat[i]} (#{m.index[i]})": int(m.index[i])
            for i in orden}


with st.sidebar:
    st.markdown("### 🧥 Probador")
    st.caption("Recomendador secuencial de moda · SASRec (Transformer) + analista con IA · datos H&M (Kaggle), 2020")
    modo = st.radio("Historial de la clienta", ["Ejemplo real", "Armar un historial", "Arranque en frío"])
    real_siguiente = None
    if modo == "Ejemplo real":
        ej = ejemplos()
        nombre_ej = st.selectbox("Ejemplo", list(ej))
        hist, consulta_sugerida, real_siguiente = ej[nombre_ej]
    elif modo == "Armar un historial":
        opciones = opciones_catalogo()
        elegidas = st.multiselect("Compras en orden (la última es la más reciente)", list(opciones),
                                  default=list(opciones)[:0], placeholder="Busca prendas por nombre, tipo o color")
        hist, consulta_sugerida = [opciones[e] for e in elegidas], "visita"
    else:
        hist, consulta_sugerida = [], "visita"
        st.info("Sin compras: el recomendador no tiene qué mirar y muestra lo más vendido de la última semana.")
    consulta = st.radio("¿Cuándo consulta?", ["visita", "carrito"], index=0 if consulta_sugerida == "visita" else 1,
                        format_func={"visita": "Vuelve otro día (próxima visita)", "carrito": "Sigue comprando hoy (completar el carrito)"}.get)
    n = st.slider("Tamaño de la lista (top-N)", 5, 20, 10)
    st.divider()
    st.caption("SASRec en CPU, cargado una vez con caché. Nada se entrena en la app. "
               f"Modelo: {rec.cfg_sasrec['bloques']} bloque, d = {rec.cfg_sasrec['d']}, {rec.parametros_sasrec:,} parámetros.".replace(",", "."))

st.markdown("## Probador de recomendaciones")
st.caption("El Transformer recomienda: la lista y su orden salen del modelo. El analista con IA solo explica, contrasta y advierte; nunca agrega, quita ni reordena prendas.")

tab_rec, tab_ia, tab_met, tab_cat = st.tabs(["🛍️ Recomendación", "🤖 Analista IA", "📈 Métricas", "🏷️ Vitrina"])

# --------------------------------------------------------------------------------------------- recomendación
with tab_rec:
    c_hist, c_att = st.columns([1.1, 1])
    with c_hist:
        st.markdown(f"<span class='etiqueta'>Historial · {len(hist)} compras, en orden</span>", unsafe_allow_html=True)
        if hist:
            att = rec.atencion(hist)
            muestra = list(enumerate(hist, 1))[-8:]
            html = ""
            for j, h in muestra:
                f = rec.ficha(h)
                html += tarjeta({**f}, f"<br><span class='w'>H{j} · atención de SASRec {att[j - 1]:.0%}</span>")
            if len(hist) > 8: st.caption(f"Se muestran las 8 más recientes de {len(hist)}.")
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.write("Sin compras (arranque en frío).")
    with c_att:
        st.markdown("<span class='etiqueta'>Transparencia · ¿dónde miró SASRec?</span>", unsafe_allow_html=True)
        if hist:
            att = rec.atencion(hist)
            ult = list(range(max(0, len(hist) - 12), len(hist)))
            dfa = pd.DataFrame({"compra": [f"H{j + 1} · {rec.ficha(hist[j])['prenda'][:28]}" for j in ult],
                                "peso de atención": att[ult]})
            st.bar_chart(dfa, x="compra", y="peso de atención", horizontal=True, color="#b5452f", height=320)
            st.caption("Pesos α de la única cabeza del único bloque, en la fila de la última posición (la que puntúa "
                       "el catálogo). Suman 1 sobre todo el historial. Dicen dónde miró el modelo, no por qué compró la clienta.")
        else:
            st.caption("Sin historial no hay atención que mostrar.")
        if real_siguiente:
            st.markdown(f"<span class='etiqueta'>Lo que la clienta compró después (oculto al modelo)</span>", unsafe_allow_html=True)
            st.markdown(tarjeta(rec.ficha(real_siguiente)), unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {NOMBRES_MODELO['sasrec']}")
        st.caption("Modelo puro, tal como se evaluó en la Entrega 1.")
        df_s = rec.recomendar(hist, "sasrec", n=n, tarea=consulta)
        mostrar_lista(df_s)
    with col2:
        st.markdown(f"#### {NOMBRES_MODELO['hibrido']} · innovación")
        st.caption("Fusión por rangos de SASRec + ítem–ítem; a lo sumo una variante de color por prenda.")
        df_h = rec.recomendar(hist, "hibrido", n=n, tarea=consulta)
        mostrar_lista(df_h)
    st.divider()
    st.markdown("#### Líneas base")
    tb = st.tabs([NOMBRES_MODELO[m] for m in ("knn", "reglas", "pop")])
    for t, m in zip(tb, ("knn", "reglas", "pop")):
        with t:
            mostrar_lista(rec.recomendar(hist, m, n=n, tarea=consulta))
    if real_siguiente:
        listas = {"SASRec": df_s, "Híbrido": df_h}
        aciertos = [f"{k}: posición {int(d.loc[d.item_idx == real_siguiente, 'posicion'].iat[0])}" for k, d in listas.items()
                    if (d.item_idx == real_siguiente).any()]
        st.info("Compra real siguiente en la lista → " + ("; ".join(aciertos) if aciertos else f"no aparece en el top-{n} de SASRec ni del híbrido."))

# --------------------------------------------------------------------------------------------- analista
with tab_ia:
    try:                                                    # sin secrets.toml, st.secrets lanza un error
        clave = st.secrets.get("GEMINI_API_KEY")
        if clave and clave.startswith("pega"): clave = None   # plantilla sin completar
        modelo_llm = st.secrets.get("MODELO_LLM", MODELO_LLM)
    except Exception:
        clave, modelo_llm = None, MODELO_LLM
    c1, c2 = st.columns([2, 1])
    with c1:
        modelo_analizado = st.radio("Lista que analiza", ["sasrec", "hibrido"], horizontal=True, format_func=NOMBRES_MODELO.get)
    with c2:
        modelo_base = st.selectbox("Línea base de contraste", ["knn", "reglas", "pop"], format_func=NOMBRES_MODELO.get)
    if not clave:
        st.warning("Falta la clave de la API de Gemini (`GEMINI_API_KEY` en Secrets; es gratis en aistudio.google.com). El resto del dashboard funciona sin ella.")
    else:
        if "analista" not in st.session_state:
            st.session_state.analista = Analista(clave, rec, modelo_llm)
            st.session_state.llamadas = 0
            st.session_state.registro = []
        firma = hashlib.md5(json.dumps([hist, consulta, modelo_analizado, modelo_base, n]).encode()).hexdigest()
        if st.session_state.get("firma") != firma:
            st.session_state.firma = firma
            st.session_state.estado = nuevo_estado(rec, hist, consulta, modelo_analizado, n=n, modelo_base=modelo_base,
                                                   metricas=metricas_de(modelo_analizado))
            st.session_state.chat = []

        def llamar(pregunta=None):
            if st.session_state.llamadas >= MAX_LLAMADAS_SESION:
                st.error(f"Se alcanzó el tope de {MAX_LLAMADAS_SESION} llamadas al analista en esta sesión.")
                return
            with st.spinner("El analista está leyendo el contexto…"):
                try:
                    r = st.session_state.analista.responder(st.session_state.estado, pregunta)
                except Exception as e:                              # sin internet, cuota agotada, etc.
                    st.error(f"No se pudo consultar al analista: {e}")
                    return
            st.session_state.llamadas += 1
            st.session_state.chat.append({"pregunta": pregunta, **r})
            st.session_state.registro.append({"pregunta": pregunta or "análisis inicial", "tokens_entrada": r["uso"]["input_tokens"],
                                              "tokens_salida": r["uso"]["output_tokens"] + r["uso"]["thinking_tokens"],
                                              "costo_usd": r["costo_usd"], "costo_equivalente_usd": r["costo_equivalente_usd"],
                                              "latencia_s": r["latencia_s"], "problemas": len(r["problemas"])})

        if not st.session_state.chat:
            llamar()                                                # análisis inicial automático

        for m in st.session_state.chat:
            if m["pregunta"]:
                st.chat_message("user").write(m["pregunta"])
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(m["texto"])
                for hrr in m["herramientas"]:
                    st.caption(f"↻ Reconsultó al modelo ({hrr['entrada'].get('modelo', 'mismo modelo')}): " + " · ".join(hrr["lista"][:10]))
                if m["problemas"]:
                    st.error("Control de honestidad: " + "; ".join(m["problemas"]))
                else:
                    st.caption("✔ Control de honestidad: solo citó prendas del contexto y no reordenó la lista.")
                st.caption(f"{miles(m['uso']['input_tokens'])} tokens de entrada · {miles(m['uso']['output_tokens'])} de salida "
                           f"(+{miles(m['uso']['thinking_tokens'])} de razonamiento) · {m['latencia_s']:.1f} s · "
                           f"US$ {m['costo_usd']:.4f} (plan gratuito; pagado ≈ US$ {m['costo_equivalente_usd']:.4f}) · {m['modelo_llm']}")
        pregunta = st.chat_input("Pregúntale al analista (p. ej.: ¿y si hubiera comprado P2?, ¿hay sesgo a lo popular?)")
        if pregunta:
            llamar(pregunta)
            st.rerun()
        if st.session_state.registro:
            reg = pd.DataFrame(st.session_state.registro)
            st.caption(f"Sesión: {len(reg)} llamadas · US$ {reg.costo_usd.sum():.4f} (pagado ≈ US$ {reg.costo_equivalente_usd.sum():.4f}) · latencia media {reg.latencia_s.mean():.1f} s · "
                       f"tope {MAX_LLAMADAS_SESION} llamadas")
    with st.expander("¿Qué recibe el analista? (contexto estructurado)"):
        if "estado" in st.session_state:
            st.json(st.session_state.estado["ctx"], expanded=False)
        st.caption("Instrucciones de sistema versionadas en src/analista_prompt.md. Sin identificadores de clientas.")

# --------------------------------------------------------------------------------------------- métricas
with tab_met:
    st.markdown("#### Prueba: último ítem de cada clienta (36.168), ranking completo de 17.801 prendas")
    t = METRICAS[METRICAS.conjunto == "prueba"][["modelo", "HR@5", "HR@10", "HR@20", "NDCG@10", "MRR", "Cobertura@10", "ajuste_s"]]
    st.dataframe(t.style.format({c: "{:.2%}" for c in ["HR@5", "HR@10", "HR@20", "Cobertura@10"]} |
                                {"NDCG@10": "{:.4f}", "MRR": "{:.4f}", "ajuste_s": "{:,.1f}"}), hide_index=True, use_container_width=True)
    st.caption("ajuste_s = segundos de entrenamiento en CPU. La cobertura detecta el sesgo a lo popular: "
               "la popularidad recomienda el 0,1 % del catálogo; SASRec, el 68 %.")
    if INNOVACION is not None:
        st.markdown("#### Innovación: híbrido SASRec + ítem–ítem con diversidad")
        cols = [c for c in ["modelo", "conjunto", "HR@10", "NDCG@10", "MRR", "Cobertura@10", "Listas con variantes repetidas",
                            "Prendas distintas@10", "Novedad@10 (bits)"] if c in INNOVACION.columns]
        st.dataframe(INNOVACION[cols], hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------------------------- vitrina
with tab_cat:
    st.markdown("#### Lo más vendido de la última semana de datos")
    c1, c2 = st.columns(2)
    linea = c1.selectbox("Línea", ["Todas"] + sorted(rec.mapa.index_group_name.unique()))
    grupo = c2.selectbox("Grupo", ["Todos"] + sorted(rec.mapa.product_group_name.unique()))
    orden = np.argsort(-rec.pop_reciente)
    filas = []
    for it in orden:
        if it == 0: continue
        f = rec.ficha(it)
        if (linea != "Todas" and f["linea"] != linea) or (grupo != "Todos" and f["grupo"] != grupo): continue
        filas.append({**f, "ventas_semana": int(rec.pop_reciente[it])})
        if len(filas) == 24: break
    cols = st.columns(3)
    for k, f in enumerate(filas):
        cols[k % 3].markdown(tarjeta(f, f"<br><span class='w'>{f['ventas_semana']} ventas en la semana</span>"), unsafe_allow_html=True)
    st.caption(f"Catálogo: {rec.n_items:,} prendas · ventas contadas en la muestra de entrenamiento.".replace(",", "."))
