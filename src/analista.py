"""
Analista con IA (API de Gemini, plan gratuito). El Transformer recomienda; el analista solo explica, contrasta y advierte.

- `construir_contexto`: arma el JSON con historial (atributos + pesos de atención), lista del modelo con puntajes,
  lista de una línea base, popularidad e indicadores calculados por código (el LLM no calcula, lee).
- `Analista.responder`: llama a Gemini con las instrucciones versionadas de src/analista_prompt.md y una herramienta,
  `reconsultar_modelo`, que es la ÚNICA forma de obtener otra lista: la ejecuta el recomendador, no el LLM.
- `verificar`: control automático de honestidad sobre cada respuesta (referencias inexistentes, prendas del
  catálogo que no estaban en el contexto, listas numeradas en otro orden que el del modelo).
- Se registran tokens y latencia de cada llamada. En el plan gratuito el costo es US$ 0; además se informa el costo
  equivalente con la tarifa pagada publicada, para saber cuánto costaría en producción.
"""
import json
import re
import time
from pathlib import Path

import numpy as np

MODELO_LLM = "gemini-3.5-flash-lite"      # plan gratuito de Google AI Studio; rápido (≈1 s) para una demo en vivo
MODELO_RESPALDO = "gemini-3.1-flash-lite"   # si se agota la cuota gratuita o el principal no responde
PRECIOS_USD_MTOK = {                      # tarifa pagada publicada (entrada, salida) por millón de tokens; gratis = 0
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
}
MAX_TOKENS_ANALISIS, MAX_TOKENS_CHAT, MAX_RONDAS_HERRAMIENTA = 4000, 3000, 2   # incluye tokens de razonamiento
RUTA_PROMPT = Path(__file__).with_name("analista_prompt.md")

CONSULTA_TXT = {"carrito": "sigue comprando hoy (completar el carrito)", "visita": "vuelve otro día (próxima visita)"}

HERRAMIENTA = {
    "name": "reconsultar_modelo",
    "description": ("Vuelve a pedir la recomendación al recomendador con un historial modificado y/o con otro modelo. "
                    "Es la única forma de obtener una lista distinta: tú no puedes armarla. Devuelve la nueva lista, "
                    "en el orden del modelo, con referencias Q1…Qn."),
    "parameters": {
        "type": "object",
        "properties": {
            "agregar_al_final": {"type": "array", "items": {"type": "string"},
                                 "description": "Referencias (P#, B#, Q#) de prendas a sumar al final del historial, como si la clienta las hubiera comprado."},
            "quitar": {"type": "array", "items": {"type": "string"},
                       "description": "Referencias H# de compras a sacar del historial."},
            "modelo": {"type": "string", "enum": ["sasrec", "hibrido", "knn", "reglas", "pop"],
                       "description": "sasrec = Transformer; hibrido = innovación del grupo; knn = ítem–ítem; reglas = reglas de asociación; pop = populares."},
            "consulta": {"type": "string", "enum": ["carrito", "visita"]},
        },
    },
}


def _prenda(rec, it, ref, extra=None):
    f = rec.ficha(it)
    d = {"ref": ref, "prenda": f["prenda"], "tipo": f["tipo"], "grupo": f["grupo"], "color": f["color"],
         "linea": f["linea"], "ranking_popularidad": f["ranking_popularidad"]}
    if extra: d.update(extra)
    return d


def _indicadores(rec, hist, lista, base, refs_lista, refs_hist):
    prod_l = [int(rec.producto[i]) for i in lista]
    grupos = {}
    for ref, p in zip(refs_lista, prod_l): grupos.setdefault(p, []).append(ref)
    repetidas = [v for v in grupos.values() if len(v) > 1]
    prod_h = {int(rec.producto[h]): r for h, r in zip(hist, refs_hist)}
    otro_color = [f"{r} es otro color de {prod_h[p]}" for r, p in zip(refs_lista, prod_l) if p in prod_h]
    rk = [int(rec.rank_pop[i]) for i in lista]
    return {
        "largo_historial": len(hist),
        "arranque_en_frio": len(hist) <= 1,
        "variantes_de_color_repetidas_en_la_lista": repetidas,
        "prendas_de_la_lista_que_son_otro_color_de_una_compra": otro_color,
        "en_comun_con_linea_base": [r for r, i in zip(refs_lista, lista) if i in set(base)],
        "ranking_popularidad_mediano_lista": int(np.median(rk)) if rk else None,
        "ranking_popularidad_mediano_historial": int(np.median([rec.rank_pop[h] for h in hist])) if len(hist) else None,
        "prendas_de_la_lista_en_el_1pct_mas_vendido": int(sum(r <= rec.n_items // 100 for r in rk)),
        "lineas_en_la_lista": {k: int(v) for k, v in zip(*np.unique([rec.ficha(i)["linea"] for i in lista], return_counts=True))},
    }


def construir_contexto(rec, hist, consulta, modelo, n=10, modelo_base="knn", metricas=None):
    """Devuelve (contexto, refs) donde refs mapea cada referencia (H#, P#, B#) a su item_idx."""
    hist = [int(h) for h in hist]
    att = rec.atencion(hist) if hist else np.zeros(0)
    df = rec.recomendar(hist, modelo, n=n, tarea=consulta)
    dfb = rec.recomendar(hist, modelo_base, n=n, tarea=consulta)
    refs = {}
    historial = []
    for j, (h, w) in enumerate(zip(hist, att), 1):
        refs[f"H{j}"] = h
        historial.append(_prenda(rec, h, f"H{j}", {"peso_atencion": round(float(w), 3)}))
    lista = []
    for _, r in df.iterrows():
        ref = f"P{r.posicion}"; refs[ref] = int(r.item_idx)
        lista.append(_prenda(rec, r.item_idx, ref, {"posicion": int(r.posicion), "puntaje": round(float(r.puntaje), 4), "por_que": r.por_que}))
    base = []
    for _, r in dfb.iterrows():
        ref = f"B{r.posicion}"; refs[ref] = int(r.item_idx)
        base.append(_prenda(rec, r.item_idx, ref, {"posicion": int(r.posicion), "por_que": r.por_que}))
    from recomendador import NOMBRES_MODELO
    ctx = {
        "consulta": CONSULTA_TXT[consulta],
        "historial": historial,
        "recomendacion": {"modelo": NOMBRES_MODELO[modelo], "lista": lista},
        "linea_base": {"modelo": NOMBRES_MODELO[modelo_base], "lista": base},
        "indicadores": _indicadores(rec, hist, df.item_idx.tolist(), dfb.item_idx.tolist(),
                                    [f"P{p}" for p in df.posicion], [f"H{j}" for j in range(1, len(hist) + 1)]),
        "catalogo": {"prendas": rec.n_items, "nota_ranking": "ranking_popularidad 1 = la más vendida en entrenamiento"},
    }
    if metricas: ctx["metricas_modelo"] = metricas
    return ctx, refs


def _normalizar(x):
    return " ".join(re.sub(r"[^\w\s]", " ", str(x).lower()).split())


def verificar(texto, refs_validas, rec, items_contexto, orden_modelo=None):
    """Control de honestidad. Devuelve lista de problemas (vacía = sin problemas detectados)."""
    problemas = []
    citadas = set(re.findall(r"\b([HPBQ]\d{1,2})\b", texto))
    inexistentes = sorted(citadas - set(refs_validas))
    if inexistentes:
        problemas.append(f"cita referencias que no existen en el contexto: {', '.join(inexistentes)}")
    nombres_ctx = {_normalizar(rec.mapa.loc[i, "prod_name"]) for i in items_contexto}
    texto_l = f" {_normalizar(texto)} "
    for nombre in sorted(nombres_ctx, key=len, reverse=True):     # primero se borran las prendas que sí estaban
        texto_l = texto_l.replace(f" {nombre} ", " ")
    fuera = [n for n in rec.nombres_catalogo if n not in nombres_ctx and f" {n} " in texto_l]
    if fuera:
        problemas.append(f"menciona prendas del catálogo que no estaban en el contexto: {', '.join(sorted(fuera)[:5])}")
    numeradas = [re.findall(r"\bP(\d{1,2})\b", l) for l in texto.splitlines() if re.match(r"\s*\d+[.)]\s", l)]
    primeras = [int(x[0]) for x in numeradas if x]
    if len(primeras) >= 3 and primeras != sorted(primeras):
        problemas.append("presenta una lista numerada de prendas en un orden distinto al del modelo")
    return problemas


class Analista:
    def __init__(self, api_key, rec, modelo_llm=MODELO_LLM, plan_gratuito=True):
        from google import genai
        from google.genai import types
        self.genai, self.types = genai, types
        self.cliente = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))   # ms; evita esperas indefinidas
        self.rec, self.modelo_llm, self.plan_gratuito = rec, modelo_llm, plan_gratuito
        self.sistema = RUTA_PROMPT.read_text(encoding="utf-8")
        self.herramienta = types.Tool(function_declarations=[types.FunctionDeclaration(
            name=HERRAMIENTA["name"], description=HERRAMIENTA["description"],
            parameters_json_schema=HERRAMIENTA["parameters"])])
        self._con_razonamiento_bajo = True
        if not hasattr(rec, "nombres_catalogo"):
            nombres = {_normalizar(x) for x in rec.mapa.prod_name.unique()}
            rec.nombres_catalogo = {x for x in nombres if len(x) >= 8 and " " in x}   # evita palabras sueltas

    def costo_equivalente_usd(self, uso, modelo=None):
        """Lo que costaría la llamada con la tarifa pagada (los tokens de razonamiento se cobran como salida)."""
        p_in, p_out = PRECIOS_USD_MTOK.get(modelo or self.modelo_llm, (0, 0))
        return (uso["input_tokens"] * p_in + (uso["output_tokens"] + uso["thinking_tokens"]) * p_out) / 1e6

    def _config(self, max_tokens):
        t = self.types
        kw = dict(system_instruction=self.sistema, tools=[self.herramienta], max_output_tokens=max_tokens,
                  temperature=0.3, automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True))
        if self._con_razonamiento_bajo:
            kw["thinking_config"] = t.ThinkingConfig(thinking_level="low")
        return t.GenerateContentConfig(**kw)

    def _llamar(self, contenidos, max_tokens):
        """Una llamada a Gemini con reintentos: cuota (429) → espera y luego modelo de respaldo."""
        errores = self.genai.errors
        modelos = [self.modelo_llm] + ([MODELO_RESPALDO] if MODELO_RESPALDO != self.modelo_llm else [])
        ultimo = None
        for modelo in modelos:
            for intento in range(2):
                try:
                    return self.cliente.models.generate_content(model=modelo, contents=contenidos,
                                                                config=self._config(max_tokens)), modelo
                except errores.ClientError as e:
                    ultimo = e
                    if e.code == 400 and self._con_razonamiento_bajo and "thinking" in str(e).lower():
                        self._con_razonamiento_bajo = False          # modelo sin control de razonamiento
                        continue
                    if e.code == 429:
                        if intento == 0: time.sleep(5); continue
                        break                                        # pasa al modelo de respaldo
                    raise
                except errores.ServerError as e:                     # 500/503: sobrecarga momentánea
                    ultimo = e
                    if intento == 0: time.sleep(3); continue
                    break
        raise ultimo

    def _ejecutar_herramienta(self, args, estado):
        refs = estado["refs"]
        hist = list(estado["hist"])
        args = dict(args or {})
        quitar = {refs[r] for r in args.get("quitar", []) if r in refs and r.startswith("H")}
        hist = [h for h in hist if h not in quitar]
        no_validas = [r for r in list(args.get("agregar_al_final", [])) + list(args.get("quitar", [])) if r not in refs]
        hist += [refs[r] for r in args.get("agregar_al_final", []) if r in refs]
        modelo = args.get("modelo") or estado["modelo"]
        consulta = args.get("consulta") or estado["consulta"]
        df = self.rec.recomendar(hist, modelo, n=estado["n"], tarea=consulta)
        nueva = []
        for _, r in df.iterrows():
            ref = f"Q{r.posicion}"; refs[ref] = int(r.item_idx); estado["items"].add(int(r.item_idx))
            nueva.append(_prenda(self.rec, r.item_idx, ref, {"posicion": int(r.posicion), "por_que": r.por_que}))
        from recomendador import NOMBRES_MODELO
        estado["reconsultas"].append({"modelo": modelo, "consulta": consulta, "historial": hist, "lista": df.item_idx.tolist()})
        return {"historial_usado": [self.rec.ficha(h)["prenda"] for h in hist], "modelo": NOMBRES_MODELO[modelo],
                "consulta": CONSULTA_TXT[consulta], "lista": nueva,
                "referencias_ignoradas": no_validas}

    def responder(self, estado, pregunta=None):
        """estado: dict con ctx, refs, hist, modelo, consulta, n, items, mensajes, reconsultas. Devuelve dict con texto y métricas."""
        t = self.types
        primera = not estado["mensajes"]
        if primera:
            texto_usuario = ("Contexto de la recomendación (JSON):\n" + json.dumps(estado["ctx"], ensure_ascii=False)
                             + "\n\n" + (pregunta or "Haz el análisis inicial."))
        else:
            texto_usuario = pregunta
        estado["mensajes"].append(t.Content(role="user", parts=[t.Part.from_text(text=texto_usuario)]))
        uso = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "cached_tokens": 0}
        t0, rondas, herramientas, modelos_usados = time.perf_counter(), 0, [], []
        max_tokens = MAX_TOKENS_ANALISIS if primera else MAX_TOKENS_CHAT
        costo_eq = 0.0
        while True:
            r, modelo = self._llamar(estado["mensajes"], max_tokens)
            modelos_usados.append(modelo)
            u = r.usage_metadata
            parcial = {"input_tokens": (u.prompt_token_count or 0) + (u.tool_use_prompt_token_count or 0) if u else 0,
                       "output_tokens": (u.candidates_token_count or 0) if u else 0,
                       "thinking_tokens": (u.thoughts_token_count or 0) if u else 0,
                       "cached_tokens": (u.cached_content_token_count or 0) if u else 0}
            for k in uso: uso[k] += parcial[k]
            costo_eq += self.costo_equivalente_usd(parcial, modelo)
            contenido = r.candidates[0].content if r.candidates and r.candidates[0].content else None
            if contenido is not None:
                estado["mensajes"].append(contenido)          # conserva firmas de razonamiento para la herramienta
            llamadas = r.function_calls or []
            if not llamadas or rondas >= MAX_RONDAS_HERRAMIENTA:
                break
            rondas += 1
            partes = []
            for fc in llamadas:
                salida = self._ejecutar_herramienta(fc.args, estado)
                herramientas.append({"entrada": dict(fc.args or {}), "lista": [x["prenda"] for x in salida["lista"]]})
                partes.append(t.Part.from_function_response(name=fc.name, response={"resultado": salida}))
            estado["mensajes"].append(t.Content(role="user", parts=partes))
        texto = "\n".join(p.text for p in (contenido.parts or []) if getattr(p, "text", None) and not getattr(p, "thought", False)).strip() \
            if contenido is not None else ""
        if not texto:
            texto = "(El modelo no devolvió texto; vuelve a intentarlo.)"
        latencia = time.perf_counter() - t0
        problemas = verificar(texto, estado["refs"].keys(), self.rec, estado["items"])
        return {"texto": texto, "uso": uso, "costo_usd": 0.0 if self.plan_gratuito else costo_eq,
                "costo_equivalente_usd": costo_eq, "latencia_s": latencia, "modelo_llm": modelos_usados[-1],
                "herramientas": herramientas, "problemas": problemas}


def nuevo_estado(rec, hist, consulta, modelo, n=10, modelo_base="knn", metricas=None):
    ctx, refs = construir_contexto(rec, hist, consulta, modelo, n=n, modelo_base=modelo_base, metricas=metricas)
    return {"ctx": ctx, "refs": refs, "hist": [int(h) for h in hist], "modelo": modelo, "consulta": consulta, "n": n,
            "items": set(refs.values()), "mensajes": [], "reconsultas": []}
