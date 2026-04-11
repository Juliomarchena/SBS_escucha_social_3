"""
=============================================================
  MICROHELP — API Escucha Social SBS v3.0
  Arquitectura híbrida: RSS → BERT → Claude (solo si riesgo)

  Flujo:
    1. RSS feeds peruanos  → obtiene noticias reales
    2. BERT local          → filtra rápido y gratis
    3. Claude API          → profundiza solo score >= 20

  Costo real: Claude analiza ~10-20% de las noticias
  Autor: Julio Marchena · MICROHELP
=============================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import xml.etree.ElementTree as ET
import json
import io
import csv
import os

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL     = "https://api.anthropic.com/v1/messages"
UMBRAL_CLAUDE     = 20  # Si BERT da score >= este valor, Claude profundiza

app = FastAPI(
    title="API Escucha Social SBS v3.0",
    description="RSS Peruanos + BERT local + Claude para noticias críticas",
    version="3.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Google News RSS — sin API key, hasta 100 noticias por empresa
GOOGLE_NEWS_BASE = "https://news.google.com/rss/search?hl=es-419&gl=PE&ceid=PE:es-419&q="

# RSS fijos para noticias generales del sector financiero peruano
RSS_FEEDS_GENERALES = {
    "andina":      "https://andina.pe/agencia/rss.aspx?id=4",
    "elcomercio":  "https://elcomercio.pe/arc/outboundfeeds/rss/?outputType=xml",
}

# Estructura de alias por empresa:
# Cada empresa tiene 3 tipos de búsqueda:
#   "prensa"      → noticias generales
#   "bvl"         → hechos de importancia en bolsa
#   "regulatorio" → sanciones y resoluciones SMV/SBS
EMPRESAS_ALIAS = {
    "credicorp":     {
        "prensa":      ["Credicorp Peru", "BCP Banco de Credito"],
        "bvl":         ["Credicorp BVL hecho importancia"],
        "regulatorio": ["Credicorp SMV sancion SBS"],
    },
    "bcp":           {
        "prensa":      ["BCP Banco de Credito Peru"],
        "bvl":         ["BCP BVL hecho importancia acciones"],
        "regulatorio": ["BCP SBS sancion multa"],
    },
    "interbank":     {
        "prensa":      ["Interbank Peru banco"],
        "bvl":         ["Interbank BVL hecho importancia"],
        "regulatorio": ["Interbank SBS sancion multa"],
    },
    "bbva":          {
        "prensa":      ["BBVA Peru banco"],
        "bvl":         ["BBVA Peru BVL hecho importancia"],
        "regulatorio": ["BBVA Peru SBS sancion"],
    },
    "scotiabank":    {
        "prensa":      ["Scotiabank Peru"],
        "bvl":         ["Scotiabank Peru BVL hecho importancia"],
        "regulatorio": ["Scotiabank Peru SBS sancion"],
    },
    "mibanco":       {
        "prensa":      ["Mibanco Peru microfinanzas"],
        "bvl":         ["Mibanco BVL hecho importancia"],
        "regulatorio": ["Mibanco SBS sancion multa"],
    },
    "rimac":         {
        "prensa":      ["Rimac Seguros Peru"],
        "bvl":         ["Rimac Seguros BVL hecho importancia"],
        "regulatorio": ["Rimac Seguros SMV SBS sancion"],
    },
    "pacifico":      {
        "prensa":      ["Pacifico Seguros Peru"],
        "bvl":         ["Pacifico Seguros BVL hecho importancia"],
        "regulatorio": ["Pacifico Seguros SMV SBS sancion"],
    },
    "prima afp":     {
        "prensa":      ["Prima AFP Peru pensiones"],
        "bvl":         ["Prima AFP BVL hecho importancia"],
        "regulatorio": ["Prima AFP SBS sancion"],
    },
    "integra afp":   {
        "prensa":      ["Integra AFP Peru pensiones"],
        "bvl":         ["Integra AFP BVL hecho importancia"],
        "regulatorio": ["Integra AFP SBS sancion"],
    },
    "alicorp":       {
        "prensa":      ["Alicorp Peru"],
        "bvl":         ["Alicorp BVL hecho importancia acciones"],
        "regulatorio": ["Alicorp SMV sancion"],
    },
    "ferreycorp":    {
        "prensa":      ["Ferreycorp Peru Ferreyros"],
        "bvl":         ["Ferreycorp BVL hecho importancia"],
        "regulatorio": ["Ferreycorp SMV sancion"],
    },
    "banbif":        {
        "prensa":      ["BanBif Peru banco"],
        "bvl":         ["BanBif BVL hecho importancia"],
        "regulatorio": ["BanBif SBS sancion"],
    },
    "caja piura":    {
        "prensa":      ["Caja Piura CMAC Peru"],
        "bvl":         ["Caja Piura BVL hecho importancia"],
        "regulatorio": ["Caja Piura SBS sancion multa"],
    },
    "caja huancayo": {
        "prensa":      ["Caja Huancayo CMAC Peru"],
        "bvl":         ["Caja Huancayo BVL hecho importancia"],
        "regulatorio": ["Caja Huancayo SBS sancion multa"],
    },
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "es-PE,es;q=0.9",
}

print("⏳ Cargando modelo BERT...")
modelo_bert = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
print("✅ BERT listo")

# ── Modelos Pydantic ──────────────────────────────────────────────────────────
class EmpresaRequest(BaseModel):
    empresa: str
    max_articulos: int = 30
    ventana_contexto: int = 300

class NoticiaAnalizada(BaseModel):
    titulo: str
    fuente: str
    fecha: str
    url: str
    contexto: str
    sentimiento: str
    score_riesgo: int
    score_confianza: float
    analizado_por: str
    razon_claude: str
    categoria: str          # prensa | bvl | regulatorio

class ResultadoAnalisis(BaseModel):
    empresa: str
    fecha_analisis: str
    total_menciones: int
    noticias_positivas: int
    noticias_negativas: int
    noticias_neutrales: int
    score_riesgo_promedio: float
    nivel_alerta: str
    fuentes_consultadas: list[str]
    analizadas_bert: int
    escaladas_claude: int
    noticias: list[NoticiaAnalizada]

# ═══════════════════════════════════════════════════════
# CAPA 1 — RSS
# ═══════════════════════════════════════════════════════
def buscar_google_news(query: str, categoria: str, max_items: int = 15) -> list[dict]:
    """Busca en Google News RSS y devuelve artículos etiquetados por categoría."""
    encontrados = []
    q   = query.replace(" ", "+")
    url = f"{GOOGLE_NEWS_BASE}{q}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        root  = ET.fromstring(resp.content)
        items = root.findall(".//item")
        print(f"[Google News] [{categoria}] '{query}': {len(items)} noticias")
        for item in items[:max_items]:
            titulo     = item.findtext("title", "") or ""
            link       = item.findtext("link",  "") or ""
            pub_date   = item.findtext("pubDate", "") or ""
            fuente_tag = item.find("source")
            fuente     = fuente_tag.text if fuente_tag is not None else "Google News"
            encontrados.append({
                "titulo":      titulo.strip(),
                "fuente":      fuente,
                "fecha":       pub_date[:16] if pub_date else datetime.now().strftime("%Y-%m-%d"),
                "url":         link.strip(),
                "descripcion": titulo,
                "categoria":   categoria,
            })
    except Exception as e:
        print(f"[WARN] Google News [{categoria}]: {str(e)[:60]}")
    return encontrados


def obtener_articulos_google(alias: dict, max_art: int) -> list[dict]:
    """
    Busca en 3 dimensiones para cada empresa:
      prensa      → noticias generales de medios
      bvl         → hechos de importancia en bolsa
      regulatorio → sanciones SMV/SBS
    """
    encontrados = []
    vistos      = set()
    por_cat     = max_art // 3  # distribuir equitativamente

    for categoria, terminos in alias.items():
        for termino in terminos:
            arts = buscar_google_news(termino, categoria, por_cat)
            for art in arts:
                if art["url"] not in vistos:
                    vistos.add(art["url"])
                    encontrados.append(art)

    return encontrados[:max_art]


def obtener_articulos_rss(alias, max_art: int) -> list[dict]:
    """Wrapper principal — acepta alias dict o list para compatibilidad."""
    if isinstance(alias, dict):
        return obtener_articulos_google(alias, max_art)
    # fallback: alias es lista simple
    terminos_dict = {"prensa": alias}
    return obtener_articulos_google(terminos_dict, max_art)

def extraer_texto_url(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        sopa = BeautifulSoup(resp.content, "html.parser")
        return " ".join(p.get_text() for p in sopa.find_all("p"))
    except Exception:
        return ""

def buscar_contexto(texto: str, terminos: list[str], ventana: int) -> str:
    texto_lower = texto.lower()
    for t in terminos:
        pos = texto_lower.find(t.lower())
        if pos != -1:
            return texto[max(0, pos - ventana): pos + ventana].strip()
    return texto[:ventana].strip()

# ═══════════════════════════════════════════════════════
# CAPA 2 — BERT (gratis, siempre)
# ═══════════════════════════════════════════════════════
def analizar_bert(contexto: str) -> tuple[str, int, float]:
    try:
        res       = modelo_bert(contexto[:512])[0]
        estrellas = int(res["label"][0])
        confianza = round(float(res["score"]), 4)
        if estrellas <= 2:
            return "NEGATIVO", min(40 + (2 - estrellas) * 30, 100), confianza
        elif estrellas == 3:
            return "NEUTRAL", 10, confianza
        else:
            return "POSITIVO", 0, confianza
    except Exception:
        return "NEUTRAL", 10, 0.5

# ═══════════════════════════════════════════════════════
# CAPA 3 — Claude (solo si score >= UMBRAL_CLAUDE)
# ═══════════════════════════════════════════════════════
def analizar_claude(titulo: str, contexto: str, empresa: str, score_bert: int) -> tuple:
    if not ANTHROPIC_API_KEY:
        return None, None, "Claude no configurado — usando resultado BERT"

    prompt = f"""Eres analista de riesgo financiero de la SBS Perú.

Analiza esta noticia sobre "{empresa}":
TITULAR: {titulo}
CONTEXTO: {contexto[:600]}
SCORE BERT PREVIO: {score_bert}/100

Responde SOLO en este JSON exacto sin texto adicional:
{{"sentimiento": "NEGATIVO", "score_riesgo": 65, "razon": "explicacion ejecutiva máximo 20 palabras"}}

Escala de score:
- NEGATIVO 60-100: fraude, quiebra, sanción SBS, pérdidas graves, demanda judicial
- NEGATIVO 30-59: mora elevada, caída utilidades, problemas operativos
- NEUTRAL  10-25: cambios directivos, eventos informativos sin impacto claro
- POSITIVO 0-5:   utilidades récord, expansión, inversión nueva

Responde en español. La razón debe ser ejecutiva y directa."""

    try:
        resp   = requests.post(ANTHROPIC_URL, headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }, json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 150,
            "messages": [{"role": "user", "content": prompt}],
        }, timeout=20)
        texto  = resp.json()["content"][0]["text"].strip().replace("```json","").replace("```","")
        result = json.loads(texto)
        return result.get("sentimiento","NEUTRAL"), int(result.get("score_riesgo", score_bert)), result.get("razon","")
    except Exception as e:
        return None, None, f"Error Claude: {str(e)[:60]}"

# ═══════════════════════════════════════════════════════
# PIPELINE HÍBRIDO
# ═══════════════════════════════════════════════════════
def analizar_articulo_hibrido(art: dict, terminos: list, ventana: int, empresa: str) -> NoticiaAnalizada:
    texto    = extraer_texto_url(art["url"])
    if len(texto.strip()) < 100:
        texto = art.get("descripcion", art["titulo"])
    contexto = buscar_contexto(texto, terminos, ventana)

    # CAPA 2: BERT
    sent_bert, score_bert, confianza = analizar_bert(contexto)
    print(f"[BERT] '{art['titulo'][:50]}' → {sent_bert} score={score_bert}")

    sent_final  = sent_bert
    score_final = score_bert
    razon       = ""
    motor       = "BERT"

    # CAPA 3: Claude solo si score >= umbral
    if score_bert >= UMBRAL_CLAUDE:
        print(f"[ESCALA→CLAUDE] score={score_bert}")
        sc, sr, razon = analizar_claude(art["titulo"], contexto, empresa, score_bert)
        if sc is not None:
            sent_final  = sc
            score_final = sr
            motor       = "BERT+Claude"

    return NoticiaAnalizada(
        titulo=art["titulo"], fuente=art["fuente"], fecha=art["fecha"],
        url=art["url"], contexto=contexto[:400],
        sentimiento=sent_final, score_riesgo=score_final,
        score_confianza=confianza, analizado_por=motor, razon_claude=razon,
        categoria=art.get("categoria", "prensa"),
    )

def obtener_analisis(empresa: str, max_art: int, ventana: int) -> ResultadoAnalisis:
    alias_raw = EMPRESAS_ALIAS.get(empresa.lower(), None)
    if alias_raw is None:
        # empresa libre: construir alias generico con 3 dimensiones
        alias_raw = {
            "prensa":      [f"{empresa} Peru"],
            "bvl":         [f"{empresa} BVL hecho importancia"],
            "regulatorio": [f"{empresa} SBS SMV sancion"],
        }
    articulos = obtener_articulos_rss(alias_raw, max_art)

    if not articulos:
        return ResultadoAnalisis(
            empresa=empresa, fecha_analisis=datetime.now().strftime("%Y-%m-%d %H:%M"),
            total_menciones=0, noticias_positivas=0, noticias_negativas=0,
            noticias_neutrales=0, score_riesgo_promedio=0, nivel_alerta="SIN DATOS",
            fuentes_consultadas=list(RSS_FEEDS.keys()), analizadas_bert=0,
            escaladas_claude=0, noticias=[]
        )

    noticias = []
    pos = neg = neu = scores = bert_c = claude_c = 0
    # extraer todos los terminos para busqueda de contexto
    if isinstance(alias_raw, dict):
        terminos = [t for lista in alias_raw.values() for t in lista]
    else:
        terminos = alias_raw

    for art in articulos:
        n = analizar_articulo_hibrido(art, terminos, ventana, empresa)
        noticias.append(n)
        if n.sentimiento == "POSITIVO": pos += 1
        elif n.sentimiento == "NEGATIVO": neg += 1
        else: neu += 1
        scores += n.score_riesgo
        if n.analizado_por == "BERT+Claude": claude_c += 1
        else: bert_c += 1

    total      = len(noticias)
    score_prom = round(scores / total, 2) if total else 0
    nivel      = "ALTO" if score_prom >= 40 else "MEDIO" if score_prom >= 15 else "BAJO"

    print(f"\n[FIN] {empresa}: {total} noticias | BERT={bert_c} | Claude={claude_c} | Alerta={nivel}")

    return ResultadoAnalisis(
        empresa=empresa, fecha_analisis=datetime.now().strftime("%Y-%m-%d %H:%M"),
        total_menciones=total, noticias_positivas=pos, noticias_negativas=neg,
        noticias_neutrales=neu, score_riesgo_promedio=score_prom, nivel_alerta=nivel,
        fuentes_consultadas=list(set(a["fuente"] for a in articulos)),
        analizadas_bert=bert_c, escaladas_claude=claude_c, noticias=noticias,
    )

# ═══════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════
@app.post("/analizar-empresa", response_model=ResultadoAnalisis)
def analizar_empresa(req: EmpresaRequest):
    return obtener_analisis(req.empresa, req.max_articulos, req.ventana_contexto)

@app.get("/exportar-negativas/{empresa}")
def exportar_negativas(empresa: str):
    r = obtener_analisis(empresa, 30, 300)
    neg = [n for n in r.noticias if n.sentimiento == "NEGATIVO"]
    if not neg: raise HTTPException(404, "Sin noticias negativas.")
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Fecha","Título","Fuente","Sentimiento","Score","Confianza","Motor","Razón Claude","Contexto","URL"])
    for n in neg:
        w.writerow([n.fecha,n.titulo,n.fuente,n.sentimiento,n.score_riesgo,n.score_confianza,n.analizado_por,n.razon_claude,n.contexto,n.url])
    out.seek(0)
    return StreamingResponse(iter([out.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=negativas_{empresa}_{datetime.now().strftime('%Y%m%d')}.csv"})

@app.get("/exportar-todas/{empresa}")
def exportar_todas(empresa: str):
    r = obtener_analisis(empresa, 30, 300)
    if not r.noticias: raise HTTPException(404, "Sin noticias.")
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Fecha","Título","Fuente","Sentimiento","Score","Confianza","Motor","Razón Claude","Contexto","URL"])
    for n in r.noticias:
        w.writerow([n.fecha,n.titulo,n.fuente,n.sentimiento,n.score_riesgo,n.score_confianza,n.analizado_por,n.razon_claude,n.contexto,n.url])
    out.seek(0)
    return StreamingResponse(iter([out.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=todas_{empresa}_{datetime.now().strftime('%Y%m%d')}.csv"})

@app.get("/empresas")
def listar_empresas():
    return {"empresas": list(EMPRESAS_ALIAS.keys()), "fuentes": list(RSS_FEEDS.keys()), "umbral_claude": UMBRAL_CLAUDE}

@app.get("/health")
def health():
    return {
        "status": "ok", "version": "3.0", "arquitectura": "hibrida",
        "capa_1": "RSS feeds peruanos (sin API key)",
        "capa_2": "BERT local (gratis, siempre activo)",
        "capa_3": f"Claude API (solo noticias con score >= {UMBRAL_CLAUDE})",
        "claude_disponible": bool(ANTHROPIC_API_KEY),
        "medios": list(RSS_FEEDS.keys()),
    }

@app.get("/")
def root():
    return {"mensaje": "SBS Escucha Social v3.0 ✅ — RSS → BERT → Claude (híbrido)"}