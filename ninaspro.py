"""
Herramientas del curso "IA Aplicada: Agentes" de Niñas Pro.

Estas funciones existen para que las alumnas puedan elegir un modelo que
funcione sin tener que leer código de asyncio en la Clase 02. La lógica vive
acá afuera a propósito: el notebook debe mostrar solo lo que se está
enseñando ese día.

Uso desde el notebook:

    import importlib, urllib.request
    urllib.request.urlretrieve(
        "https://nifalconi.github.io/ninas-pro-agentes/ninaspro.py", "ninaspro.py")
    import ninaspro
    importlib.reload(ninaspro)   # sin esto, una sesion ya abierta sigue con la version vieja
    from ninaspro import catalogo_gratis, probar_modelos, usar_modelos, usar_plan_b

    modelos = catalogo_gratis()
    await probar_modelos(client, ["minimax/minimax-m3:free", ...])
    MODELO, SETTINGS = usar_modelos(client, ["minimax/minimax-m3:free", ...],
                                    temperature=0.7, max_tokens=2048)
"""

import asyncio
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from agents import ModelSettings, OpenAIChatCompletionsModel, set_default_openai_client
from openai import AsyncOpenAI

CATALOGO_URL = "https://openrouter.ai/api/v1/models?sort=pricing-low-to-high"
BASE_URL = "https://openrouter.ai/api/v1"
# El Plan B corre sobre una clave con saldo, así que puede usar modelos de pago.
# Van en orden de preferencia y OpenRouter salta al siguiente si uno falla; el
# último es el más veterano, para que la cadena no se quede sin piso cuando los
# de arriba se retiren.
MODELOS_PLAN_B = [
    "mistralai/mistral-small-3.2-24b-instruct",
    "amazon/nova-lite-v1",
    "openai/gpt-4o-mini",
]

# Los modelos de razonamiento gastan tokens "pensando" antes de escribir, y ese
# gasto no se ve en la respuesta. Medido con ling-3.0-flash-fin:free sobre una
# explicación de matemáticas: con 800 se queda sin espacio y devuelve la
# respuesta vacía; con 2048 contesta 2.206 caracteres.
MAX_TOKENS = 2048

_MOTIVOS = {
    "RateLimitError": "saturado ahora (429)",
    "PermissionDeniedError": "bloqueado (403)",
    "NotFoundError": "ya no existe (404)",
    "APIStatusError": "necesita créditos (402)",
    "TimeoutError": "muy lento (>30s)",
}


def _uptime(mid):
    """Qué porcentaje del último día respondió este modelo, según OpenRouter."""
    url = f"{CATALOGO_URL.split('?')[0]}/{urllib.parse.quote(mid, safe='/')}/endpoints"
    try:
        endpoints = json.load(urllib.request.urlopen(url, timeout=15))["data"]["endpoints"]
        vistos = [e["uptime_last_1d"] for e in endpoints if e.get("uptime_last_1d") is not None]
        return round(sum(vistos) / len(vistos), 1) if vistos else None
    except Exception:
        return None


def catalogo_gratis():
    """Muestra los modelos gratis que hay hoy en OpenRouter, del más estable al menos.

    No gasta consultas: el catálogo y el uptime son datos públicos, no llamadas
    al modelo. La velocidad no sale de acá porque OpenRouter no la publica; para
    eso hay que probarlos con probar_modelos().
    """
    datos = json.load(urllib.request.urlopen(CATALOGO_URL))["data"]
    gratis = [m for m in datos if m["id"].endswith(":free")]

    with ThreadPoolExecutor(max_workers=8) as pool:
        uptimes = dict(zip([m["id"] for m in gratis],
                           pool.map(_uptime, [m["id"] for m in gratis])))
    gratis.sort(key=lambda m: -(uptimes[m["id"]] if uptimes[m["id"]] is not None else -1))

    print(f"Hay {len(gratis)} modelos GRATIS en OpenRouter hoy.")
    print("Ordenados por lo estable que estuvo cada uno en las últimas 24 horas.\n")
    print(f"{'MODELO':<50}{'FUNCIONÓ':>10}{'CONTEXTO':>12}")
    print("-" * 72)
    for m in gratis:
        up = uptimes[m["id"]]
        print(f"{m['id']:<50}{(f'{up}%' if up is not None else 's/d'):>10}"
              f"{m['context_length']:>12,}")

    print("\nEsta celda no gasta consultas: solo lee datos públicos.")
    print("Un porcentaje bajo significa que ese modelo se cae seguido.")
    return [m["id"] for m in gratis]


async def _probar_uno(client, mid):
    inicio = time.time()
    try:
        respuesta = await asyncio.wait_for(
            client.chat.completions.create(
                model=mid,
                messages=[{"role": "user", "content": "Di 'hola' y nada más."}],
                max_tokens=300,
            ),
            timeout=30,
        )
    except Exception as error:
        motivo = _MOTIVOS.get(type(error).__name__, type(error).__name__)
        return mid, "NO", round(time.time() - inicio, 1), motivo

    segundos = round(time.time() - inicio, 1)
    mensaje = respuesta.choices[0].message
    texto = (mensaje.content or "").strip()
    if not texto:
        return mid, "NO", segundos, "se quedó sin espacio razonando"

    estilo = "razona antes" if getattr(mensaje, "reasoning", None) else "directo"
    return mid, "SÍ", segundos, f'{estilo} → "{texto[:22]}"'


async def probar_modelos(client, candidatos):
    """Prueba cada modelo y los ordena por velocidad.

    Gasta una consulta por modelo. Devuelve los que respondieron, para que la
    alumna pueda copiarlos a su lista.
    """
    print(f"Probando {len(candidatos)} modelos. "
          f"Gasta {len(candidatos)} de tus 50 consultas diarias.\n")

    resultados = await asyncio.gather(*[_probar_uno(client, m) for m in candidatos])
    resultados.sort(key=lambda r: (r[1] != "SÍ", r[2]))
    sirven = [mid for mid, estado, _, _ in resultados if estado == "SÍ"]

    print(f"{'MODELO':<48}{'SIRVE':<8}{'SEG':>6}  DETALLE")
    print("-" * 92)
    for mid, estado, segundos, detalle in resultados:
        print(f"{mid:<48}{estado:<8}{segundos:>6}  {detalle}")

    print(f"\n>>> Te sirven {len(sirven)} de {len(candidatos)}, los más rápidos primero.")
    print(">>> Copia los que quieras a la lista MODELOS de la celda siguiente:\n")
    print("MODELOS = [")
    for mid in sirven[:3]:
        print(f'    "{mid}",')
    print("]")
    return sirven


def usar_modelos(client, modelos, temperature=0.7, max_tokens=MAX_TOKENS):
    """Deja listo el agente con el primer modelo y el resto como respaldo.

    temperature y max_tokens quedan como parámetros para que se puedan cambiar
    desde el notebook: son los dos que se explican en clase.

    El respaldo lo resuelve OpenRouter: si el primero falla, prueba el
    siguiente sin que el notebook tenga que reintentar.
    """
    modelo = OpenAIChatCompletionsModel(model=modelos[0], openai_client=client)
    settings = ModelSettings(
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"models": modelos[1:]},
    )

    print(f"✓ Modelo principal: {modelos[0]}")
    if modelos[1:]:
        print(f"  Si falla, OpenRouter prueba: {', '.join(modelos[1:])}")
    print(f"  temperature={temperature}  max_tokens={max_tokens}")
    return modelo, settings


def usar_plan_b(clave):
    """Cambia a la clave de la profe y a un modelo de pago.

    Devuelve un cliente nuevo, así que hay que reasignar las tres variables.
    Los agentes ya creados siguen apuntando al modelo anterior: hay que volver
    a ejecutar la celda que los crea.
    """
    client = AsyncOpenAI(base_url=BASE_URL, api_key=clave)
    set_default_openai_client(client)

    modelo = OpenAIChatCompletionsModel(model=MODELOS_PLAN_B[0], openai_client=client)
    settings = ModelSettings(
        temperature=0.7,
        max_tokens=MAX_TOKENS,
        extra_body={"models": MODELOS_PLAN_B[1:]},
    )

    print(f"✓ Plan B activado con {MODELOS_PLAN_B[0]}.")
    print("  Ahora vuelve a ejecutar la celda donde creaste tu agente,")
    print("  y sigue desde ahí. (Tu agente anterior quedó con el modelo viejo.)")
    return client, modelo, settings
