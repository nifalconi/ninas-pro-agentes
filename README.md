# Niñas Pro — IA Aplicada: Agentes

Herramientas del curso. Los notebooks las descargan al empezar cada sesión.

## Uso desde Colab

```python
import importlib
import urllib.request

urllib.request.urlretrieve(
    "https://nifalconi.github.io/ninas-pro-agentes/ninaspro.py",
    "ninaspro.py")

import ninaspro
importlib.reload(ninaspro)   # descarta la versión que quedó cargada de antes
from ninaspro import catalogo_gratis, probar_modelos, usar_modelos, usar_plan_b
```

El `importlib.reload` no es opcional: `urllib.request.urlretrieve` sobrescribe
el archivo, pero `from ninaspro import ...` devuelve el módulo que Python ya
tiene en memoria. Sin el reload, una sesión abierta sigue con la versión vieja
por más que bajes la nueva.

---

## `catalogo_gratis()`

Lista los modelos gratis de hoy, del más estable al menos. No gasta consultas:
lee el catálogo público y el uptime, que no son llamadas al modelo.

```
Hay 18 modelos GRATIS en OpenRouter hoy.
Ordenados por lo estable que estuvo cada uno en las últimas 24 horas.

MODELO                                              FUNCIONÓ    CONTEXTO
------------------------------------------------------------------------
dots-studio/dots-3-note-preview:free                  100.0%     512,000
inclusionai/ling-3.0-flash-fin:free                   100.0%     262,144
poolside/laguna-s-2.1:free                            100.0%     262,144
thinkingmachines/inkling-small:free                   100.0%   1,048,576
minimax/minimax-m3:free                                99.8%   1,048,576
...
nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free     86.6%     256,000
nvidia/nemotron-3.5-lightning:free                     51.5%   1,000,000

Esta celda no gasta consultas: solo lee datos públicos.
Un porcentaje bajo significa que ese modelo se cae seguido.
```

Devuelve la lista de ids, ordenada igual que la tabla.

La velocidad no sale de acá porque OpenRouter no la publica: los campos
`latency_last_30m` y `throughput_last_30m` de su API vienen `null` en todos los
modelos. Para saber cuánto tardan hay que llamarlos, que es lo que hace la
función siguiente.

---

## `probar_modelos(client, candidatos)`

Los prueba de verdad y los ordena por velocidad. Gasta **una consulta por
modelo**, así que conviene pasarle una lista corta.

```
Probando 4 modelos. Gasta 4 de tus 50 consultas diarias.

MODELO                                          SIRVE      SEG  DETALLE
--------------------------------------------------------------------------
poolside/laguna-s-2.1:free                      SÍ         2.4  razona antes → "Hola."
minimax/minimax-m3:free                         SÍ         2.6  directo → "hola"
inclusionai/ling-3.0-flash-fin:free             SÍ         3.0  razona antes → "Hola"
google/gemma-4-31b-it:free                      NO         2.4  saturado ahora (429)

>>> Te sirven 3 de 4, los más rápidos primero.
>>> Copia los que quieras a la lista MODELOS de la celda siguiente:

MODELOS = [
    "poolside/laguna-s-2.1:free",
    "minimax/minimax-m3:free",
    "inclusionai/ling-3.0-flash-fin:free",
]
```

La columna DETALLE distingue dos cosas que importan:

* **directo** — contesta y ya.
* **razona antes** — "piensa" antes de escribir, gastando tokens que no se ven
  en la respuesta. Funciona, pero es más lento y necesita `max_tokens` holgado.
* **NO / se quedó sin espacio razonando** — es un modelo de razonamiento al que
  no le alcanzó el presupuesto y devolvió texto vacío.

Es `async`, así que va con `await`:

```python
await probar_modelos(client, ["minimax/minimax-m3:free", "poolside/laguna-s-2.1:free"])
```

---

## `usar_modelos(client, modelos, temperature=0.7, max_tokens=2048)`

Arma las dos variables que usan todos los ejercicios, y devuelve las dos juntas.

```python
MODELO, SETTINGS = usar_modelos(client, MODELOS)
```

```
✓ Modelo principal: minimax/minimax-m3:free
  Si falla, OpenRouter prueba: minimax/minimax-m2.7:free
  temperature=0.7  max_tokens=2048
```

### Qué recibe

| Parámetro | Obligatorio | Por defecto | Para qué |
|---|---|---|---|
| `client` | sí | — | El `AsyncOpenAI` apuntando a OpenRouter |
| `modelos` | sí | — | Lista de ids. El **primero** se usa; el resto queda de respaldo |
| `temperature` | no | `0.7` | Qué tan creativo. `0` responde casi siempre igual, `1` se suelta |
| `max_tokens` | no | `2048` | Cuánto puede escribir como máximo |

### `modelos` es una lista, y el orden importa

El primero es el que usa tu agente. **Los demás son el respaldo**: si el primero
se cae, OpenRouter salta solo al siguiente sin que tengas que hacer nada.

```python
MODELOS = [
    "minimax/minimax-m3:free",              # el que usa tu agente
    "minimax/minimax-m2.7:free",            # respaldo, si el de arriba falla
    "inclusionai/ling-3.0-flash-fin:free",  # respaldo, si fallan los dos
]
```

Pasarle uno solo también funciona, pero te quedás sin red:

```python
MODELOS = ["minimax/minimax-m3:free"]   # si este se cae, se cayó la clase
```

Con esa lista, las dos formas de llamarla son válidas:

```python
# corta: usa 0.7 y 2048
MODELO, SETTINGS = usar_modelos(client, MODELOS)

# larga: los pone a la vista para poder cambiarlos
MODELO, SETTINGS = usar_modelos(
    client,
    MODELOS,
    temperature=0.2,    # respuestas más parecidas entre sí
    max_tokens=2048,
)
```

Y así queda repartido lo que devuelve:

```
MODELO   -> minimax/minimax-m3:free
SETTINGS -> temperature, max_tokens, y los otros dos como respaldo
```

En el notebook del curso va la forma larga a propósito: las diapositivas 11 y 12
explican los dos parámetros, y si quedan implícitos las alumnas nunca los ven.

**No bajes `max_tokens`.** Medido con `ling-3.0-flash-fin:free` pidiéndole una
explicación de matemáticas: con `800` devuelve `finish_reason=length` y la
respuesta **vacía**, porque gasta el presupuesto entero razonando. Con `2048`
contesta 2.206 caracteres.

### Cuándo salta el respaldo, y cuándo no

Por dentro, `modelos[1:]` viaja a OpenRouter en `extra_body={"models": ...}`.
El salto lo decide OpenRouter, no el notebook. Probado contra los cuatro casos:

| Falla del principal | ¿Salta? |
|---|---|
| Modelo retirado, 404 "no endpoints" | sí |
| Retirado con prefijo de vendor | sí |
| Saturado, 429 | sí |
| Id mal escrito, 400 | **no** |

Que un error de tipeo falle fuerte es lo correcto: no querés que un `minmax`
mal escrito use otro modelo en silencio.

---

## `usar_plan_b(clave)`

Cambia a una clave con saldo y a modelos de pago. Para cuando la alumna se queda
sin cupo diario, su clave falla, o todos los gratis están saturados a la vez.

```python
client, MODELO, SETTINGS = usar_plan_b("sk-or-v1-...")
```

Devuelve **tres** cosas, no dos: el cliente también cambia. Y los agentes ya
creados siguen apuntando al modelo anterior, así que hay que volver a ejecutar
la celda que los crea.

Su propia cadena de respaldo está en `MODELOS_PLAN_B`, y termina en
`openai/gpt-4o-mini` — el más veterano de los tres, para que no se quede sin
piso cuando los de arriba se retiren.

---

## Por qué vive acá y no en el notebook

Dos razones.

**Los modelos gratis se retiran seguido.** Si la lista queda escrita dentro del
notebook, cada vez que uno muere hay que repartir un notebook nuevo. Acá se
arregla un archivo y los notebooks ya entregados quedan al día, porque lo bajan
de nuevo en cada sesión.

**El notebook debe mostrar solo lo que se enseña ese día.** La Clase 02 trata de
`Agent` y `Runner`, no de `asyncio.gather` ni de formatear tablas. Esa plomería
vive acá afuera a propósito.

## Qué toca cambiar cuando algo se rompe

| Síntoma | Dónde mirar |
|---|---|
| El modelo por defecto ya no existe | La lista `MODELOS` en el notebook, y `MODELOS_PLAN_B` acá |
| El agente responde vacío | `MAX_TOKENS` |
| El Plan B da 402 | La clave se quedó sin saldo: `openrouter.ai/settings/keys` |
