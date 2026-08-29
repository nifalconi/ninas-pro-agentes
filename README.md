# Niñas Pro — IA Aplicada: Agentes

Herramientas del curso. Los notebooks las descargan al empezar cada sesión.

## Uso desde Colab

```python
import urllib.request

urllib.request.urlretrieve(
    "https://nifalconi.github.io/ninas-pro-agentes/ninaspro.py",
    "ninaspro.py")

from ninaspro import catalogo_gratis, probar_modelos, usar_modelos, usar_plan_b
```

## Qué hay acá

| Función | Para qué |
|---|---|
| `catalogo_gratis()` | Lista los modelos gratis que hay hoy en OpenRouter. No gasta consultas. |
| `probar_modelos(client, candidatos)` | Prueba cada modelo, mide cuánto tarda y marca cuáles no responden. |
| `usar_modelos(client, modelos)` | Deja listo el agente con el primero y el resto como respaldo. |
| `usar_plan_b(clave)` | Cambia a una clave con saldo y a un modelo de pago. |

## Por qué vive acá y no en el notebook

Los modelos gratis de OpenRouter se retiran seguido. Si la lista queda escrita
dentro del notebook, cada vez que un modelo muere hay que repartir un notebook
nuevo. Acá se arregla un archivo y los notebooks ya entregados quedan al día,
porque lo bajan de nuevo en cada sesión.
