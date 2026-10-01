# Comparación: detector YOLOv8m (modelo 1) vs clasificador YOLOv8m-cls (modelo 2)

**Veredicto: el modelo 1 es mejor en todo lo que importa para la banda. El modelo 2 no
debe usarse en producción, ni solo ni combinado.** Reproducible con
`python colab/comparar_modelos.py`.

---

## 1. Cara a cara sobre los mismos huevos

Terreno común: test del dataset `huevos` (145 imágenes, **183 huevos**: 140 rotos, 43 sanos).
Cada modelo con sus umbrales calibrados (detector: política del backend; clasificador:
`TAU_ROTO=0.37`, `TAU_DUDA=0.21`, calibrados en su val).

| estrategia | roto → **empaque** | roto no descartado | sano botado | sano a revisión |
|---|---|---|---|---|
| **1. Detector solo (producción)** | **0/140 (0 %)** | 9/140 | **0/43** | 2/43 |
| 2. Clasificador, recorte perfecto (su techo) | 37/140 (26 %) | 41/140 | 6/43 | 6/43 |
| 3. Dos etapas (detector recorta, clasificador juzga) | 31/140 (22 %) | 43/140 | 6/43 | 6/43 |
| 4. Combinado: dañado si *cualquiera* sospecha | 0/140 | 8/140 | 6/43 | 8/43 |
| 5. Combinado: si no coinciden → revisión | 0/140 | 44/140 | 0/43 | 14/43 |

- El clasificador manda **1 de cada 4 huevos rotos al empaque**.
- Combinarlos no suma: la estrategia 4 atrapa **un** roto más que el detector solo
  (132 vs 131) a cambio de botar el 14 % de los sanos; la 5 no bota sanos pero
  triplica el trabajo humano.

### No es un problema de umbral

AUC de p(Roto) separando rotos de sanos, **sin umbral**, mismos 183 huevos:

| | AUC |
|---|---|
| Detector | **0.975** |
| Clasificador (con recorte perfecto) | 0.833 |
| Diferencia | **+0.143**, IC95 bootstrap [+0.083, +0.213] |

El intervalo no cruza el cero. Y 25 de los 140 rotos reciben p(Roto) < 0.05: el
clasificador está **seguro** de que están bien. Ningún umbral los rescata.

---

## 2. Robustez sobre los mismos huevos

| prueba | detector | clasificador (recorte perfecto) |
|---|---|---|
| Raya de lapicero en 43 sanos → DAÑADO | 24 (56 %) | **37 (86 %)** |
| Desenfoque r=3 en 140 rotos → BUENO | **3** (+63 a revisión) | **116 (83 %)** |
| Sobreexposición 2.4× en 140 rotos → no detectado | **70 (50 %)** | — (recorte dado) |
| Sobreexposición 2.4× en 140 rotos → BUENO | 8 | 60 (43 %) |

- **La hipótesis de que la clase "Sucio" enseñaría que "mancha ≠ fisura" queda refutada**:
  con la raya, el clasificador bota el 86 % de los sanos y **ninguno** sale como
  "bueno (sucio)".
- Con desenfoque, la compuerta de nitidez del backend salva al detector (63 a revisión,
  solo 3 al empaque); el clasificador manda al empaque el 83 %.
- La sobreexposición es mala para los dos: el detector **pierde la mitad de los huevos**.
  La iluminación no es un detalle del montaje: es un requisito.

> **Corrección sobre el detector.** En [`ANALISIS_CAMPO.md`](ANALISIS_CAMPO.md) reporté que
> la raya voltea 6 de 20 sanos (30 %). Esa medida usó 20 imágenes. Sobre los **43** sanos del
> test, la cifra es **24/43 (56 %)**. Cita la de 56 %: es la muestra más amplia.

---

## 3. Por qué el modelo 2 salió así

**Tenía muchos menos datos de los que aparentaba.**

| | detector | clasificador |
|---|---|---|
| Imágenes de train | 678 | 1.451 |
| **Grupos independientes en train** | **678** | **220** |
| Frames de video en train | 0 | 991 (68 %) de 14 videos |
| "Sucio" de train que sale de videos | — | **387 de 387 (100 %)** |

El dataset anuncia 5.305 imágenes, pero tras quitar copias aumentadas quedaron 2.063
nombres distintos, y de esos el 48 % son frames de 16 videos (~100 frames cada uno). El
modelo aprendió qué es un huevo **sucio** mirando unos pocos huevos filmados, y lo
evaluaron fotos.

**Síntomas en el entrenamiento** (150 épocas, 22 min):
- Exactitud en val estancada en **~0.70 de promedio** en todas las ventanas de 20
  épocas; el "mejor" 0.869 (época 81) es un pico aislado.
- `val/loss` sube de 1.67 a 2.75 mientras `train/loss` baja a 0.009: memorización.

**En su propio test** (306 imágenes del dataset CDD): exactitud 87.3 %, 4 % de rotos al
empaque, pero **17 % de los huevos buenos botados**. Inaceptable en una planta.

---

## 4. Límites de esta evaluación

- El terreno común es el dominio del detector (fotos del dataset `huevos`): **favorece al
  detector**. La prueba simétrica —el detector sobre el test de CDD— no se pudo correr
  (las imágenes de CDD solo están en Colab). Aun así, los números del clasificador en su
  propio test ya son malos (17 % de buenos botados).
- 43 huevos sanos es una muestra chica: las cifras del lado "sano" tienen intervalos amplios.
- **Pregunta abierta sobre el split**: 5.305 imágenes → 2.063 nombres distintos. Si son la
  misma foto subida varias veces por el autor, quitarlas fue correcto. Si son huevos
  distintos con el mismo nombre de archivo, el split descartó datos válidos y un modelo 2
  reentrenado podría mejorar. Se resuelve con una celda de diagnóstico (ver abajo).

---

## 5. Qué hacer

1. **Mantener el modelo 1 en producción.** No integrar el modelo 2.
2. En la sustentación, presentar el modelo 2 como **resultado negativo documentado**: un
   dataset "más grande" no sirve si son 14 videos; y la hipótesis de la clase "Sucio" se
   probó y se refutó. Eso es rigor, no fracaso.
3. La mejora que sí vale la pena sigue siendo la de datos para el modelo 1 (negativos
   duros con marcas, sellos y rayas; huevos abiertos sin fisura lineal), más
   **iluminación controlada** en el montaje.

### Diagnóstico de la pregunta abierta (celda para Colab, tras correr las celdas 1–5)

```python
from collections import defaultdict
por_base = defaultdict(list)
for it in todos:
    por_base[it["base"]].append(it)
mezcladas = [b for b, v in por_base.items() if len({it["clase_orig"] for it in v}) > 1]
print(f"{len(todos)} archivos | {len(por_base)} nombres distintos")
print(f"nombres con archivos en CLASES DISTINTAS: {len(mezcladas)}  <- si es alto, son huevos distintos")
print("ejemplos:", mezcladas[:10])
```

Si ese número es alto (cientos), el mismo nombre se usó para huevos distintos y hay que
cambiar la regla de "copia aumentada" para exigir además similitud visual.
