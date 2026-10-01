# Comparación: modelo 1 (YOLOv8m, solo `huevos`) vs v3 (YOLOv8l, 8 fuentes)

**Veredicto: la v3 es mejor para la banda y debe reemplazar al modelo 1.** En el benchmark
empatan, y ese benchmark favorece al modelo 1. Lo que decide es otra cosa: la v3 corrige las fallas
medidas en campo y no cae en el atajo del modelo 1, que **deja pasar 1 de cada 5 rotos cuando la foto
es de celular**. Tiene una debilidad concreta: bota huevos **sanos marrones** con brillo o sobre fondos
raros. Se controla con una banda de revisión y con fotos propias. Cuesta 1,6× más tiempo por foto.

Reproducible con `v3/comparar_v1_v3.py` (cara a cara, robustez, integridad) y
`v3/prueba_contexto.py` (atajo de contexto). Resultados crudos en `v3/entrenamientov3/*.json`.

---

## 1. El entrenamiento salió sano

- **170 épocas en 6,6 h, sin desconexiones** (140 s por época: más rápido que lo estimado).
- La regla funcionó como se diseñó: **meseta detectada en la época 150** y fase final de 20 épocas,
  sin mosaico y con el LR bajando de 5,1e-4 a 1,0e-5. **La mejor época (163, mAP50-95 de val 0,815)
  salió dentro de la fase final**: cortar en seco en la 150 habría dejado ~2 puntos en la mesa.
- Sin sobreajuste: las pérdidas de validación nunca suben. Las métricas de val oscilan ±2–3 puntos
  entre épocas (val multi-fuente de 362 fotos), por eso importa que `best.pt` sea el máximo.
- Datos reales: **3.849 originales** (train 2.950 / val 362 / test 357 = 80,4 / 9,9 / 9,7) más
  el benchmark (180), y **400 imágenes de negativos sintéticos** solo en train, con 408 marcas
  (162 sellos, 199 rayas, 47 de suciedad).
  Las 8 aumentaciones fotométricas quedaron registradas en `args.yaml`.
- La auditoría hizo su trabajo:
  - **360a quedó fuera entera**: en esa versión solo están etiquetados los rotos
    (`white/brown cracked egg`). Sus sanos habrían entrado como fondo, que es justo lo que se quería evitar.
  - `carlos` (351 → 18) y `cracked` (76 → 20) resultaron ser **copias** de otras fuentes.
  - `aau` se detectó como convención "región" (encierra la grieta) y se convirtió.

## 2. Integridad y compatibilidad

| verificación | resultado |
|---|---|
| ONNX vs PyTorch (8 fotos) | diferencia máx. de probabilidad **2,3e-06**, cajas 0,001 px |
| formato para el backend | salida `[1, 6, 8400]`, clases `{0: 'Roto', 1: 'Sano'}`: **no hay que tocar código** |
| `best.pt` | ultralytics 8.4.157, época 163, mAP50-95 de val 0,8153 (coincide con `results.csv`) |
| el ONNX del backend | es el del entrenamiento del modelo 1 (mismo sha256) |

## 3. Fuga hacia el benchmark

- Las **145 fotos del test de `huevos` quedaron todas en el benchmark** de la v3: ninguna en su
  train/val/test. Ninguna otra fuente usa esos nombres.
- El benchmark de la v3 tiene 180 fotos, no 145. Las otras 35 son 34 de `jqutx` y 1 de `huevos`
  (`not_damaged_51`), que la protección agrupó por parecido visual. **`jqutx` contiene copias
  renombradas de tus fotos** (`cracked_507`…). Las exactas se eliminaron y las cercanas se pusieron
  en cuarentena.
- **Riesgo residual, no verificable desde aquí**: una copia de `jqutx` muy alterada (recortada,
  rotada en otro ángulo) podría estar en el train de la v3. Para descartarlo hacen falta las imágenes de
  `jqutx`. Mitiga que la v3 **no** saque ventaja en el benchmark (sección 4), que es donde se notaría.
- Por eso la comparación usa **las 145 fotos originales**, que ninguno de los dos vio.

## 4. Cara a cara en `huevos/test` (145 fotos, 183 huevos: 140 rotos, 43 sanos)

Mismo código del backend para los dos, mismo emparejamiento y ruteo que el notebook. Cada modelo con
sus umbrales: modelo 1 `0.35 / 0.85` (producción), v3 `0.07 / 0.75` (calibrados en su val).

**Detección: empate.**

| | mAP50 | mAP50-95 | mAP50-95 con `model.val()` de ultralytics |
|---|---|---|---|
| modelo 1 | 0,977 | 0,924 | 0,919 |
| v3 | 0,971 | 0,929 | 0,922 |
| v3 − modelo 1 | IC95 [−0,031, +0,019] | IC95 [−0,020, +0,027] | |

**Ruteo (lo que pasaría en la banda):**

| | roto → **empaque** | roto no detectado | roto a revisión | sano **botado** | sano a revisión |
|---|---|---|---|---|---|
| modelo 1 | **0/140** | 7/140 | 2/140 | **0/43** | 2/43 |
| v3 | **0/140** | 6/140 | 0/140 | **4/43** (9 %) | 1/43 |

- McNemar pareado: rotos atrapados p = 1,00; sanos aceptados p = 0,45. **Ninguna diferencia es
  estadísticamente demostrable** con 43 sanos.
- Sin umbrales, entre los huevos que ambos detectan: AUC de p(Roto) **1,000** (modelo 1) vs **0,994**
  (v3). En esta cancha, el modelo 1 separa perfecto. La sección 6 explica por qué eso no es buena señal.
- Los "rotos no detectados" de los dos son casi todos **fragmentos sueltos de cáscara** etiquetados
  como Roto (damaged_454, damaged_168, damaged_39): no son huevos enteros de una banda.

## 5. Robustez: las fallas que se vieron en campo (mismos huevos, mismas transformaciones)

| prueba | modelo 1 | v3 |
|---|---|---|
| **raya de lapicero en sanos → botados** | **25/43 (58 %)** | **5/43 (12 %)** |
| sello de tinta en sanos → botados | 2/43 | 2/43 |
| desenfoque r=3 en rotos → se escapan | 24/140 (17 %) | 10/140 (7 %) |
| **sobreexposición 2,4× en rotos → se escapan** | **79/140 (56 %)** | **41/140 (29 %)** |
| subexposición 0,35× en rotos → se escapan | 11/140 (8 %) | 7/140 (5 %) |
| JPEG calidad 20 en rotos → se escapan | 18/140 (13 %) | 8/140 (6 %) |
| JPEG calidad 20 en sanos → botados | 0/43 | 4/43 |

"Se escapan" = aceptados + no detectados (un huevo sin detectar sigue de largo por la banda).
El diferencial funciona: con la raya, p(Roto) medio de un sano pasa de **0,54 a 0,08**.
La sobreexposición fuerte sigue siendo el punto débil de ambos: la v3 **no detecta el 28 %** de los
rotos (el modelo 1, el 51 %).
Controlar la luz de la banda sigue siendo necesario.

## 6. Por qué el modelo 1 "gana" en su cancha: aprendió el tipo de foto

En `huevos`, **la fuente de la foto casi define la etiqueta**:

| fuente (train) | Roto | Sano |
|---|---|---|
| `damaged` (escenas exteriores: piso, ladrillo, materas) | 656 | **0** |
| `not_damaged` | **0** | 124 |
| `IMG` (celular, en la mano o en el mesón) | 2 | **86** |

El test repite la misma estructura, así que un modelo que reconozca **el tipo de foto** saca notas
casi perfectas sin mirar la cáscara. Para probarlo, se pegó el mismo huevo en fotos de uno y otro
contexto. Son 18 fotos por contexto × 6 huevos donantes de otras fotos = 108 montajes por celda, iguales
para los dos modelos.

| huevo pegado | en foto de… | modelo 1 se equivoca | v3 se equivoca |
|---|---|---|---|
| **roto** | **celular** (la fuente "dice" sano) | **20,4 %** se escapan | **3,7 %** |
| roto | `damaged` (la fuente "dice" roto) | 0,0 % | 0,9 % |
| sano | celular | 4,6 % botados | 13,9 % |
| sano | `damaged` | 55,6 % botados | 73,1 % |

- **El atajo peligroso es del modelo 1**: un roto evidente en una foto de celular se le escapa 1 de
  cada 5 veces solo por el contexto. Es exactamente lo que viste en campo: rotos aceptados en fotos
  tomadas con el celular. La v3 casi no cambia (+2,8 puntos contra +20,4).
- **Hay un atajo que comparten, y la v3 lo tiene más fuerte**: "escena exterior sobre el piso =
  roto". Los sanos pegados ahí quedan limpios (revisado a ojo: sin yema ni cáscaras alrededor) y
  aun así los botan. La v3 lo hereda de `huevos` y de las copias renombradas en `jqutx`. En la banda
  el fondo es constante, así que afecta poco, pero es la razón de mantener un fondo neutro y fijo.

## 7. La debilidad concreta de la v3

Los 4 sanos que la v3 bota en el test son **todos marrones y todos fotos de celular**:

- celular: 4/19 (21 %); `not_damaged`: 0/24.
- marrones: 4/18 (22 %); blancos: 0/25. En rotos, empate por color.

Uno (IMG_5152) es un huevo brillante con reflejos de ventana: la v3 le da p(Roto) = 0,91. Revisado a
resolución completa: **no tiene fisura** (la etiqueta está bien, el error es del modelo). Otro está
sobre granito moteado. En su propio test multi-fuente, la v3 bota el 14 % de los sanos.

**Mitigación medida — banda de revisión** (sensibilidad en este benchmark, no calibración):

| política v3 | rotos al empaque | rotos a revisión | sanos botados | sanos a revisión |
|---|---|---|---|---|
| calibrada: descarte si p(Roto) ≥ 0,07 | 0/140 | 0 | 4/43 | 1 |
| banda: descarte ≥ 0,35; revisión 0,07–0,35 | **0/140** | 3 | **1/43** | 4 |
| — con raya de lapicero | | | 5 → **3**/43 | 1 → 3 |

Misma seguridad, 3 sanos menos a la basura, a cambio de ~4 % de huevos a revisión humana. Requiere
un cambio pequeño en `routing.py` (un umbral más, `TAU_REVISION`), porque con dos umbrales un huevo
con p(Roto) = 0,2 y p(Sano) = 0,8 se aceptaría.

## 8. Velocidad

Camino completo de producción (decodificar, red, NMS, nitidez), 40 fotos, medido en esta PC:

| | todos los hilos | 2 hilos (como el t2.large) |
|---|---|---|
| modelo 1 | 234 ms | 872 ms |
| v3 | 566 ms | **1.400 ms** (p95 1.880) |

En el `t2.large` será algo peor (núcleos más lentos y créditos de CPU que se agotan con carga
sostenida). Para la banda simulada, a ~1 foto cada 2 s, alcanza. Si no, YOLOv8m con la misma receta
da la mitad.

## 9. Fotos reales de campo (44 fotos del celular, 21-sep)

Son las fotos que el servidor guardó en `evidencias/curaduria/` durante tus pruebas con la app.
Pasaron por el **pipeline real del backend**, con fusión de cajas duplicadas y seguimiento, y cada
modelo con su política desplegada (modelo 1: `0.35 / 0.85`; v3: `0.35 / 0.75` con banda desde 0,07).
Las etiquetas las puse yo a ojo, con acercamientos a resolución completa. Se excluyeron:

- 2 escenas con muchos huevos pequeños;
- 3 fotos de pantalla (los dos las descartan);
- 4 casos que no se pueden juzgar (foto movida, o línea que puede ser grieta o tinta).

Hojas en `v3/entrenamientov3/campo_revision/`.

| huevos | modelo 1 | v3 |
|---|---|---|
| **24 rotos → al EMPAQUE** | **4 (17 %)** | **1 (4 %)** |
| 24 rotos → descarte / revisión | 16 / 4 | 21 / 2 |
| **19 sanos → botados** | **2** | **0** |
| 19 sanos → revisión | 3 | 1 |
| **5 sanos con raya de lapicero → botados** | **4** | **0** (4 aceptados, 1 a revisión) |

- Los 4 rotos que acepta el modelo 1 son justo la falla que reportaste: huevo marrón abierto, en
  foto de celular (1a8b4037038c, 8e01f75cbbe4, d2ebb6b49c20, ef4daeeaa6b4).
- **El error de la v3** es 9b29d86af4c3: huevo con la parte de arriba abierta y un trozo de cáscara
  debajo, aceptado con p(Roto) = 0,04. El modelo 1 sí lo descarta.
- Sesgo de la muestra: son fotos en las que el modelo 1 **dudó** (solo se guardan las que tienen
  algo en revisión), así que son las difíciles para él. En fotos al azar la diferencia sería menor.
- Las rayas negras son de lapicero: **confirmado por el usuario**, que las dibujó para la prueba. En
  esos 5 huevos sanos, la v3 acierta y el modelo 1 bota 4.

## 10. Límites de esta evaluación

- **43 sanos** en el cara a cara: las diferencias de 2–4 huevos no son significativas.
- El benchmark común tiene el **sesgo de fuente** de la sección 6: favorece al modelo 1.
- Las fotos de campo (sección 9) son pocas, están sesgadas hacia las difíciles para el modelo 1 y
  las etiqueté a ojo. Confirmar las rayas de lapicero y etiquetar más fotos es lo siguiente.
- La prueba de contexto usa **montajes** (iguales para los dos modelos). Mide la dirección del efecto,
  no su tamaño exacto en fotos reales.
- El test multi-fuente de la v3 no se puede usar con el modelo 1: sus imágenes no están en local, y
  las de `huevos` que contiene son del train del modelo 1.

## 11. Qué hacer

1. ~~Desplegar la v3~~ **Hecho (2026-09-25)**: `~/huevos_v3` en el puerto 8000, con el modelo 1 de
   respaldo en el 8001. Ver `v3/despliegue/LEEME.md`.
2. ~~Banda de revisión~~ **Hecha**: `TAU_REVISION=0.07` en `routing.py`. Con la v3 reproduce
   exactamente la tabla de la sección 7 (0 rotos al empaque, 1/43 sanos botados).
   - **Además se corrigió un defecto de los dos modelos**: YOLO a veces deja una caja "Roto" y otra
     "Sano" sobre el mismo huevo, y el backend las trataba como dos huevos con rutas contrarias
     (en campo, un roto salía a la vez ACEPTADO y DESCARTE).
   - Ahora `fusionar_mismo_huevo` las une (IoU ≥ 0,6) y manda la más sospechosa, como entre frames.
   - En el test no cambia ninguna cifra; en campo corrige esos casos.
3. **Fondo neutro y fijo, luz difusa sin reflejos**: ataca los dos puntos débiles medidos
   (sobreexposición y sanos brillantes).
4. **v3.1 con fotos propias**: las de `evidencias/curaduria/` etiquetadas, más huevos sanos
   **marrones** fotografiados en la mano y en la banda. Es la forma de romper los atajos de contexto
   que ninguna fuente pública resuelve.
