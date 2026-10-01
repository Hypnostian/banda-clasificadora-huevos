# Evaluación del entrenamiento `yolov8m_huevos_v1`

Revisión hecha sobre `huevos/` y `entrenamientos/yolov8m_huevos_v1/`.
Las métricas de **validación** son las que produjo Ultralytics durante el run.
Las métricas de **test no existían en la carpeta**: las generé corriendo
`best.onnx` sobre `huevos/test/` con el mismo post-procesamiento que usa el
backend (`backend/scripts/evaluar_test.py`, reproducible).

---

## 1. Resumen ejecutivo

| Pregunta | Veredicto |
|---|---|
| ¿Las métricas son aceptables para producción simulada? | **Sí.** mAP50 0.977 / mAP50-95 0.924 en test. No hay que reentrenar para subir números. |
| ¿La clase `Sano` quedó subrepresentada en desempeño? | **No, y es lo contrario de lo esperado.** `Sano` rinde **mejor** que `Roto` (AP50-95 0.942 vs 0.905). El desbalance 3:1 no hizo daño. |
| ¿Qué tan grave es el falso negativo? | La confusión `Roto→Sano` es **prácticamente cero (0 casos en test)**. El riesgo real es otro: **no detectar el huevo** (6.4 % de fuga a conf 0.25). Se mitiga con política, no con reentrenamiento. |
| ¿Hay overfitting? | **No.** `val/cls_loss` seguía bajando en la época 140 y el early stopping cortó solo. |
| ¿Riesgo principal del proyecto? | **Brecha de dominio.** El dataset no es de banda transportadora. Ver §6 — es lo único que exige acción antes de sustentar. |

---

## 2. El dataset, como está realmente

| Split | Imágenes | Instancias | Roto | Sano |
|---|---|---|---|---|
| train | 678 | 908 | 696 (76.7 %) | 212 (23.3 %) |
| valid | 145 | 203 | 141 (69.5 %) | 62 (30.5 %) |
| test  | 145 | 183 | 140 (76.5 %) | 43 (23.5 %) |

Verificaciones adicionales que hice:

- **Sin fuga entre splits.** Crucé los nombres base (quitando el hash `.rf.` de
  Roboflow): 0 coincidencias train↔valid, train↔test y valid↔test. Las métricas
  no están infladas por duplicados, que es el error más común en datasets
  exportados de Roboflow.
- **Sin imágenes vacías**, 1.34 cajas por imagen en promedio (máx. 18 en una foto
  de cubeta).
- **Resolución uniforme** 480×640 vertical (97 % de las imágenes).
- **Tamaño del objeto**: el área relativa de la caja es p10 = 1.1 %, p50 = 7.7 %,
  p90 = 20.7 % del frame. El huevo ocupa típicamente **~28 % del ancho** de la
  imagen. Este dato importa para §6.

---

## 3. Entrenamiento

- **140 de 150 épocas.** Cortó por `patience=40` — la mejor época fue la **100**
  y a los 40 sin mejora paró. El early stopping funcionó como debía.
- **1.01 h en T4.**
- Mejor época (100): `P 0.963 · R 0.966 · mAP50 0.983 · mAP50-95 0.917`.

### ¿Overfitting?

No. Promedios por ventana de 20 épocas:

| épocas | mAP50-95 (media) | train/cls_loss | val/cls_loss |
|---|---|---|---|
| 41–60   | 0.856 | 0.762 | 0.628 |
| 61–80   | 0.866 | 0.645 | 0.557 |
| 81–100  | 0.891 | 0.548 | 0.473 |
| 101–120 | 0.903 | 0.472 | 0.421 |
| 121–140 | 0.908 | 0.396 | 0.410 |

La pérdida de validación **acompaña** a la de entrenamiento hasta el final; no se
abre la tijera. En la época 136 se nota la caída brusca de `train/*` porque
`close_mosaic=15` apagó el mosaic — es esperado, no es overfitting, y las
métricas de validación no se movieron.

Lo que sí hay es **meseta**: desde la época ~100 el mAP50-95 oscila entre 0.902 y
0.917 sin tendencia. Con 678 imágenes ese es el techo. **Más épocas no dan más.**

---

## 4. Desempeño por clase

### Validación (Ultralytics, época 100)

| Clase | AP@0.5 |
|---|---|
| Roto | 0.978 |
| Sano | 0.987 |
| **todas** | **0.983** |

### Test (medido con `best.onnx`, conf 0.001 / IoU 0.45)

| Clase | imgs | inst | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| Roto | 104 | 140 | 0.942 | 0.936 | 0.961 | 0.905 |
| Sano | 41 | 43 | 0.977 | 0.977 | 0.993 | **0.942** |
| **todas** | 145 | 183 | **0.960** | **0.956** | **0.977** | **0.924** |

**La conclusión contraintuitiva:** la clase minoritaria (`Sano`, 23 % de las
instancias) es la que **mejor** rinde. Esto tiene una explicación de fondo y es
un buen argumento de sustentación:

> El desbalance no es el factor dominante; la **varianza intra-clase** sí lo es.
> `Sano` es visualmente homogénea (un huevo íntegro se parece a otro huevo
> íntegro). `Roto` engloba fisuras capilares, abolladuras, cáscara hundida y
> huevos reventados con yema afuera. Aprender `Roto` es más difícil aunque haya
> 3× más ejemplos.

**Acción requerida: ninguna sobre el balanceo de clases.** Aplicar oversampling o
focal loss para "arreglar" el desbalance atacaría un problema que no existe.

---

## 5. Matriz de confusión y coste de los errores

### Validación (conteos absolutos, conf 0.25)

| | True Roto | True Sano | True fondo |
|---|---|---|---|
| **Pred Roto** | 135 | 1 | 11 |
| **Pred Sano** | 1 | 61 | 5 |
| **Pred fondo (no detectado)** | 5 | 0 | — |

### Test — fuga de rotos según el umbral

"Fuga" = huevo roto que **no** se manda a descarte, por cualquier causa.

La última columna es el **costo total** bajo la matriz asimétrica
(`c_fn:c_fp = 12:1`), que es lo que realmente se minimiza:

| conf | Roto OK | Roto→Sano | Roto no detectado | **FUGA** | Sano OK | FP sobre fondo | **costo** |
|---|---|---|---|---|---|---|---|
| 0.05 | 135 | 0 | 5 | **3.57 %** | 43 | 40 | **100** |
| 0.10 | 134 | 0 | 6 | 4.29 % | 43 | 31 | 103 |
| **0.15** | 133 | 0 | 7 | **5.00 %** | 43 | 22 | **106** |
| 0.20 | 131 | 0 | 9 | 6.43 % | 42 | 20 | 128 |
| 0.25 | 131 | 0 | 9 | 6.43 % | 42 | 19 | 127 |
| 0.30 | 131 | 0 | 9 | 6.43 % | 42 | 17 | 125 |
| 0.35 | 131 | 0 | 9 | 6.43 % | 42 | 13 | 121 |
| 0.45 | 129 | 0 | 11 | 7.86 % | 42 | 9 | 141 |
| 0.60 | 127 | 1 | 12 | 9.29 % | 42 | 8 | 164 |
| 0.70 | 123 | 0 | 17 | 12.14 % | 42 | 7 | 211 |

Tres lecturas que valen para la defensa:

1. **La columna `Roto→Sano` es 0 casi en todo el rango.** El clasificador de dos
   clases está esencialmente resuelto. El error del sistema **no** es "confundir
   roto con sano".
2. **El error real es el detector**, no el clasificador: 7–17 huevos rotos que el
   modelo no ve. Por eso la palanca correcta es **bajar el umbral de detección**,
   no reentrenar.
3. **El precio de bajarlo son falsos positivos sobre fondo** (40 a conf 0.05 vs 7
   a conf 0.70). Como un roto que se cuela cuesta 12× lo que cuesta botar un
   sano, el intercambio conviene: el costo mínimo está en **conf 0.05**.

### Por qué el umbral desplegado es 0.15 y no 0.05

El mínimo matemático es 0.05, pero la curva de costo es casi plana entre 0.05 y
0.15 (100 → 106, un 6 % peor) **con la mitad de falsos positivos**. Tomo 0.15
como margen de robustez, porque el conjunto de test no contiene frames de banda
vacía y por tanto no mide el comportamiento ante objetos que no son huevos.

Para cubrir ese hueco hice una prueba aparte: **8 frames sintéticos de banda
vacía** (gris liso, oscuro, claro, verde, ruido gaussiano, cinta modular con
listones, madera desenfocada, gradiente de iluminación).

> **Resultado: 0 detecciones fantasma en todos los frames, incluso a conf 0.05.**

Eso confirma dos cosas: el modelo **no alucina huevos** sobre superficies
uniformes o texturadas, y por lo tanto los "FP sobre fondo" de la tabla son casi
con seguridad **huevos reales sin anotar** en las fotos de cubeta (una imagen
tiene 18 huevos y no todos están etiquetados) — ruido de anotación, no error del
modelo. Aun así, la prueba usa fondos sintéticos y no cubre desorden real
(manos, herramientas, etiquetas), así que el margen de 0.15 se mantiene.

`CONF_MIN_DETECCION=0.15`, y el número es auditable: `scripts/evaluar_test.py`
recalcula la tabla y el óptimo de costo con los pesos que haya en el `.env`.

---

## 6. El hallazgo que sí exige acción: brecha de dominio

Inspeccionando `val_batch0_pred.jpg` y la distribución de tamaños de caja, el
dataset **no es de banda transportadora**. Son fotos de celular de huevos
individuales: sostenidos en la mano, sobre mesones de granito, en el alféizar de
una ventana, junto a un balón, sobre la estufa, en cubetas de cartón.

Condiciones de entrenamiento vs. condiciones de la demo:

| | Dataset | Banda real |
|---|---|---|
| Huevos por frame | 1 (mediana) | varios |
| Tamaño del huevo | ~28 % del ancho | mucho menor |
| Ángulo | picado cercano | fijo cenital |
| Movimiento | estático | desplazamiento + motion blur |
| Fondo | doméstico variado | superficie uniforme |

**Por qué importa:** el 0.977 de mAP50 es válido *en el dominio de fotos de
huevos individuales*. Un jurado que pregunte "¿esto funciona en una banda?"
tiene razón en preguntarlo, y la respuesta honesta requiere evidencia.

Hay dos caminos, y **el primero es obligatorio**:

**a) Alinear la demo con el dominio entrenado (costo: 0).**
La app de celular *es* la cámara: se captura de cerca, uno o pocos huevos por
frame, con la misma geometría de las fotos de entrenamiento. Así el modelo opera
dentro de su dominio y las métricas reportadas son defendibles tal cual.

**b) Adaptación de dominio (costo: ~2 h, y es lo que sube la nota).**
Grabar 2–3 min de video propio con huevos sobre una superficie en movimiento
(una bandeja empujada a mano basta), extraer ~150 frames, etiquetar en Roboflow y
hacer *fine-tune* 30 épocas desde `best.pt` con `lr0=0.0005`. Reportar las dos
tablas: dominio original y dominio banda.

Eso convierte el proyecto de "corrí un notebook de YOLO" en "identifiqué una
brecha de dominio, la medí y la cerré", que es exactamente el tipo de criterio
que diferencia un trabajo de ingeniería de un tutorial.

---

## 7. Latencia en CPU — lo que decide el despliegue

Medido sobre las 145 imágenes de test con ONNX Runtime, imgsz 640.

| Modelo | hilos | media | p95 | FPS |
|---|---|---|---|---|
| FP32 | 1 | 793 ms | 884 ms | 1.26 |
| FP32 | **2** | **400 ms** | 455 ms | **2.50** |
| FP32 | 4 | 297 ms | 326 ms | 3.37 |
| FP32 | 12 | 320 ms | 348 ms | 3.13 |
| INT8 dinámico | 2 | 464 ms | 569 ms | 2.15 |

Desglose a 12 hilos: preproceso 5.3 ms · **inferencia 227.6 ms** · post-proceso +
NMS 0.6 ms. El NMS en numpy es irrelevante frente a la red.

**La cuantización INT8 dinámica no sirve aquí — lo probé y es más lenta.** La
razón: `quantize_dynamic` solo cuantiza `MatMul`/`Gemm`, y YOLOv8 es
prácticamente todo `Conv`. El resultado es un modelo de 26 MB (vs 104 MB) con la
exactitud intacta (Δ mAP50-95 = +0.0007) pero **16 % más lento**. Queda
documentado como camino descartado con evidencia; si hiciera falta acelerar, la
palanca correcta es **re-exportar a `imgsz=480`** (≈1.8× más rápido) o
cuantización estática QDQ con calibración.

**Para AWS (2 vCPU): ~400 ms por frame ⇒ ~9.000 huevos/hora de techo teórico.**
Suficiente para la banda simulada, y es el número que hay que citar en la
sustentación en vez de inventar un "tiempo real" que no se sostiene.

---

## 8. Veredicto sobre reentrenar

**No reentrenar para mejorar métricas en este dataset.** El modelo está en la
meseta de lo que permiten 678 imágenes; las épocas 100–140 ya no movieron nada.

**Sí hacer un *fine-tune* de dominio** con frames propios de banda (§6b). Es otra
justificación, y es la única que cambia el resultado.

Si aun así se quisiera exprimir el dataset actual, el orden de retorno sería:
1. Revisar las anotaciones faltantes en las fotos de cubeta (varios "FP" son
   huevos reales sin etiquetar).
2. TTA en validación (`augment=True`) para reportar un techo superior.
3. `imgsz=800` — los huevos pequeños en cubeta son los que más se pierden.

Ninguna de las tres cambia la arquitectura del sistema.
