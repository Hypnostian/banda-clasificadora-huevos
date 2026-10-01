# Modelo v3 — detector de huevos Roto / Sano

| archivo | qué es |
|---|---|
| [`entrenar_v3_yolov8.ipynb`](entrenar_v3_yolov8.ipynb) | **El notebook para Colab.** Se sube con *Archivo → Subir notebook*. |
| `v3_datos_lib.py` | Curación multi-fuente: auditoría, conversión de convenciones, dedup entre fuentes, split y negativos duros. |
| `v3_entrenamiento.py` | Regla de parada con cierre ordenado + aumentaciones fotométricas. |
| `v3_inferencia.py` | Inferencia idéntica a la del backend (probabilidades iguales bit a bit, verificado). |
| `test_v3_datos_lib.py` | 21 pruebas de la curación sobre fuentes sintéticas con los problemas reales. |
| `prueba_mini_entrenamiento.py` | Entrenamiento **real** en miniatura con ultralytics 8.4.157: cierre ordenado, reanudación y export compatible con el backend. |
| `validar_notebook_v3.py` | Ensayo de **punta a punta** del notebook con entrenamiento real (solo Roboflow y Drive simulados). |
| `generar_notebook_v3.py` | Genera el `.ipynb` incrustando los módulos probados. |
| `entrenamientov3/` | **El modelo entrenado** (170 épocas, `best.onnx` listo para el backend) y sus métricas. |
| `comparar_v1_v3.py` | Cara a cara contra el modelo 1 con el código del backend: detección, ruteo, robustez, integridad, latencia. |
| `prueba_contexto.py` | Prueba del atajo de contexto (¿el modelo mira el huevo o el tipo de foto?). |

> **Resultado:** la v3 empata en el benchmark y es mucho más robusta en las fallas de campo.
> Detalle y veredicto en [`docs/COMPARACION_V1_V3.md`](../docs/COMPARACION_V1_V3.md).

## Los datos: qué se buscó y qué se eligió

Se revisaron ~40 proyectos de Roboflow Universe más fuentes académicas (Mendeley, Kaggle, IEEE).
Criterio: **detección con cajas**, huevos **sanos y rotos etiquetados en la misma imagen**, y el número
de **originales descargables**. La cifra que anuncia cada proyecto engaña: Roboflow solo deja descargar
*versiones*, y la mayoría horneó una aumentación ×3 dentro de train.

| fuente | clases | originales en la versión | nota |
|---|---|---|---|
| `huevos` (tu dataset) | Roto / Sano | 968 | su **test es el benchmark común** con el modelo 1 |
| indra-bayu | egg / cracked / 0 | ~1.579 | sin aumentación; fotos de estudio |
| 360digitmg-swnfx | roto/normal × **blanco/marrón** | ~973 | trae el **color real** |
| digit360mg | roto/normal × **blanco/marrón** | ~954 | casi seguro las mismas fotos que la anterior: la dedup lo resuelve |
| new-workspace-jqutx | crack / normal | ~792 | |
| carlos-oliveira | crack / good | ~327 | |
| aau | crack / egg | ~213 | probablemente encierra solo la grieta: se convierte |
| crackedeggs | cracked / normal | ~149 | el proyecto dice 690; la versión, 312 archivos |
| ducks (egg-quality) | limpio / fisurado / **manchado** | ~384 | segmentación → cajas; pato excluido; manchado = negativo duro *real* |

Bruto ~6.300; tras quitar duplicados se esperan **~4.500**, unas **5 veces** los datos del modelo 1.

**Descartados, y por qué** — tan importante como lo elegido:
- *Egg Crack* (kay-test) y *Egg Crack 3*: anuncian 2.333 y 2.098, pero sus versiones tienen **167 y 144**.
- ~15 proyectos de la familia *360DigiTMG / Innodatatics*: son **copias** del mismo curso, y varios
  etiquetan **solo los rotos**: los sanos quedan como *fondo* y contradicen a las demás fuentes.
- *Egg Quality Grading* (≈20 proyectos): sangre, calcio y suciedad; **no hay fisuras**.
- *egg-quality* (PD): preprocesado a **escala de grises**.
- CDD_eggs_SJ: clasificación (sin cajas) y dominado por 14 videos (ver `docs/COMPARACION_MODELOS.md`).
- Mendeley *Good and Bad Eggs*: clasificación, 1.000 originales.

## El diferencial

1. **Negativos duros sintéticos: marca ≠ fisura.** Los huevos comerciales vienen sellados, y en campo
   se rayan y se ensucian. El modelo 1 bota el **56 %** de los sanos con una raya de lapicero. La v3
   entrena con sellos de tinta, rayas y suciedad dibujados sobre huevos **sanos**, etiquetados como
   sanos (solo en train). Se evalúa con un generador **distinto** al de train, para no medirse a sí misma.
2. **Robustez apuntada a las fallas medidas**: desenfoque, cámara movida, sobre y subexposición, JPEG
   del celular y ruido. Por defecto, Ultralytics aplica desenfoque con p = 0,01 y brillo/JPEG con p = 0.
3. **Desempeño por color de cáscara**, con el color estimado del recorte y **validado** contra las
   fuentes que traen el color real.
4. **Curación auditada**: detecta sola si una fuente encierra el huevo o solo la grieta, deduplica
   entre fuentes aunque la foto venga rotada o espejada, y protege el benchmark común.

## Split: 80/10/10

Con ~4.500 originales independientes, un 10 % de test son ~450 imágenes y ~600 huevos: la fuga se mide
con un intervalo de unos ±2,5 puntos. Además, el **benchmark común** (183 huevos) es una segunda
evaluación que no depende del split. Detección aprovecha más cada imagen extra que clasificación.
**La regla es explícita**: si tras deduplicar quedan menos de 3.000 originales, el notebook pasa solo
a **70/15/15** y lo imprime.

## Entrenamiento

- **YOLOv8l a 640**, ultralytics 8.4.157. Sin shear, perspectiva ni mixup: la geometría del huevo
  intacta, como en el modelo 1.
- **Regla**: mínimo 150 épocas. Después, 30 épocas sin mejorar al menos 0,3 puntos de mAP50-95 →
  **fase final de 20 épocas** sin mosaico y con la tasa de aprendizaje bajando a su mínimo. Tope 300.
  Cortar en seco dejaría la tasa alta y el mosaico abierto, que es donde se ganan los últimos puntos.
- **Tiempo** (estimado a partir de una medición real): el modelo 1 (YOLOv8m, 678 fotos de train) tardó
  **23 s por época** en tu Colab. La v3 tiene ~6–7 veces más fotos de train (con los negativos) y YOLOv8l
  cuesta ~2 veces más → **~5 min por época en una T4**. La regla termina, como pronto, en la época 170
  y como tarde en la 300: **de 14 a 25 horas** en T4, unas 7–12 en L4 y 3–6 en A100. Con
  `MODELO_BASE = "yolov8m.pt"` es la mitad. Desde la época 2, el notebook imprime el ritmo **real** y
  cuánto falta. Con Colab gratis serán varias sesiones: se reanuda desde Drive y la regla da la misma
  decisión al reanudar, incluso si se corta en plena fase final (probado).

## Qué quedó verificado

- 21 pruebas de la curación: convenciones, dedup con rotación/espejo, benchmark protegido, marcas dentro del huevo, color.
- Entrenamiento **real** en miniatura: cierre ordenado, mosaico cerrado, tasa al mínimo, tope respetado,
  desconexión + reanudación (también **en plena fase final**, con un control que demuestra que sin el
  arreglo esa época correría con mosaico y la tasa alta) y **ONNX que el backend carga sin cambios**.
- La inferencia del notebook da **exactamente** las mismas probabilidades que el backend.
- Ensayo de punta a punta del notebook con entrenamiento real.

**No verificado desde aquí**: la calidad de etiquetado de cada fuente (el notebook la audita y muestra
ejemplos con cajas en la celda 8: revísalos antes de dejarlo entrenar horas) y la licencia de 360a,
jqutx y ducks (las demás son CC BY 4.0). Hay que citar todas las fuentes en el informe.

## Después de entrenar

Copia `weights/best.onnx` a `backend/modelos/best.onnx` (guarda antes el del modelo 1) y pon en
`backend/.env` los umbrales que imprime la celda 17. No hay que tocar código. En el `t2.large`, YOLOv8l
tarda ~1,3 s por foto; si necesitas más velocidad, reentrena con `MODELO_BASE = "yolov8m.pt"`.
