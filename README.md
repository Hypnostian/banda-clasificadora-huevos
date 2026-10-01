# Banda clasificadora de huevos — CNN End2End

Detección de huevos rotos sobre una banda transportadora emulada, de punta a punta:
**app móvil (cámara) → API FastAPI → YOLOv8 en ONNX Runtime → política de ruteo por costo
asimétrico → eventos → actuador (servo) + dashboard de planta**.

## Integrantes

- Juan Sebastian Lizcano Jaimes
- Angie Katherine Suarez Ortiz

---

## Qué hay en producción: la v3

| | modelo 1 | **v3 (en uso)** |
|---|---|---|
| Modelo | YOLOv8m | **YOLOv8l** |
| Datos | 1 dataset (968 fotos) | **8 fuentes, 3.849 originales + 400 negativos sintéticos** |
| mAP50-95 en el test común (145 fotos) | 0,924 | 0,929 (empate) |
| Sano con raya de lapicero → descartado | **58 %** | **12 %** |
| Roto sobreexpuesto → se escapa | 56 % | 29 % |
| **Fotos reales de campo: rotos al empaque (de 24)** | **4** | **1** |
| Fotos reales de campo: sanos botados (de 19) | 2 | 0 |
| Latencia en el servidor (2 vCPU) | ~0,6 s | ~1,3 s |

El diferencial de la v3 frente a otros detectores de huevos:

1. **Negativos duros sintéticos.** En entrenamiento se dibujan sellos de tinta, rayas de
   lapicero y suciedad sobre huevos **sanos**, para que el modelo aprenda que una marca no es
   una fisura.
2. **Robustez a las fallas medidas en campo**: desenfoque, cámara movida, exposición, JPEG
   del celular y ruido.
3. **Curación multi-fuente auditada.** Detecta la convención de etiquetado de cada fuente,
   deduplica entre fuentes aunque la foto venga rotada o espejada, y protege el benchmark común.
4. **Banda de revisión.** Ante una sospecha moderada el huevo no se bota ni se acepta: va a
   un humano.

Además se detectó y documentó un **sesgo fuente↔etiqueta** en el dataset original: las fotos de
celular eran todas de huevos sanos. El modelo 1 lo aprendió y por eso dejaba pasar rotos en el
campo. Todo el análisis está en [`docs/COMPARACION_V1_V3.md`](docs/COMPARACION_V1_V3.md).

---

## Arquitectura

```
  App Expo (cámara de la banda) ── POST /inspeccion ──►  FastAPI (AWS, 2 vCPU, sin GPU)
                                                          │
                     FacadeInspeccionHuevos  (único punto de entrada)
                       1. DetectorHuevosONNX     letterbox → ONNX Runtime → NMS
                       2. fusionar_mismo_huevo   cajas duplicadas = un huevo (peor caso)
                       3. SeguimientoBanda       mismo huevo entre frames (IoU + TTL)
                       4. MotorDecision          costo asimétrico 12:1 → ruta
                       5. publica el evento en el bus (memoria o Kafka/Redpanda)
                                                          │
                        ┌─────────────────────────────────┼──────────────────────┐
                   ACTUADOR (servo)              MÉTRICAS (dashboard SSE)    CURADURÍA (candidatos
                                                                              para reentrenar)
```

**Las rutas de la banda (v3)**

| p(Roto) | ruta |
|---|---|
| ≥ 0,35 | **DESCARTE**: el servo lo desvía |
| 0,07 – 0,35 | **REVISIÓN**: inspección humana y candidato de reentrenamiento |
| < 0,07, p(Sano) ≥ 0,75 y foto nítida | **ACEPTADO**: va al empaque |
| lo demás (foto borrosa, sin evidencia) | REVISIÓN |

---

## Estructura

```
backend/            API FastAPI (patrón Facade), detector ONNX, política, eventos, dashboard
  modelos/          best.onnx del modelo 1
  scripts/          smoke_test, evaluar_test, simular_banda, diagnósticos, setup en AWS
app-movil/          app Expo (SDK 57): cámara de la banda, cambio de versión por puerto
firmware/           Arduino + servo SG90
huevos/             dataset original (train / valid / test, formato YOLO)
entrenamientos/     entrenamiento del modelo 1 (YOLOv8m): pesos y métricas
colab/              modelo 2 (clasificador sobre CDD, descartado) y su notebook
v3/                 versión 3
  entrenar_v3_yolov8.ipynb   notebook de Colab (curación + entrenamiento + evaluación)
  v3_*.py                    módulos probados que incrusta el notebook
  entrenamientov3/           modelo entrenado (best.onnx / best.pt), métricas, manifiesto
  comparar_v1_v3.py          comparación con el código de producción
  prueba_contexto.py         prueba del sesgo de contexto
  despliegue/                scripts para subir y operar la v3 en el servidor
datos_campo/        fotos reales tomadas con la app (las que fueron a revisión)
docs/               evaluación, decisiones, análisis de campo y comparaciones
```

Los pesos de los modelos (`*.onnx`, `*.pt`) están en la sección **Releases** del repositorio.
Hay que descargarlos y ponerlos en su carpeta:

| archivo del Release | va en |
|---|---|
| `v3_yolov8l.onnx` / `v3_yolov8l.pt` | `v3/entrenamientov3/weights/best.onnx` / `best.pt` |
| `modelo1_yolov8m.onnx` | `backend/modelos/best.onnx` y `entrenamientos/yolov8m_huevos_v1/weights/best.onnx` |
| `modelo1_yolov8m.pt` | `entrenamientos/yolov8m_huevos_v1/weights/best.pt` |
| `modelo2_yolov8m_cls.onnx` / `.pt` | `colab/entrenamientomodel2/weights/best.onnx` / `best.pt` |

---

## Cómo correrlo

**Servidor (AWS, Ubuntu).** La v3 corre en `~/huevos_v3`, puerto **8000**, y el modelo 1 queda
de respaldo en el **8001**. Para instalar o actualizar desde la raíz del proyecto (Git Bash):

```bash
bash v3/despliegue/subir_v3.sh
```

En el servidor: `banda` (estado), `banda-log`, `banda-smoke`, `banda-cambiar 1|v3`.
Detalle en [`v3/despliegue/LEEME.md`](v3/despliegue/LEEME.md). El script necesita la llave
SSH del servidor, que **no** está en el repositorio.

**App móvil (Expo Go).**

```bash
cd app-movil
npm install
npx expo start
```

Escanear el QR con Expo Go. En el engranaje se elige la versión: **v3 · puerto 8000**.

**Backend en local.**

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python scripts/smoke_test.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Reentrenar la v3.** Subir `v3/entrenar_v3_yolov8.ipynb` a Colab con GPU, poner la API key de
Roboflow en los *Secrets* (`ROBOFLOW_API_KEY`) y ejecutar todo.

---

## Endpoints

| Método | Ruta | Para qué |
|---|---|---|
| `GET` | `/` | dashboard en tiempo real |
| `GET` | `/salud` | estado: modelo cargado, bus, actuador |
| `POST` | `/inspeccion` | procesar un frame y devolver la ruta de cada huevo |
| `GET` | `/metricas` · `/stream` | KPIs de planta (JSON y SSE) |
| `GET`/`PUT` | `/politica` | leer o mover los umbrales en caliente |
| `POST` | `/banda/reiniciar` | poner el turno en cero |
| `GET` | `/docs` | Swagger |

## Documentación

- [`docs/COMPARACION_V1_V3.md`](docs/COMPARACION_V1_V3.md): v3 contra el modelo 1, robustez, sesgo de contexto y fotos de campo.
- [`docs/EVALUACION_MODELO.md`](docs/EVALUACION_MODELO.md): evaluación del modelo 1.
- [`docs/COMPARACION_MODELOS.md`](docs/COMPARACION_MODELOS.md): por qué se descartó el modelo 2.
- [`docs/ANALISIS_CAMPO.md`](docs/ANALISIS_CAMPO.md): fallas vistas en campo y su diagnóstico.
- [`docs/DECISIONES.md`](docs/DECISIONES.md): decisiones de arquitectura.
- [`docs/DESPLIEGUE.md`](docs/DESPLIEGUE.md): servidor y despliegue.

## Fuentes de datos

- Dataset `huevos` (propio del curso, Roboflow).
- Roboflow Universe: indra-bayu/egg-shell-cracked-detections, digit360mg/egg-crack-detection,
  new-workspace-jqutx/egg-crack-detection, carlos-oliveira-5vnxi/crack-egg, aau-nupnz/egg-crack,
  crackedeggs/cracked-eggs, ducks-sfodd/egg-quality. La lista completa y la licencia de cada
  una están en [`v3/README.md`](v3/README.md).
