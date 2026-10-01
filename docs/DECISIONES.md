# Decisiones de arquitectura (ADR)

Cada decisión trae el motivo y lo que se descartó. Sirven como guion de
sustentación: un jurado no pregunta "¿qué hiciste?", pregunta "¿por qué así?".

---

## ADR-01 — Inferencia con `.onnx`, no con `.pt`

**Decisión: ONNX Runtime con `best.onnx`.** PyTorch + Ultralytics no se instalan
en el servidor.

**Por qué:**

| | `.pt` (Ultralytics + PyTorch) | `.onnx` (ONNX Runtime) |
|---|---|---|
| Instalación | ~2.5 GB, 5–10 min | ~60 MB, <60 s |
| RAM en reposo | ~1.2 GB | ~350 MB |
| Latencia CPU (2 vCPU) | comparable o peor | 400 ms medidos |
| Dependencias | torch, torchvision, opencv, ultralytics | onnxruntime, numpy, Pillow |
| Licencia | Ultralytics AGPL-3.0 en el runtime | solo el peso del modelo |

Los tres argumentos decisivos, en orden:

1. **La ventana de 4 horas.** Bajar 2.5 GB de ruedas de PyTorch en una instancia
   de laboratorio puede tomar 10 minutos y puede fallar. Con ONNX el setup
   completo son ~3 min, verificados por `smoke_test.py`.
2. **La RAM.** Una `t3.medium` tiene 4 GB. Torch + Ultralytics + FastAPI +
   Docker/Redpanda deja un margen incómodo. ONNX no.
3. **No hay GPU**, así que la única ventaja real de PyTorch (CUDA) no aplica.

**Costo aceptado:** hay que escribir el letterbox y el NMS a mano, porque el
export salió con `nms=False` / `end2end=False` (verificado leyendo los metadatos
del `.onnx`: `opset 12`, entrada `images [1,3,640,640]`, salida `output0
[1,6,8400]`). Ya está hecho y **validado**: `backend/scripts/evaluar_test.py`
reproduce las métricas del run sobre el split de test. El NMS en numpy cuesta
**0.6 ms** contra 227 ms de la red, o sea que es gratis.

**`best.pt` no se bota:** se conserva para reentrenar, re-exportar a otro `imgsz`
y para el *fine-tune* de dominio. Es el artefacto fuente; el `.onnx` es el
artefacto de despliegue.

---

## ADR-02 — Expo managed, no Android Studio nativo

**Decisión: Expo (Expo Go para desarrollo; dev build solo si se necesita).**
Reutilizar exactamente el patrón del proyecto de detección de placas.

**Por qué:**

- Todo lo que necesita la app existe en Expo managed: `expo-camera` para
  capturar, `expo-image-manipulator` para comprimir, `fetch` +`FormData` para el
  multipart. Nada de esto requiere código nativo.
- **El cuello de botella no es el teléfono, es el servidor.** El backend tarda
  ~400 ms por frame; el teléfono captura y comprime en ~80 ms. Optimizar el lado
  nativo no compraría nada.
- **El tiempo de build es el enemigo.** Con Expo Go se escanea un QR y corre.
  Con Android Studio hay que configurar SDK, Gradle y emulador, y un build
  fallido en la semana de entrega es tiempo que no se recupera.
- Ya hay experiencia previa con ese patrón exacto. Reutilizar lo conocido reduce
  la superficie de riesgo justo donde no hay margen.

**Cuándo sí valdría ir nativo (y no es este caso):** si se quisiera inferencia
*on-device* (TFLite/NCNN) o un *frame processor* a 30 FPS con
`react-native-vision-camera`. Eso exige dev build y no aporta al objetivo, que
es demostrar un pipeline end-to-end con backend.

**Dos detalles que sí hay que cuidar** (ya resueltos en el código):
- Android bloquea HTTP en claro; por eso `app.json` trae
  `expo-build-properties` con `usesCleartextTraffic: true` e iOS
  `NSAllowsArbitraryLoads`. AWS Academy no da HTTPS fácil.
- **La IP pública cambia cada sesión del Learner Lab**, así que la URL del
  backend es un campo configurable persistido con `AsyncStorage`, no una
  constante en el código.

---

## ADR-03 — Simulación de la banda: software primero, hardware opcional

**Decisión: la simulación es de software y es la ruta principal. El servo por
serial es una capa opcional que se enchufa sin tocar el resto.**

**Por qué:** la demo no puede depender de que un SG90 arranque el día de la
sustentación. El patrón `Actuador` tiene dos implementaciones
(`ActuadorSimulado`, `ActuadorSerial`) detrás de la misma interfaz; se cambia con
una variable de entorno. El simulado **respeta el tiempo real de actuación del
servo (180 ms)** para que el throughput medido no sea una fantasía.

**Qué hace realista la simulación (esto es lo que hay que mostrar):**

1. **Tres salidas físicas**, no dos: ACEPTADO / DESCARTE / REVISIÓN.
2. **Tick de banda** desde la app: captura cada N ms, configurable, que es la
   velocidad emulada de la línea.
3. **Seguimiento entre frames.** Un huevo quieto frente a la cámara 3 segundos
   se contaría 8 veces sin esto. Un tracker por IoU con TTL asigna un `egg_id`
   estable y solo los huevos **nuevos** generan evento. Sin esto el throughput
   del dashboard sería mentira.
4. **Simulador headless** (`scripts/simular_banda.py`): reproduce una corrida
   completa a N huevos/min sin teléfono. Es el plan B de la demo y la
   herramienta para medir el techo del backend.

**Recomendación sobre el hardware:** sí vale la pena llevar el Arduino + SG90 +
3 LEDs (~$15 000 COP). El impacto visual de ver el brazo moverse es
desproporcionado frente al costo. Pero **conéctalo como extra, nunca como
dependencia**: si no arranca, `ACTUADOR_TIPO=simulado` y la demo sigue idéntica.
El firmware está en `firmware/banda_servo.ino`.

---

## ADR-04 — El diferencial: validación y ajuste de la propuesta

### Veredicto sobre la propuesta de Kafka

**Kafka/eventos es correcto y hay que conservarlo, pero solo no alcanza.**

Lo que está bien:
- Es una arquitectura genuinamente apropiada para el problema (no es Kafka
  puesto por ponerlo): un clasificador en banda es un flujo de eventos con
  múltiples consumidores de velocidades distintas.
- Se apoya en tu stack real (Python, FastAPI, Kafka, Docker), lo que te permite
  sustentarlo a profundidad.
- Permite una demo memorable: **apagar el consumidor del servo en vivo y mostrar
  que el dashboard sigue recibiendo eventos**. Eso demuestra desacople en 10
  segundos, que es más de lo que logra cualquier diapositiva.

El riesgo, dicho sin rodeos: un jurado puede responder *"eso es infraestructura,
no valor agregado sobre el problema de visión"*. Y tendría parte de razón: el
enunciado pide CNN para detectar huevos dañados, y Kafka no mejora ni un punto
de mAP.

### El diferencial reforzado: cuatro piezas, no una

Kafka pasa de ser *el* diferencial a ser **el sustrato** que hace posibles las
otras tres, que sí son de dominio:

**1. Ruteo por costo asimétrico con banda de incertidumbre — el núcleo.**
La banda no decide con `argmax` ni con umbral 0.5. Tiene una matriz de costos
(un roto que se cuela cuesta 12× lo que cuesta botar un sano) y **tres** rutas.
Los umbrales no son inventados: salen de la tabla de fuga medida en test (§5 de
`EVALUACION_MODELO.md`). Cuando el modelo no está seguro, **no decide**: manda a
revisión humana. Eso es un sistema de decisión, no un clasificador. Y se puede
mover en vivo desde el dashboard y ver cómo cambia el ruteo.

**2. Ciclo cerrado de datos (active learning).**
El tercer consumidor guarda exactamente los frames de la franja de
incertidumbre, con su evidencia, en `evidencias/curaduria/candidatos.jsonl` con
un campo `etiqueta_humana` en blanco. La operación de la banda **produce el
dataset de la siguiente versión del modelo**, y produce los casos más valiosos:
aquellos donde el modelo dijo "no sé".

**3. Trazabilidad por huevo.**
Cada huevo tiene un `egg_id` y un log de eventos inmutable: cuándo pasó, qué
creyó el modelo, qué ruta tomó y por qué. En una planta de alimentos eso es
auditoría de calidad, y ningún tutorial de "cámara + YOLO + servo" lo tiene.

**4. KPIs de planta, no de ML.**
El dashboard muestra tasa de rotura, throughput y tasa de revisión — lo que le
importa a un jefe de línea — no `mAP50-95`.

**Cómo enunciarlo en una frase:**

> "El valor agregado no es detectar huevos rotos: eso ya está resuelto. Es
> convertir una predicción en una decisión industrial auditable, con costo
> asimétrico, con una salida explícita para la incertidumbre, y con el ciclo de
> datos cerrado — sobre una arquitectura de eventos que permite que el
> dashboard, el actuador y la curaduría fallen de forma independiente."

### Sobre el alcance dado las 4 horas

La propuesta original (Kafka + 3 consumidores) **no es demasiado**, con una
condición de diseño que ya está implementada: **el bus es intercambiable.**

- `BUS_TIPO=memoria` → bus in-process, misma semántica de tópicos y grupos,
  arranca siempre. Es el plan B.
- `BUS_TIPO=kafka` → Redpanda en Docker.

Y **Redpanda en vez de Kafka + ZooKeeper**: un solo binario, sin JVM, arranca en
~5 s contra 60–90 s, ~400 MB de RAM contra ~1.5 GB. Con 2 vCPU y 4 h de ventana
eso no es preferencia estética.

Si Kafka no levanta el día de la demo, se cambia una variable de entorno y
**nada más se mueve**. Ese es el punto de haber programado contra una interfaz.

---

## ADR-05 — Un solo worker de uvicorn

`--workers 1`. ONNX Runtime ya paraleliza sobre todos los núcleos; varios
procesos de uvicorn se pelean la misma CPU, suben la p95 y **no** suben el
throughput. Además, cada worker cargaría su propia copia del modelo (~350 MB).

Por lo mismo, la inferencia está serializada con un `asyncio.Lock` dentro de la
fachada: N peticiones concurrentes no se disputan los mismos núcleos.
