"""Genera entrenar_cdd_eggs_yolov8cls.ipynb.

La biblioteca de split se incrusta LEIDA DEL ARCHIVO PROBADO
(cdd_split_lib.py), no copiada a mano: el codigo que corre en Colab es
exactamente el que paso test_cdd_split_lib.py.
"""
import json
from pathlib import Path

AQUI = Path(__file__).parent
LIB = (AQUI / "cdd_split_lib.py").read_text(encoding="utf-8")

celdas = []


def md(texto: str) -> None:
    celdas.append({"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n").splitlines(True)})


def code(texto: str) -> None:
    celdas.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                   "source": texto.strip("\n").splitlines(True)})


# =============================================================================
md(r"""
# Segundo modelo: clasificador de huevos con YOLOv8-cls (CDD_eggs_SJ)

Dataset: [CDD_eggs_SJ](https://universe.roboflow.com/sj-ytwh0/cdd_eggs_sj) — 5.305 imágenes originales,
3 clases (`good`, `broken`, `dirty`), licencia **CC BY 4.0** (hay que citarlo en el informe; la cita está al final).

## Lo que hay que saber antes de correr esto

**1. Es un dataset de CLASIFICACIÓN, no de detección.** No trae cajas. Por eso se entrena
`yolov8m-cls` (una etiqueta por imagen), no `yolov8m` detector. En el proyecto encaja como
**segunda etapa**: el detector que ya tienes encuentra el huevo → se recorta → este modelo lo juzga.

**2. La versión publicada tiene fuga entre splits.** Hay frames del mismo video
(`20251002_174109_007`, `_028`, `_034`…) y ráfagas de fotos con segundos de diferencia. Un split aleatorio
pone un frame en train y su vecino en test: el modelo "ya vio" ese huevo y las métricas salen infladas.
Este notebook **agrupa todo lo que es el mismo huevo** (copias aumentadas, frames de video, ráfagas y
casi-duplicados visuales) y reparte **grupos**, no imágenes.

**3. Split 70/15/15, no 80/10/10.** Con ~5.3k originales, los ~530 originales que se pierden de train
cambian poco a un modelo preentrenado, pero un test 50 % más grande estrecha el intervalo de confianza de
la **fuga** (el KPI del proyecto), y un val más grande hace más estable el early stopping. Además, al
repartir grupos (que vienen en bloques) hace falta holgura para que cada clase quede bien estratificada.

**4. La versión v2 viene estirada a 224×224 y con shear ±10° horneado.** Se entrena a 224 (la resolución
nativa de `-cls`). Para la máxima calidad, ver la nota al final sobre hacer *fork* y exportar sin
aumentaciones.

## Qué responde el modelo

El modelo aprende 3 clases (`Roto`, `Sano`, `Sucio`), pero la **decisión** es jerárquica, como en una
planta real: primero *¿está dañado o no?*, y solo si está bueno, *¿limpio o sucio?*

| veredicto | cuándo | en la banda |
|---|---|---|
| ✗ **DAÑADO** | p(Roto) ≥ `TAU_ROTO` | descarte (servo) |
| ⚠ **REVISIÓN** | `TAU_DUDA` ≤ p(Roto) < `TAU_ROTO` | inspección humana |
| ✓ **BUENO (SUCIO)** | p(Roto) < `TAU_DUDA` y p(Sucio) > p(Sano) | lavado → empaque |
| ✓ **BUENO** | p(Roto) < `TAU_DUDA` y p(Sano) ≥ p(Sucio) | empaque |

Todo se decide con **un solo número**, p(Roto), y dos cortes. Se explica en una frase: *"si el modelo
cree que hay entre X % y Y % de probabilidad de que esté roto, lo mira una persona"*. Los dos cortes se
calibran en **val**, nunca en test:
- `TAU_ROTO`: el que minimiza el costo (un roto que se cuela cuesta 12 veces botar uno bueno).
- `TAU_DUDA`: la banda de revisión más ancha que no mande a revisión más del 5 % de los huevos buenos.

La revisión por **foto borrosa** ("no lo ve bien") ya la hace el backend con la compuerta de nitidez.

## Antes de empezar
1. **Entorno de ejecución → Cambiar tipo de entorno → T4 GPU.**
2. En roboflow.com → *Settings → API Keys* copia la **Private API Key** (la de la tabla, *Full access*).
   **No** la *Publishable* que empieza con `rf_`: esa es solo para inferencia en el navegador.
3. En Colab: panel izquierdo → icono de **llave (Secretos)** → **+ Agregar nuevo secreto** →
   nombre `ROBOFLOW_API_KEY`, valor = tu Private API Key → activa **Acceso al notebook**.
   Así la clave nunca queda escrita en el notebook.
4. **Entorno de ejecución → Ejecutar todas.** Tarda ~35–45 min en T4.
   - Al principio pide **permiso para Google Drive**: acéptalo. Se pide al inicio a propósito: si lo
     pidiera al final, tras 40 minutos sin que estés, la sesión podría cerrarse y perderías el modelo.
   - El entrenamiento se guarda **directo en Drive** época a época: si Colab se desconecta, no pierdes lo hecho.
   - La última celda te pide **subir fotos tuyas** de huevos para ver cómo las clasifica.

No pongas tu *Workspace ID* en ningún lado: el dataset pertenece al workspace `sj-ytwh0` (su autor),
y tu clave solo se usa para autorizar la descarga.
""")

# =============================================================================
code(r"""
# @title 1. Entorno: GPU + librerías (mismas versiones que tu primer entrenamiento)
import subprocess
gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
print(gpu or "SIN GPU -> Entorno de ejecucion > Cambiar tipo de entorno > T4 GPU")
assert gpu, "Activa la GPU antes de seguir"

# ultralytics 8.4.157 = la misma version con la que se entreno el detector: el export ONNX sale igual.
%pip install -q ultralytics==8.4.157 roboflow==1.5.0 onnx onnxruntime onnxslim
import ultralytics
ultralytics.checks()
""")

code(r"""
# @title 2. Configuración (todo lo ajustable está aquí)
# Workspace del AUTOR del dataset (no el tuyo): tu API key solo autoriza la descarga.
WORKSPACE = "sj-ytwh0"
PROYECTO  = "cdd_eggs_sj"
VERSION   = 2                     # la ultima publicada

# Como mapear las 3 clases del dataset a las del proyecto:
#   3clases            broken->Roto, good->Sano, dirty->Sucio   (recomendado)
#   binario            broken->Roto, good+dirty->Sano           (sucio no es dano)
#   binario_sin_sucios broken->Roto, good->Sano, dirty se descarta
MODO_CLASES = "3clases"  # @param ["3clases", "binario", "binario_sin_sucios"]

PROPORCIONES = {"train": 0.70, "val": 0.15, "test": 0.15}

MODELO_BASE = "yolov8m-cls.pt"  # @param ["yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt"]
IMGSZ    = 224    # resolucion nativa de -cls y de la v2 del dataset
BATCH    = 64
SEMILLA  = 42

# Entrenamiento largo: minimo de epocas, y parada solo tras PACIENCIA epocas sin una
# mejora SIGNIFICATIVA (>= MEJORA_MINIMA de exactitud en val). Ultralytics no trae
# ninguna de las dos cosas: las implementa un callback en la celda 9.
MIN_EPOCAS    = 150    # @param {type:"integer"}
PACIENCIA     = 30     # @param {type:"integer"}
MEJORA_MINIMA = 0.005  # @param {type:"number"}
EPOCAS_MAX    = 200    # @param {type:"integer"}

# Sobremuestreo en TRAIN de clases minoritarias hasta esta fraccion de la mayoritaria.
# (cls_pw NO sirve en clasificacion en 8.4.157: la perdida es cross_entropy sin pesos.)
SOBREMUESTREO = 0.6

# Una sola copia por original tambien en train. La v2 trae cada huevo de train DUPLICADO
# (flip+shear horneados): con eso cada "epoca" son ~2 pasadas por huevo y un entrenamiento
# de 150+ epocas memoriza el doble de rapido. Las aumentaciones en linea ya cubren flip y shear.
UNA_COPIA_EN_TRAIN = True

# Matriz de costos del proyecto: dejar pasar un roto cuesta 12 veces botar uno bueno.
COSTO_FN, COSTO_FP = 12.0, 1.0

# Carga maxima de REVISION: que fraccion de huevos BUENOS (limpios o sucios) se tolera
# mandar a inspeccion humana. Con esto se calibra la banda de duda en val.
MAX_REVISION_BUENOS = 0.05  # @param {type:"number"}

GUARDAR_EN_DRIVE = True  # @param {type:"boolean"}

from pathlib import Path
from datetime import datetime
NOMBRE_RUN = f"cdd_{MODO_CLASES}_{Path(MODELO_BASE).stem}_{datetime.now():%Y%m%d_%H%M}"
ORIGEN  = Path("/content/cdd_original")
LIMPIO  = Path("/content/cdd_limpio")

# Drive se monta AL INICIO a proposito: si el permiso se pidiera al final, tras 40 min de
# entrenamiento sin nadie frente a la pantalla, la sesion podria cerrarse y perder el modelo.
# Ademas el run se guarda directo en Drive, epoca a epoca: sobrevive a una desconexion.
if GUARDAR_EN_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    CARPETA_RUNS = Path("/content/drive/MyDrive/eggdetection/runs_cdd")
else:
    CARPETA_RUNS = Path("/content/runs_cdd")
CARPETA_RUNS.mkdir(parents=True, exist_ok=True)
print("run:", NOMBRE_RUN, "\nse guarda en:", CARPETA_RUNS / NOMBRE_RUN)
""")

md(r"""
### 3. Biblioteca de split sin fuga
Esta celda escribe `cdd_split_lib.py`, la lógica que agrupa "el mismo huevo" y reparte grupos.
Es **exactamente** el archivo probado localmente con `test_cdd_split_lib.py` sobre un dataset sintético
con fugas plantadas (copias aumentadas, frames de video, ráfagas, duplicados recomprimidos y volteados).
""")
# %%writefile TIENE que ser la primera linea de la celda: si va detras de un
# '# @title', IPython no la reconoce como magia de celda y la celda falla.
code("%%writefile cdd_split_lib.py\n" + LIB)

code(r"""
# @title 4. Descargar el dataset desde Roboflow
from google.colab import userdata
try:
    API_KEY = userdata.get("ROBOFLOW_API_KEY")
except Exception as e:
    raise RuntimeError(
        "Falta la API key. Panel izquierdo > icono de llave (Secretos) > agrega ROBOFLOW_API_KEY "
        "con tu clave de https://app.roboflow.com/settings/api y activa 'Acceso al notebook'."
    ) from e

from roboflow import Roboflow
rf = Roboflow(api_key=API_KEY)
ds = rf.workspace(WORKSPACE).project(PROYECTO).version(VERSION).download(
    "folder", location=str(ORIGEN), overwrite=True)
print("\ndescargado en:", ds.location)
for d in sorted(p for p in ORIGEN.rglob("*") if p.is_dir()):
    n = sum(1 for f in d.iterdir() if f.is_file())
    if n:
        print(f"  {str(d.relative_to(ORIGEN)):<28} {n:>6} archivos")
""")

code(r"""
# @title 5. Inventario y mapeo de clases al proyecto
import importlib, cdd_split_lib as L
importlib.reload(L)
from collections import Counter

MAPEOS = {
    "3clases":            {"broken": "Roto", "good": "Sano", "dirty": "Sucio"},
    "binario":            {"broken": "Roto", "good": "Sano", "dirty": "Sano"},
    "binario_sin_sucios": {"broken": "Roto", "good": "Sano", "dirty": None},
}
mapeo = MAPEOS[MODO_CLASES]

todos = L.inventariar(ORIGEN)
encontradas = sorted({it["clase_orig"] for it in todos})
sin_mapeo = [c for c in encontradas if c.lower() not in mapeo]
if sin_mapeo:
    raise RuntimeError(f"Clases sin mapeo: {sin_mapeo}. El dataset trae: {encontradas}. Ajusta MAPEOS.")

items = []
for it in todos:
    destino = mapeo[it["clase_orig"].lower()]
    if destino:
        it["clase"] = destino
        items.append(it)

print(f"archivos: {len(todos)}  |  usados en modo '{MODO_CLASES}': {len(items)}")
print(f"originales distintos (sin contar copias aumentadas): {len({it['base'] for it in items})}\n")
tabla = Counter((it["split_orig"], it["clase_orig"]) for it in todos)
print(f"{'clase original':<16}" + "".join(f"{s:>9}" for s in ("train", "val", "test")) + "   -> proyecto")
for c in encontradas:
    print(f"{c:<16}" + "".join(f"{tabla[(s, c)]:>9}" for s in ("train", "val", "test"))
          + f"   -> {mapeo[c.lower()] or '(descartada)'}")
""")

code(r"""
# @title 6. ¿Hay fuga en el split publicado? (agrupar lo que es el mismo huevo)
# Hash de 256 bits con umbral 16 y tope de 30 por grupo: calibrado sobre 968 fotos reales.
# (La primera version usaba 64 bits y sin tope: encadeno el 91.5% del dataset en un grupo.)
L.calcular_hashes(items, lado=16)
info = L.agrupar(items, gap_rafaga_s=20, duracion_max_rafaga_s=120, umbral_hash=16, max_grupo_hash=30)

print(f"\n{len({it['base'] for it in items})} originales -> {info['n_grupos']} grupos 'mismo huevo'")
print("uniones por causa:")
for k, v in info["razones"].most_common():
    print(f"  {k:<26} {v:>6}")
print(f"  uniones por hash rechazadas por el tope: {info['rechazadas_por_tope']}")

print("\ngrupos mas grandes (si uno solo tiene una fraccion grande del dataset, algo anda mal):")
for g in L.grupos_mas_grandes(items, info):
    print(f"  {g['originales']:>6} originales ({100*g['originales']/len({it['base'] for it in items}):4.1f}%)"
          f"  {g['causas']}  ej: {g['ejemplos']}")

fuga = L.fuga_en_split_original(items)
print("\n=== FUGA EN EL SPLIT PUBLICADO ===")
print(f"grupos repartidos entre varios splits: {fuga['grupos_cruzados']}")
print(f"imagenes de val/test que comparten huevo con train: "
      f"{fuga['imgs_eval_contaminadas']} de {fuga['total_eval']} "
      f"({100*fuga['imgs_eval_contaminadas']/max(1,fuga['total_eval']):.1f}%)")
print("-> cualquier metrica reportada sobre ese test esta inflada en esa proporcion." if fuga["grupos_cruzados"]
      else "-> el split publicado no tenia fuga detectable.")
""")

code(r"""
# @title 7. Split nuevo 70/15/15 por grupos, estratificado, y verificación independiente
import csv
asignado = L.split_estratificado_por_grupos(items, PROPORCIONES, semilla=SEMILLA)
# Se DETIENE aqui si algun split queda muy por debajo de su proporcion (lo que paso en el
# primer entrenamiento: val=227 y test=226 imagenes). Mejor un error ahora que 1 h de GPU tirada.
reparto = L.revisar_reparto(items, asignado, PROPORCIONES, info)
print("reparto de originales:", {k: f"{100*v:.1f}%" for k, v in reparto.items()})
final = L.materializar(items, asignado, LIMPIO, sobremuestrear_hasta=SOBREMUESTREO,
                       una_copia_en_train=UNA_COPIA_EN_TRAIN, semilla=SEMILLA)

print("\n=== VERIFICACION ===")
sin_fuga = L.verificar_sin_fuga(final, umbral_hash=16)
print("CERO fuga: ningun original, grupo ni casi-duplicado cruza splits" if sin_fuga
      else "ATENCION: revisa los mensajes de fuga de arriba")

clases = sorted({it["clase"] for it in items})
unicos = {s: Counter(it["clase"] for it in {i["base"]: i for i in final[s]}.values()) for s in final}
total = Counter()
for c in unicos.values():
    total.update(c)
print(f"\n{'clase':<8}" + "".join(f"{s:>16}" for s in ("train", "val", "test")))
for c in clases:
    print(f"{c:<8}" + "".join(f"{unicos[s][c]:>8} ({100*unicos[s][c]/total[c]:4.1f}%)" for s in ("train", "val", "test")))
print(f"\narchivos en disco: " + ", ".join(f"{s}={len(final[s])}" for s in ("train", "val", "test"))
      + "  (train incluye copias aumentadas y sobremuestreo)")

n_roto_test = unicos["test"].get("Roto", 0)
lo, hi = L.ic95_proporcion(round(0.03 * n_roto_test), n_roto_test)
print(f"\nCon {n_roto_test} rotos en test, una fuga real de 3% se mediria con IC95 ~[{100*lo:.1f}%, {100*hi:.1f}%]")

# Manifiesto: que archivo fue a que split (reproducibilidad y auditoria)
with open(LIMPIO / "manifiesto_split.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["split", "clase", "archivo", "original", "grupo", "split_publicado"])
    for s, lista in final.items():
        for it in lista:
            w.writerow([s, it["clase"], it["ruta"].name, it["base"], it["grupo"], it["split_orig"]])
print("manifiesto:", LIMPIO / "manifiesto_split.csv")
""")

code(r"""
# @title 8. Vista previa: revisa a ojo que las etiquetas tengan sentido
import matplotlib.pyplot as plt, random
from PIL import Image
random.seed(SEMILLA)
fig, ejes = plt.subplots(len(clases), 8, figsize=(16, 2.2 * len(clases)))
for fila, c in enumerate(clases):
    muestras = random.sample(sorted((LIMPIO / "train" / c).glob("*")), 8)
    for col, f in enumerate(muestras):
        ax = ejes[fila][col] if len(clases) > 1 else ejes[col]
        ax.imshow(Image.open(f)); ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(c, fontsize=13)
plt.suptitle("Muestras de train por clase", fontsize=14); plt.tight_layout(); plt.show()
""")

md(r"""
## Hiperparámetros: por qué no son los de fábrica

| parámetro | fábrica | aquí | por qué |
|---|---|---|---|
| `scale` | 0.5 | **0.25** | El recorte aleatorio conserva ≥75 % de la imagen (fábrica: ≥50 %; ImageNet puro: 8 %). Un recorte chico puede dejar **fuera la fisura** y mantener la etiqueta "Roto": ruido de etiqueta. |
| `erasing` | 0.4 | **0.15** | Borrar un rectángulo al azar regulariza, pero en un defecto pequeño puede borrar la única evidencia. |
| `flipud` | 0.0 | **0.5** | Un huevo no tiene "arriba": el volteo vertical es aumentación válida y gratis. |
| `hsv_v` | 0.4 | **0.45** | Más variación de brillo: en campo la sobreexposición hizo fallar al detector. |
| `dropout` | 0.0 | **0.15** | Regulariza la cabeza de clasificación con ~3.7k originales. |
| `optimizer` | auto | **AdamW, lr0=5e-4, cos_lr** | *Fine-tuning* estable desde pesos de ImageNet. |
| `auto_augment` | randaugment | randaugment | Incluye variaciones de nitidez, que ayudan contra el desenfoque. |
""")

code(r"""
# @title 9. Entrenar: mínimo 150 épocas; para si pasan 30 sin mejora significativa
# Si Colab se desconecto a mitad: vuelve a correr las celdas 1 a 8 (recrean el mismo split,
# todo usa SEMILLA), pega aqui el nombre de la carpeta del run viejo (esta en
# MyDrive/eggdetection/runs_cdd/) y corre esta celda y las siguientes.
REANUDAR = ""  # @param {type:"string"}

import csv
from ultralytics import YOLO

def parada_con_minimo(trainer):
    # Ultralytics no trae "minimo de epocas" ni "mejora minima": se implementan aqui.
    # Corre al final de cada epoca, ANTES de que el entrenador decida si sigue.
    # Recalcula todo desde results.csv, asi que funciona igual al reanudar.
    if not Path(trainer.csv).exists():
        return
    with open(trainer.csv, encoding="utf-8") as fh:
        filas = [{k.strip(): v for k, v in f.items()} for f in csv.DictReader(fh)]
    col = next((k for k in filas[0] if "top1" in k), None) if filas else None
    if col is None:
        return
    hist = [float(f[col]) for f in filas]
    ref, ref_ep = hist[0], 1                   # ultima mejora SIGNIFICATIVA
    for ep, v in enumerate(hist, 1):
        if v > ref + MEJORA_MINIMA:
            ref, ref_ep = v, ep
    epoca, sin_mejora = len(hist), len(hist) - ref_ep
    estado = (f"  [parada] epoca {epoca}: top1 val={hist[-1]:.4f} | referencia {ref:.4f} (epoca {ref_ep})"
              f" | {sin_mejora}/{PACIENCIA} sin mejorar >= {MEJORA_MINIMA:g}")
    if epoca < MIN_EPOCAS:
        print(estado + f" | minimo {MIN_EPOCAS}: sigue")
    elif sin_mejora >= PACIENCIA:
        trainer.stop = True
        print(estado + " -> SE DETIENE: paso el minimo y lleva la paciencia completa sin mejora")
    else:
        print(estado)

if REANUDAR:
    RUN = CARPETA_RUNS / REANUDAR
    assert (RUN / "weights" / "last.pt").exists(), f"No encuentro {RUN / 'weights' / 'last.pt'}"
    print("reanudando desde", RUN / "weights" / "last.pt")
    modelo = YOLO(str(RUN / "weights" / "last.pt"))
    modelo.add_callback("on_fit_epoch_end", parada_con_minimo)
    modelo.train(resume=True)
else:
    modelo = YOLO(MODELO_BASE)
    modelo.add_callback("on_fit_epoch_end", parada_con_minimo)
    modelo.train(
        data=str(LIMPIO), imgsz=IMGSZ, batch=BATCH,
        epochs=EPOCAS_MAX,   # tope; el callback decide cuando parar despues de MIN_EPOCAS
        patience=0,          # 0 = apaga la parada de fabrica (paraba a las 15 sin mejora)
        # En el run anterior train/loss llego a 0.035 mientras val caia de 0.868 a 0.75:
        # sobreajuste. Para 150+ epocas hace falta mas regularizacion, no menos.
        dropout=0.3, weight_decay=0.01,
        optimizer="AdamW", lr0=5e-4, cos_lr=True, warmup_epochs=2,
        scale=0.25, erasing=0.15, fliplr=0.5, flipud=0.5, hsv_h=0.015, hsv_s=0.5, hsv_v=0.45,
        auto_augment="randaugment",
        cache="ram", workers=2, seed=SEMILLA, deterministic=False,
        project=str(CARPETA_RUNS), name=NOMBRE_RUN, exist_ok=True, plots=True,
    )
    RUN = Path(modelo.trainer.save_dir)
print("\npesos:", RUN / "weights" / "best.pt")
""")

code(r"""
# @title 10. Curvas: ¿aprendió o memorizó?
import pandas as pd
df = pd.read_csv(RUN / "results.csv"); df.columns = [c.strip() for c in df.columns]
col_acc = next(c for c in df.columns if "top1" in c)
col_tr  = next(c for c in df.columns if c.startswith("train/") and "loss" in c)
col_va  = next(c for c in df.columns if c.startswith("val/") and "loss" in c)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
a1.plot(df["epoch"], df[col_tr], label="train"); a1.plot(df["epoch"], df[col_va], label="val")
a1.set_title("Pérdida"); a1.set_xlabel("época"); a1.legend()
a2.plot(df["epoch"], df[col_acc]); a2.set_title("Exactitud top-1 en VAL"); a2.set_xlabel("época")
plt.tight_layout(); plt.show()

mejor = int(df[col_acc].idxmax())
ult = df.iloc[-1]
print(f"mejor epoca: {int(df.loc[mejor,'epoch'])}  top1 val = {df.loc[mejor, col_acc]:.4f}")
print(f"ultima epoca: {int(ult['epoch'])}  train/loss={ult[col_tr]:.4f}  val/loss={ult[col_va]:.4f}")
brecha = ult[col_va] - ult[col_tr]
print("-> sin senales fuertes de sobreajuste" if brecha < 0.3 else
      f"-> OJO: val/loss supera a train/loss por {brecha:.2f}: posible sobreajuste")
""")

code(r"""
# @title 11. Evaluación en TEST con las métricas del proyecto
import numpy as np, json
modelo = YOLO(str(RUN / "weights" / "best.pt"))
NOMBRES = [modelo.names[i] for i in range(len(modelo.names))]
IDX = {n: i for i, n in enumerate(NOMBRES)}
R, S = IDX["Roto"], IDX["Sano"]

def cargar(split):
    rutas, y = [], []
    for i, n in enumerate(NOMBRES):
        for f in sorted((LIMPIO / split / n).glob("*")):
            rutas.append(str(f)); y.append(i)
    return rutas, np.array(y)

def predecir(fuente, lote=64):
    # Lotes a mano: con una lista de imagenes PIL, ultralytics las mete TODAS en un
    # solo lote (LoadPilAndNumpy); con cientos de imagenes eso puede agotar la GPU.
    if not len(fuente):
        return np.zeros((0, len(NOMBRES)), dtype=np.float32)
    out = []
    for i in range(0, len(fuente), lote):
        for r in modelo.predict(fuente[i:i + lote], imgsz=IMGSZ, verbose=False, stream=True):
            out.append(r.probs.data.float().cpu().numpy())
    return np.stack(out)

rutas_val, y_val = cargar("val");   P_val = predecir(rutas_val)
rutas_test, y_test = cargar("test"); P_test = predecir(rutas_test)

# Ambos cortes se calibran en VAL. Elegirlos mirando test seria hacer trampa:
# el numero de test dejaria de ser una estimacion honesta.
p_roto_val = P_val[:, R]
roto_val, bueno_val = (y_val == R), (y_val != R)

# --- 1) TAU_ROTO: a partir de aqui es DANADO. Minimo costo 12:1 en val. ---
mejor = None
for tau in np.round(np.arange(0.05, 0.96, 0.01), 2):
    d = p_roto_val >= tau
    costo = int((roto_val & ~d).sum()) * COSTO_FN + int((bueno_val & d).sum()) * COSTO_FP
    if mejor is None or costo < mejor[0]:          # en empate gana el mas bajo: el mas prudente
        mejor = (costo, float(tau))
TAU_ROTO = mejor[1]

# --- 2) TAU_DUDA: por debajo es BUENO. La banda [TAU_DUDA, TAU_ROTO) es REVISION. ---
# Se abre la banda hacia abajo mientras no mande a revision mas del MAX_REVISION_BUENOS
# de los huevos buenos de val. Mas ancha = menos rotos se cuelan, mas trabajo humano.
TAU_DUDA = TAU_ROTO
for t in np.round(np.arange(TAU_ROTO - 0.01, 0.0, -0.01), 2):
    en_banda = bueno_val & (p_roto_val >= t) & (p_roto_val < TAU_ROTO)
    if en_banda.sum() / max(1, bueno_val.sum()) > MAX_REVISION_BUENOS:
        break
    TAU_DUDA = float(t)

print(f"Calibrado en VAL ({len(y_val)} imagenes):")
print(f"  DANADO     si p(Roto) >= {TAU_ROTO:.2f}   (costo minimo {COSTO_FN:g}:{COSTO_FP:g})")
print(f"  REVISION   si {TAU_DUDA:.2f} <= p(Roto) < {TAU_ROTO:.2f}   (<= {100*MAX_REVISION_BUENOS:.0f}% de los buenos)")
print(f"  BUENO      si p(Roto) < {TAU_DUDA:.2f}" + ("   -> y si p(Sucio) > p(Sano): BUENO (SUCIO)" if "Sucio" in IDX else ""))
print()

def veredicto(p):
    # Jerarquico, como en planta: primero danado o no; solo entre los buenos, limpio o sucio.
    if p[R] >= TAU_ROTO:
        return "DANADO"
    if p[R] >= TAU_DUDA:
        return "REVISION"
    if "Sucio" in IDX and p[IDX["Sucio"]] > p[S]:
        return "BUENO_SUCIO"
    return "BUENO"

def rutear(P):
    return np.array([veredicto(p) for p in P], dtype=object)

rutas_banda = rutear(P_test)
pred = P_test.argmax(1)

# --- metricas clasicas ---
print(f"exactitud top-1 en TEST: {(pred == y_test).mean():.4f}   ({len(y_test)} imagenes)\n")
print(f"{'clase':<8}{'P':>8}{'R':>8}{'F1':>8}{'n':>7}")
for i, n in enumerate(NOMBRES):
    tp = int(((pred == i) & (y_test == i)).sum()); fp = int(((pred == i) & (y_test != i)).sum())
    fn = int(((pred != i) & (y_test == i)).sum())
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn); f1 = 2 * p * r / max(1e-9, p + r)
    print(f"{n:<8}{p:>8.3f}{r:>8.3f}{f1:>8.3f}{int((y_test == i).sum()):>7}")

# --- matriz de confusion ---
M = np.zeros((len(NOMBRES), len(NOMBRES)), int)
for t, p in zip(y_test, pred):
    M[t, p] += 1
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(M / np.maximum(1, M.sum(1, keepdims=True)), cmap="Blues", vmin=0, vmax=1)
for i in range(len(NOMBRES)):
    for j in range(len(NOMBRES)):
        ax.text(j, i, M[i, j], ha="center", va="center", color="white" if M[i, j] > M[i].sum() / 2 else "black")
ax.set_xticks(range(len(NOMBRES)), NOMBRES); ax.set_yticks(range(len(NOMBRES)), NOMBRES)
ax.set_xlabel("predicho"); ax.set_ylabel("real"); ax.set_title("Matriz de confusión (test)")
plt.tight_layout(); plt.savefig(RUN / "confusion_test.png", dpi=150); plt.show()

# --- lo que de verdad importa: que veredicto recibe cada tipo de huevo ---
VEREDICTOS = ["BUENO", "BUENO_SUCIO", "REVISION", "DANADO"] if "Sucio" in IDX else ["BUENO", "REVISION", "DANADO"]
print(f"\nVEREDICTOS EN TEST (filas = lo que el huevo es en realidad)")
print(f"{'real':<8}" + "".join(f"{v:>14}" for v in VEREDICTOS))
for i, n in enumerate(NOMBRES):
    m = y_test == i
    print(f"{n:<8}" + "".join(f"{int((rutas_banda[m] == v).sum()):>8} {100*(rutas_banda[m] == v).mean():>4.0f}%" for v in VEREDICTOS))

m_roto, m_bueno = (y_test == R), (y_test != R)
n_roto, n_bueno = int(m_roto.sum()), int(m_bueno.sum())
fuga = int((rutas_banda[m_roto] != "DANADO").sum())
a_empaque = int(np.isin(rutas_banda[m_roto], ["BUENO", "BUENO_SUCIO"]).sum())
buenos_a_revision = int((rutas_banda[m_bueno] == "REVISION").sum())
buenos_descartados = int((rutas_banda[m_bueno] == "DANADO").sum())
lo, hi = L.ic95_proporcion(fuga, n_roto)
lo2, hi2 = L.ic95_proporcion(a_empaque, n_roto)
print(f"\nrotos que NO salen como DANADO:   {fuga}/{n_roto} = {100*fuga/max(1,n_roto):.2f}%  IC95 [{100*lo:.1f}%, {100*hi:.1f}%]")
print(f"  de esos, llegan al EMPAQUE (BUENO): {a_empaque}/{n_roto} = {100*a_empaque/max(1,n_roto):.2f}%  IC95 [{100*lo2:.1f}%, {100*hi2:.1f}%]")
print(f"  el resto cae en REVISION, que es justo para lo que existe")
print(f"buenos mandados a REVISION:       {buenos_a_revision}/{n_bueno} = {100*buenos_a_revision/max(1,n_bueno):.1f}%")
print(f"buenos botados como DANADO:       {buenos_descartados}/{n_bueno} = {100*buenos_descartados/max(1,n_bueno):.1f}%")
if "Sucio" in IDX:
    ok_sub = sum(1 for t, v in zip(y_test, rutas_banda)
                 if v in ("BUENO", "BUENO_SUCIO") and t != R and (v == "BUENO_SUCIO") == (t == IDX["Sucio"]))
    tot_sub = int(np.isin(rutas_banda[m_bueno], ["BUENO", "BUENO_SUCIO"]).sum())
    print(f"limpio vs sucio, entre los buenos: {ok_sub}/{tot_sub} bien distinguidos")

METRICAS = {"top1_test": float((pred == y_test).mean()), "tau_roto": TAU_ROTO, "tau_duda": TAU_DUDA,
            "fuga_pct": 100 * fuga / max(1, n_roto), "fuga_ic95": [100 * lo, 100 * hi],
            "fuga_a_empaque_pct": 100 * a_empaque / max(1, n_roto),
            "fuga_a_empaque_ic95": [100 * lo2, 100 * hi2],
            "buenos_a_revision_pct": 100 * buenos_a_revision / max(1, n_bueno),
            "buenos_descartados_pct": 100 * buenos_descartados / max(1, n_bueno),
            "n_test": int(len(y_test)), "n_roto_test": n_roto, "clases": NOMBRES, "modo": MODO_CLASES}
""")

code(r"""
# @title 12. Robustez: las MISMAS pruebas que tumbaron al detector en campo
from PIL import ImageDraw, ImageFilter, ImageEnhance
import random

def rayar(im, semilla=3):
    # linea oscura irregular por el centro, imitando un lapicero sobre la cascara
    im = im.copy(); d = ImageDraw.Draw(im); w, h = im.size; rng = random.Random(semilla)
    pts = [(w / 2 + w * 0.12 * np.sin(t * 3.1) + rng.uniform(-2, 2), h * (0.2 + 0.6 * t))
           for t in (i / 13 for i in range(14))]
    d.line(pts, fill=(35, 25, 20), width=max(2, round(w / 110)), joint="curve")
    return im

# --- raya de lapicero sobre huevos SANOS bien clasificados ---
# Con la clase Sucio hay un desenlace nuevo e interesante: que el modelo diga
# "BUENO (SUCIO)", o sea "es una marca, no un dano". Eso seria el acierto ideal.
sanos_ok = [rutas_test[i] for i in range(len(y_test)) if y_test[i] == S and pred[i] == S]
limpios = [Image.open(f).convert("RGB") for f in sanos_ok]
P0 = predecir(limpios); P1 = predecir([rayar(im) for im in limpios])
v_raya = rutear(P1)
volteados = int((v_raya == "DANADO").sum())
print("=== RAYA DE LAPICERO sobre huevos sanos ===")
for v in VEREDICTOS:
    print(f"  -> {v:<12} {int((v_raya == v).sum()):>4} / {len(limpios)}")
print(f"  botados como DANADO solo por la raya: {100*volteados/max(1,len(limpios)):.1f}%")
print(f"  subida media de p(Roto): {np.mean(P1[:, R] - P0[:, R]):+.3f}")
print(f"  (el detector: 6/20 = 30% a descarte, subida media +0.261)\n")

# --- desenfoque sobre huevos ROTOS (radio escalado: equivalente al de una imagen de 640 px) ---
rotos = [Image.open(rutas_test[i]).convert("RGB") for i in range(len(y_test)) if y_test[i] == R]
print("=== DESENFOQUE sobre huevos rotos ===")
print(f"{'radio equiv. 640px':>19}{'p(Roto)':>9}{'DANADO':>9}{'REVISION':>10}{'BUENO*':>8}{'no DANADO':>11}")
ROBUSTEZ = {"raya_pct": 100 * volteados / max(1, len(limpios)),
            "raya_veredictos": {v: int((v_raya == v).sum()) for v in VEREDICTOS},
            "blur": {}, "brillo": {}}
for r640 in [0, 1, 2, 3, 4, 6]:
    ims = [im.filter(ImageFilter.GaussianBlur(r640 * im.width / 640)) if r640 else im for im in rotos]
    P = predecir(ims); rr = rutear(P)
    no_danado = 100 * (rr != "DANADO").mean()
    ROBUSTEZ["blur"][r640] = no_danado
    bueno = int(np.isin(rr, ["BUENO", "BUENO_SUCIO"]).sum())
    print(f"{r640:>19}{P[:, R].mean():>9.3f}{int((rr == 'DANADO').sum()):>9}{int((rr == 'REVISION').sum()):>10}{bueno:>8}{no_danado:>10.1f}%")
print("  BUENO* = BUENO o BUENO (SUCIO): esos si llegarian al empaque")
print("  (el detector, misma definicion: radio 3 -> 12/25 rotos NO descartados = 48%)\n")

# --- brillo ---
print("=== EXPOSICION sobre huevos rotos ===")
print(f"{'brillo':>8}{'p(Roto)':>9}{'no DANADO':>11}")
for f in [0.35, 0.5, 0.7, 1.0, 1.4, 1.8, 2.4]:
    P = predecir([ImageEnhance.Brightness(im).enhance(f) for im in rotos]); rr = rutear(P)
    ROBUSTEZ["brillo"][f] = 100 * (rr != "DANADO").mean()
    print(f"{f:>8.2f}{P[:, R].mean():>9.3f}{ROBUSTEZ['brillo'][f]:>10.1f}%")
print("  (el detector, misma definicion: brillo 2.4x -> 15/25 rotos NO descartados = 60%)")
""")

code(r"""
# @title 13. Exportar a ONNX y verificar que da lo mismo que PyTorch
import onnxruntime as ort
onnx_path = modelo.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True, dynamic=False)
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
print("entrada:", sess.get_inputs()[0].name, sess.get_inputs()[0].shape,
      "| salida:", sess.get_outputs()[0].name, sess.get_outputs()[0].shape)

def preproc(ruta):
    # Replica EXACTA de classify_transforms (ultralytics 8.4.157) para que el backend haga lo mismo:
    # 1) lado corto -> 224 (bilineal)  2) recorte central 224x224  3) /255, RGB, CHW. Sin normalizar media.
    im = Image.open(ruta).convert("RGB"); w, h = im.size
    if w <= h: nw, nh = IMGSZ, int(IMGSZ * h / w)
    else:      nw, nh = int(IMGSZ * w / h), IMGSZ
    im = im.resize((nw, nh), Image.BILINEAR)
    izq, arr = int(round((nw - IMGSZ) / 2)), int(round((nh - IMGSZ) / 2))
    im = im.crop((izq, arr, izq + IMGSZ, arr + IMGSZ))
    return (np.asarray(im, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]

n = min(200, len(rutas_test))
P_onnx = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: preproc(f)})[0] for f in rutas_test[:n]])
dif = np.abs(P_onnx - P_test[:n]).max()
coinc = (P_onnx.argmax(1) == P_test[:n].argmax(1)).mean()
print(f"\nONNX vs PyTorch en {n} imagenes: coincidencia top-1 {100*coinc:.1f}%   dif. maxima {dif:.2e}")
assert coinc >= 0.99, "El ONNX no reproduce al modelo: revisar el preprocesamiento"
print("OK: el backend puede usar este ONNX con esa receta de preprocesamiento")
""")

code(r"""
# @title 14. Guardar todo (Drive o descarga)
import shutil, json
PAQ = Path("/content") / RUN.name       # RUN.name: si se reanudo, es el nombre del run original
if PAQ.exists(): shutil.rmtree(PAQ)
(PAQ / "weights").mkdir(parents=True)
shutil.copy2(RUN / "weights" / "best.pt", PAQ / "weights" / "best.pt")
shutil.copy2(onnx_path, PAQ / "weights" / "best.onnx")
for f in ["results.csv", "results.png", "args.yaml", "confusion_test.png",
          "confusion_matrix.png", "confusion_matrix_normalized.png"]:
    if (RUN / f).exists(): shutil.copy2(RUN / f, PAQ / f)
shutil.copy2(LIMPIO / "manifiesto_split.csv", PAQ / "manifiesto_split.csv")
receta = {"preprocesamiento": "RGB; lado corto a %d bilineal; recorte central %dx%d; /255; CHW; sin normalizar" % (IMGSZ, IMGSZ, IMGSZ),
          "salida": "probabilidades softmax en el orden de 'clases'", "ultralytics": "8.4.157"}
(PAQ / "metricas.json").write_text(json.dumps({**METRICAS, "robustez": ROBUSTEZ, "receta_onnx": receta}, indent=2))
zip_path = shutil.make_archive(str(PAQ), "zip", PAQ)
print("paquete:", zip_path, f"({Path(zip_path).stat().st_size/1e6:.1f} MB)")

if GUARDAR_EN_DRIVE:
    # Drive ya quedo montado en la celda 2 (y el run ya vive ahi); solo falta el zip.
    dest = Path("/content/drive/MyDrive/eggdetection"); dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, dest)
    print("copiado a Drive:", dest / Path(zip_path).name)
else:
    from google.colab import files
    files.download(zip_path)
""")

code(r"""
# @title 15. Resumen para el informe
print("=" * 70)
print(f"MODELO  {RUN.name}")
print("=" * 70)
print(f"dataset       CDD_eggs_SJ v{VERSION} (CC BY 4.0), split 70/15/15 por grupos, sin fuga")
print(f"clases        {NOMBRES}  (modo {MODO_CLASES})")
print(f"exactitud     {100*METRICAS['top1_test']:.2f}% top-1 en test ({METRICAS['n_test']} imgs)")
print(f"cortes        DANADO si p(Roto) >= {TAU_ROTO:.2f} | REVISION entre {TAU_DUDA:.2f} y {TAU_ROTO:.2f} | BUENO por debajo")
print(f"rotos a EMPAQUE   {METRICAS['fuga_a_empaque_pct']:.2f}%  IC95 [{METRICAS['fuga_a_empaque_ic95'][0]:.1f}, {METRICAS['fuga_a_empaque_ic95'][1]:.1f}]   <- el KPI del proyecto")
print(f"rotos no DANADO   {METRICAS['fuga_pct']:.2f}%  (el resto va a REVISION)")
print(f"buenos a REVISION {METRICAS['buenos_a_revision_pct']:.1f}%   buenos botados {METRICAS['buenos_descartados_pct']:.1f}%")
print()
print(f"{'prueba (menor es mejor)':<38}{'detector YOLOv8m':>18}{'este modelo':>14}")
print(f"{'sanos con raya -> DANADO':<38}{'30.0%':>18}{ROBUSTEZ['raya_pct']:>13.1f}%")
print(f"{'rotos NO DANADO, desenfoque r3':<38}{'48.0%':>18}{ROBUSTEZ['blur'][3]:>13.1f}%")
print(f"{'rotos NO DANADO, brillo 2.4x':<38}{'60.0%':>18}{ROBUSTEZ['brillo'][2.4]:>13.1f}%")
print("=" * 70)
print("Detector: medido sobre el test del dataset 'huevos' (20 sanos / 25 rotos, ver docs/ANALISIS_CAMPO.md).")
print("Son datasets distintos: la comparacion es indicativa, no un A/B controlado.")
""")

code(r"""
# @title 16. Prueba con TUS fotos de huevos (desde el celular o el PC)
# Encuadra el huevo AL CENTRO y que ocupe buena parte de la foto: el modelo mira solo
# el cuadrado central de la imagen (en produccion ese recorte lo hace el detector).
# Puedes volver a correr esta celda cuantas veces quieras.
import io
from PIL import ImageOps
from google.colab import files

ETIQUETA = {"BUENO": "✓ BUENO", "BUENO_SUCIO": "✓ BUENO (SUCIO)", "REVISION": "⚠ REVISIÓN", "DANADO": "✗ DAÑADO"}
# El color marca la decision de fondo (pasa / duda / se bota); el borde punteado marca
# "sucio" dentro de los buenos. La etiqueta de texto siempre dice el veredicto completo.
COLOR = {"BUENO": "#0ca30c", "BUENO_SUCIO": "#0ca30c", "REVISION": "#fab219", "DANADO": "#d03b3b"}
TRAZO = {"BUENO_SUCIO": (0, (4, 3))}

subidas = files.upload()   # boton "Elegir archivos": puedes seleccionar varias fotos a la vez
if not subidas:
    print("No subiste fotos. Vuelve a correr esta celda cuando quieras probar.")
else:
    # exif_transpose: las fotos del celular vienen giradas por metadatos
    imgs = [ImageOps.exif_transpose(Image.open(io.BytesIO(b))).convert("RGB") for b in subidas.values()]
    vistas = []
    for im in imgs:                       # lo que realmente ve el modelo
        w, h = im.size; lado = min(w, h); x0, y0 = (w - lado) // 2, (h - lado) // 2
        vistas.append(im.crop((x0, y0, x0 + lado, y0 + lado)))
    P = predecir(imgs)

    cols = min(4, len(imgs)); filas = -(-len(imgs) // cols)
    fig, ejes = plt.subplots(filas, cols, figsize=(4 * cols, 4.9 * filas), squeeze=False)
    for ax in ejes.flat:
        ax.set_visible(False)
    for ax, nombre, vista, p in zip(ejes.flat, subidas, vistas, P):
        v = veredicto(p)
        ax.set_visible(True); ax.imshow(vista); ax.set_xticks([]); ax.set_yticks([])
        for borde in ax.spines.values():
            borde.set_edgecolor(COLOR[v]); borde.set_linewidth(7); borde.set_linestyle(TRAZO.get(v, "-"))
        probs = "   ".join(f"{NOMBRES[i]} {p[i]:.0%}" for i in range(len(NOMBRES)))
        ax.set_title(f"{ETIQUETA[v]}\n{probs}", fontsize=10.5, color="#0b0b0b")
        ax.set_xlabel(nombre[:36], fontsize=8, color="#52514e")
    plt.tight_layout(); plt.show()

    print(f"cortes: DANADO si p(Roto) >= {TAU_ROTO:.2f} | REVISION si >= {TAU_DUDA:.2f}\n")
    for nombre, p in zip(subidas, P):
        print(f"{ETIQUETA[veredicto(p)]:<18} p(Roto)={p[R]:.3f}   {nombre}")
""")

md(r"""
## Cómo encaja en el proyecto

Este modelo **no reemplaza** al detector: lo complementa. La arquitectura recomendada es de **dos etapas**:

```
frame → detector YOLOv8m (¿dónde hay huevos?) → recorte CUADRADO con margen
      → clasificador YOLOv8m-cls (¿cómo está?) → política de ruteo → banda
```

- El recorte debe ser **cuadrado**: el preprocesamiento hace recorte central, y en un recorte
  rectangular cortaría parte del huevo.
- Dos modelos entrenados con **datasets independientes** cometen errores menos correlacionados.
  Cuando discrepan (el detector dice Sano y el clasificador Roto, o al revés), eso es una señal
  natural para mandar a `REVISION`.
- El clasificador a 224 px cuesta ~5 GFLOPs contra ~79 del detector a 640: añade poca latencia.

## Opcional: la versión más limpia posible (fork)

La v2 publicada viene **estirada a 224×224** (deforma el óvalo) y con **shear ±10° horneado**.
Para entrenar sobre los originales:

1. En la página del dataset → **Fork Dataset** (gratis, queda en tu workspace).
2. En tu copia → **Generate New Version**:
   - Preprocesamiento: *Auto-Orient* y *Resize → Fit (black edges) in 320×320* (sin estirar).
   - Aumentaciones: **ninguna** (Ultralytics aumenta en línea, distinto en cada época, y val/test quedan limpios).
3. Cambia `WORKSPACE`, `PROYECTO` y `VERSION` en la celda 2 por los de tu copia, y `IMGSZ = 320`.

El split por grupos de este notebook se sigue aplicando igual.

## Cita (obligatoria por la licencia CC BY 4.0)

```
@misc{cdd_eggs_sj_dataset,
  title = {CDD_eggs_SJ Dataset}, type = {Open Source Dataset}, author = {SJ},
  howpublished = {\url{https://universe.roboflow.com/sj-ytwh0/cdd_eggs_sj}},
  journal = {Roboflow Universe}, publisher = {Roboflow}, year = {2026}, month = {sep},
}
```
""")

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4", "name": "entrenar_cdd_eggs_yolov8cls.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": celdas,
}
salida = AQUI / "entrenar_cdd_eggs_yolov8cls.ipynb"
salida.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"generado: {salida}  ({len(celdas)} celdas)")
