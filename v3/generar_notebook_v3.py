"""Genera v3/entrenar_v3_yolov8.ipynb incrustando los modulos PROBADOS.

    python v3/generar_notebook_v3.py
"""
import json
from pathlib import Path

AQUI = Path(__file__).parent
MODULOS = {m: (AQUI / f"{m}.py").read_text(encoding="utf-8")
           for m in ("v3_datos_lib", "v3_entrenamiento", "v3_inferencia")}
celdas = []


def md(t):
    celdas.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})


def code(t):
    celdas.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                   "source": t.strip("\n").splitlines(True)})


# =============================================================================
md(r"""
# Modelo v3 — detector de huevos Roto / Sano (YOLOv8)

**Qué lo hace distinto de los detectores de huevos que ya existen**

1. **Distingue marcas de tinta de fisuras reales.** Los huevos comerciales vienen sellados (fecha,
   código de granja) y en campo se les raya o ensucia. El modelo 1 bota el **56 %** de los huevos
   sanos solo por una raya de lapicero. Aquí se entrena con **negativos duros sintéticos**: sellos,
   rayas y suciedad dibujados sobre huevos *sanos*, etiquetados como sanos. Se mide antes y después.
2. **Robustez medida a la foto real**: desenfoque, cámara movida, sobre y subexposición y JPEG del
   celular, con aumentaciones apuntadas a las fallas que se midieron en campo.
3. **Desempeño por color de cáscara** (blanco / marrón): una grieta contrasta distinto en cada una.
   El color se estima del recorte y se valida contra fuentes que traen el color real.
4. **Curación multi-fuente auditada**: 9 datasets públicos combinados con detección automática de su
   convención de etiquetado, deduplicación entre fuentes (muchos proyectos son copias de otros) y un
   **benchmark común con el modelo 1** que nunca se toca en el entrenamiento.

**Split: 80/10/10** si tras deduplicar quedan ≥ 3.000 originales (lo esperado: ~4.000+); si no,
**70/15/15**. La regla está en la celda 5 y el número real se imprime.

**Modelo: YOLOv8l a 640**, entrenamiento largo: mínimo 150 épocas; después, si pasan 30 épocas sin una
mejora significativa de mAP50-95, **no corta en seco**: entra en una fase final de 20 épocas sin mosaico
y con la tasa de aprendizaje bajando a su mínimo, y termina. Tope 300 épocas.

## Antes de empezar
1. **Entorno de ejecución → Cambiar tipo de entorno → GPU** (T4 sirve; A100/L4 si tienes Colab Pro).
2. Secreto `ROBOFLOW_API_KEY` con tu **Private API Key** (panel 🔑, con *Acceso al notebook*).
3. Sube **`huevos.zip`** (tu primer dataset, está en `D:\eggdetection\huevos.zip`) a
   **`MyDrive/eggdetection/huevos.zip`**. Su test es el benchmark común con el modelo 1.
4. **Ejecutar todas.** Pide el permiso de Drive al inicio. **Es largo.** Referencia medida: tu modelo 1
   (YOLOv8m, 678 fotos) tardó 23 s/época en Colab. La v3 tiene ~6–7 veces más fotos de train y YOLOv8l
   cuesta ~2 veces más → **~5 min/época en una T4: de 14 a 25 horas** (L4 ≈ 2× más rápida, A100 ≈ 4–5×).
   Desde la época 2, la celda 11 imprime el ritmo **real** y cuánto falta. Si Colab se desconecta,
   vuelve a correr 1–10 y reanuda en la celda 11 con `REANUDAR`: todo se guarda en Drive época a época.
""")

code(r"""
# @title 1. Entorno (mismas versiones probadas localmente con un entrenamiento real)
import subprocess
gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
print(gpu or "SIN GPU -> Entorno de ejecucion > Cambiar tipo de entorno > GPU")
assert gpu, "Activa la GPU antes de seguir"
%pip install -q ultralytics==8.4.157 roboflow==1.5.0 albumentations==2.0.8 onnx onnxruntime onnxslim
# roboflow 1.5.0 pide numpy<2.4: si pip lo cambio con numpy ya cargado, seguir mezclaria dos versiones.
import sys
nuevo = subprocess.run([sys.executable, "-c", "import numpy; print(numpy.__version__)"], capture_output=True, text=True).stdout.strip()
if "numpy" in sys.modules and sys.modules["numpy"].__version__ != nuevo:
    raise RuntimeError(f"pip cambio numpy {sys.modules['numpy'].__version__} -> {nuevo}. Entorno de ejecucion > "
                       "Reiniciar sesion y vuelve a 'Ejecutar todas' (la segunda vez la instalacion no cambia nada).")
import ultralytics
ultralytics.checks()
""")

code(r"""
# @title 2. Configuración
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------- fuentes
# mapa: clase de la fuente -> "Roto" | "Sano" | "EXCLUIR" (es un huevo que no se puede mapear:
#       la imagen entera sale) | "IGNORAR" (no es un huevo: se borra solo la caja).
# convencion: "auto" la detecta la auditoria (celda 4); "huevo"/"region" la fuerzan.
# color: clase -> color real de cascara (solo algunas fuentes lo traen; sirven para calibrar).
FUENTES = [
    dict(nombre="huevos", local="/content/drive/MyDrive/eggdetection/huevos.zip",
         mapa={"Roto": "Roto", "Sano": "Sano"}, convencion="huevo"),
    dict(nombre="indra", ws="indra-bayu", proyecto="egg-shell-cracked-detections", version=1,
         mapa={"egg": "Sano", "cracked": "Roto", "0": "EXCLUIR"}, convencion="auto"),
    dict(nombre="360a", ws="360digitmg-swnfx", proyecto="egg-crack-detection-yjhsy", version=1,
         mapa={"Cracked Egg Brown": "Roto", "Cracked Egg White": "Roto", "Normal Egg Brown": "Sano", "Normal Egg White": "Sano"},
         color={"Cracked Egg Brown": "marron", "Normal Egg Brown": "marron", "Cracked Egg White": "blanco", "Normal Egg White": "blanco"},
         convencion="auto"),
    dict(nombre="360b", ws="digit360mg", proyecto="egg-crack-detection-fahfj", version=1,
         mapa={"Brown cracked egg": "Roto", "White cracked egg": "Roto", "Brown egg": "Sano", "White egg": "Sano"},
         color={"Brown cracked egg": "marron", "Brown egg": "marron", "White cracked egg": "blanco", "White egg": "blanco"},
         convencion="auto"),
    dict(nombre="jqutx", ws="new-workspace-jqutx", proyecto="egg-crack-detection-ek7qj", version=3,
         mapa={"crack": "Roto", "normal": "Sano"}, convencion="auto"),
    dict(nombre="carlos", ws="carlos-oliveira-5vnxi", proyecto="crack-egg-8wlgl", version=2,
         mapa={"crack": "Roto", "good": "Sano"}, convencion="auto"),
    dict(nombre="aau", ws="aau-nupnz", proyecto="egg-crack-9lbqk", version=1,
         mapa={"crack": "Roto", "egg": "Sano"}, convencion="auto"),
    dict(nombre="cracked", ws="crackedeggs", proyecto="cracked-eggs", version=1,
         mapa={"cracked-egg": "Roto", "normal-egg": "Sano"}, convencion="auto"),
    dict(nombre="ducks", ws="ducks-sfodd", proyecto="egg-quality-o7xyt", version=7,   # segmentacion -> cajas
         mapa={"egg-clean": "Sano", "egg-cracked": "Roto", "egg-stained": "Sano",        # manchado = negativo duro real
               "clean": "EXCLUIR", "duck-clean": "EXCLUIR", "duck-cracked": "EXCLUIR", "duck-stained": "EXCLUIR",
               "reject": "EXCLUIR"}, convencion="auto"),
]
PRIORIDAD = [f["nombre"] for f in FUENTES]     # ante la MISMA foto en dos fuentes, gana la primera

# ---------------------------------------------------------------- modelo
# yolov8l: ~5 min/epoca en T4 (14-25 h). yolov8m: ~2,4 min/epoca (7-12 h), algo menos preciso y la
# mitad de latencia en el servidor (~0,6 s por foto en el t2.large contra ~1,3 s).
MODELO_BASE = "yolov8l.pt"  # @param ["yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
IMGSZ = 640
BATCH = {"yolov8m.pt": 32, "yolov8l.pt": 16, "yolov8x.pt": 8}[MODELO_BASE]   # T4 16 GB con AMP

# Regla de las sesiones anteriores + cierre ordenado (celda 11)
MIN_EPOCAS     = 150    # @param {type:"integer"}
PACIENCIA      = 30     # @param {type:"integer"}
MEJORA_MINIMA  = 0.003  # @param {type:"number"}   # 0.3 puntos de mAP50-95
EPOCAS_FINALES = 20     # @param {type:"integer"}  # fase final sin mosaico, LR al minimo
EPOCAS_MAX     = 300    # @param {type:"integer"}

# ---------------------------------------------------------------- datos
FRACCION_NEGATIVOS = 0.35   # fraccion de imagenes de train con huevos sanos que reciben una copia marcada
SEMILLA = 42
COSTO_FN, COSTO_FP = 12.0, 1.0          # dejar pasar un roto cuesta 12 veces botar uno sano
MAX_REVISION_SANOS = 0.05                # banda de revision: <=5 % de los sanos de val
CONF_MIN, IOU_NMS, NITIDEZ_MIN = 0.15, 0.45, 4.0   # mismos del backend

TRABAJO = Path("/content/v3"); FUENTES_DIR = TRABAJO / "fuentes"; DATASET = TRABAJO / "dataset"
from google.colab import drive
drive.mount("/content/drive")
CARPETA_RUNS = Path("/content/drive/MyDrive/eggdetection/runs_v3"); CARPETA_RUNS.mkdir(parents=True, exist_ok=True)
NOMBRE_RUN = f"v3_{Path(MODELO_BASE).stem}_{datetime.now():%Y%m%d_%H%M}"
print("run:", NOMBRE_RUN, "->", CARPETA_RUNS / NOMBRE_RUN)
""")

for nombre, desc in (("v3_datos_lib", "curación multi-fuente, auditoría, dedup, split y negativos duros"),
                     ("v3_entrenamiento", "regla de parada con cierre ordenado y aumentaciones fotométricas"),
                     ("v3_inferencia", "inferencia idéntica a la del backend (probada: probabilidades iguales bit a bit)")):
    md(f"### Módulo `{nombre}.py` — {desc}\nArchivo probado localmente; se escribe tal cual.")
    code(f"%%writefile {nombre}.py\n" + MODULOS[nombre])

code(r"""
# @title 3. Descargar las fuentes
import zipfile, shutil, importlib
import v3_datos_lib as L; importlib.reload(L)
from google.colab import userdata
from roboflow import Roboflow
rf = Roboflow(api_key=userdata.get("ROBOFLOW_API_KEY"))

if FUENTES_DIR.exists(): shutil.rmtree(FUENTES_DIR)
FUENTES_DIR.mkdir(parents=True)
disponibles = []
for f in FUENTES:
    destino = FUENTES_DIR / f["nombre"]
    try:
        if "local" in f:
            with zipfile.ZipFile(f["local"]) as z:
                z.extractall(destino)
        else:
            proy = rf.workspace(f["ws"]).project(f["proyecto"])
            proy.version(f["version"]).download("yolov8", location=str(destino), overwrite=True)
            print(f"  licencia de {f['nombre']}: {getattr(proy, 'license', '?')}")
        disponibles.append(f)
        print(f"OK  {f['nombre']}")
    except Exception as e:
        print(f"--  {f['nombre']}: no disponible ({type(e).__name__}: {str(e)[:120]}). Se sigue sin ella.")
assert any(f["nombre"] == "huevos" for f in disponibles), "Falta huevos.zip en MyDrive/eggdetection/: es el benchmark comun"
""")

code(r"""
# @title 4. Auditoría por fuente: convención de etiquetado y qué se excluye
import pandas as pd
from collections import Counter
items, filas, auditoria = [], [], {}
for f in disponibles:
    its = L.inventariar(FUENTES_DIR / f["nombre"], f["nombre"])
    au = L.auditar(its, f["mapa"])
    conv = au["convencion"] if f["convencion"] == "auto" else f["convencion"]
    st = L.mapear(its, f["mapa"], conv, f.get("color"))
    buenos = [it for it in its if not it["excluir"] and it["cajas"]]
    auditoria[f["nombre"]] = {**au, "convencion_usada": conv, "descartes": st}
    filas.append({"fuente": f["nombre"], "imagenes": len(its), "originales": au["originales"],
                  "convencion": conv + (" (auto)" if f["convencion"] == "auto" else ""),
                  "IoA roto/sano": au["ioa_roto_en_sano"], "excluidas": st.get("imagenes_excluidas", 0),
                  "usables": len(buenos), "roto": sum(c[0] == "Roto" for it in buenos for c in it["cajas"]),
                  "sano": sum(c[0] == "Sano" for it in buenos for c in it["cajas"])})
    sin = {k: v for k, v in st.items() if k.startswith("clase sin mapeo")}
    if sin:
        print(f"ATENCION {f['nombre']}: clases no previstas -> imagenes excluidas: {sin}")
    items += buenos
display(pd.DataFrame(filas).set_index("fuente"))
print("\n'region' = la fuente encierra solo la GRIETA dentro de una caja de huevo: se convierte a caja de huevo roto.")
print("Clases por fuente (cajas, area mediana relativa):")
for n, a in auditoria.items():
    print(f"  {n:<8} " + " | ".join(f"{k}: {v['cajas']} ({v['area_mediana']:.3f})" for k, v in a["clases"].items()))
""")

code(r"""
# @title 5. Deduplicar entre fuentes, agrupar y repartir (con benchmark intocable)
import csv
L.calcular_hashes(items)
items = L.deduplicar_entre_fuentes(items, PRIORIDAD, umbral=4)
info = L.agrupar(items, umbral_hash=16, max_grupo_hash=30)
print("uniones 'mismo huevo':", dict(info["razones"]), "| rechazadas por tope:", info["rechazadas_por_tope"])

es_benchmark = lambda it: it["fuente"] == "huevos" and it["split_orig"] == "test"
n_orig = len({it["base"] for it in items if not es_benchmark(it)})
PROPORCIONES = {"train": 0.80, "val": 0.10, "test": 0.10} if n_orig >= 3000 else {"train": 0.70, "val": 0.15, "test": 0.15}
print(f"\n{n_orig} originales independientes (sin benchmark) -> split "
      + "/".join(f"{int(100 * v)}" for v in PROPORCIONES.values()))

asignado = L.repartir(items, PROPORCIONES, benchmark=es_benchmark, semilla=SEMILLA)
reparto = L.revisar_reparto(items, asignado, PROPORCIONES)     # se detiene si queda desbalanceado
final = L.materializar(items, asignado, DATASET)
data_yaml = L.escribir_data_yaml(DATASET, list(final))
print("reparto de originales:", {k: f"{100 * v:.1f}%" for k, v in reparto.items()})
for s in ("train", "val", "test", "benchmark"):
    lst = final.get(s, [])
    print(f"  {s:<9} {len(lst):>5} imagenes | Roto {sum(c[0]=='Roto' for it in lst for c in it['cajas']):>5}"
          f" | Sano {sum(c[0]=='Sano' for it in lst for c in it['cajas']):>5} | fuentes {dict(Counter(it['fuente'] for it in lst))}")
print("\n=== VERIFICACION ===")
assert L.verificar_sin_fuga(final, bloquear_visual=False), "Fuga DURA: un original o grupo quedo en dos splits"
print("SIN FUGA DURA entre train, val, test y benchmark (el chequeo visual de arriba es informativo)")
with open(TRABAJO / "manifiesto_v3.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["split", "fuente", "archivo", "original", "grupo", "roto", "sano"])
    for s, lst in final.items():
        for it in lst:
            w.writerow([s, it["fuente"], it["ruta_final"].name, it["base"], it["grupo"],
                        sum(c[0] == "Roto" for c in it["cajas"]), sum(c[0] == "Sano" for c in it["cajas"])])
""")

code(r"""
# @title 6. Color de cáscara: calibrado con las fuentes que traen el color real
from PIL import Image
muestras = []
for s in ("train", "val"):
    for it in final.get(s, []):
        cc = [c for c in it["cajas"] if c[5]]
        if cc:
            im = Image.open(it["ruta_final"]).convert("RGB")
            muestras += [(*L.rasgos_color(im, c), c[5]) for c in cc]
if len(muestras) >= 40:
    COLOR = L.calibrar_color(muestras)
    print(f"umbral de saturacion {COLOR['umbral_saturacion']:.2f} | exactitud en la mitad NO usada para ajustar: "
          f"{100 * COLOR['exactitud_prueba']:.1f}% (n={COLOR['n_prueba']})")
    if COLOR["exactitud_prueba"] < 0.9:
        print("ATENCION: exactitud < 90%: el desglose por color es orientativo.")
else:
    COLOR = None
    print("Sin fuentes con color real disponibles: se omite el desglose por color.")

def color_de(im, caja):
    if COLOR is None:
        return "?"
    return "marron" if L.rasgos_color(im, caja)[0] >= COLOR["umbral_saturacion"] else "blanco"
""")

code(r"""
# @title 7. Negativos duros sintéticos (solo train) y vista previa
import matplotlib.pyplot as plt, random
NEG = L.generar_negativos_duros(final["train"], DATASET, fraccion=FRACCION_NEGATIVOS, semilla=SEMILLA)
negs = sorted((DATASET / "train" / "images").glob("*__neg.jpg"))
random.seed(SEMILLA)
fig, ejes = plt.subplots(2, 6, figsize=(18, 6.5))
for ax, f in zip(ejes.flat, random.sample(negs, min(12, len(negs)))):
    ax.imshow(Image.open(f)); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Negativos duros: huevos SANOS con sello, raya o suciedad (etiquetados como Sano)"); plt.tight_layout(); plt.show()
""")

code(r"""
# @title 8. Vista previa con cajas, una fila por fuente (revisa que las etiquetas tengan sentido)
from PIL import ImageDraw
fuentes_vistas = sorted({it["fuente"] for it in final["train"]})
fig, ejes = plt.subplots(len(fuentes_vistas), 5, figsize=(17, 3.4 * len(fuentes_vistas)), squeeze=False)
for fila, fu in enumerate(fuentes_vistas):
    cand = [it for it in final["train"] if it["fuente"] == fu]
    for col, it in enumerate(random.sample(cand, min(5, len(cand)))):
        im = Image.open(it["ruta_final"]).convert("RGB"); d = ImageDraw.Draw(im); W, H = im.size
        for c in it["cajas"]:
            x1, y1, x2, y2 = (c[1] - c[3] / 2) * W, (c[2] - c[4] / 2) * H, (c[1] + c[3] / 2) * W, (c[2] + c[4] / 2) * H
            d.rectangle([x1, y1, x2, y2], outline="#d03b3b" if c[0] == "Roto" else "#0ca30c", width=max(2, W // 160))
        ax = ejes[fila][col]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        if col == 0: ax.set_ylabel(fu, fontsize=12)
plt.suptitle("Rojo = Roto, verde = Sano"); plt.tight_layout(); plt.show()
""")

code(r"""
# @title 9. Desbalance de clases -> peso por clase
n_roto = sum(c[0] == "Roto" for it in final["train"] for c in it["cajas"])
n_sano = sum(c[0] == "Sano" for it in final["train"] for c in it["cajas"]) + NEG["imagenes"]
razon = max(n_roto, n_sano) / max(1, min(n_roto, n_sano))
CLS_PW = 0.5 if razon > 1.5 else 0.0   # cls_pw SI funciona en deteccion (8.4.157), a diferencia de clasificacion
print(f"train: Roto {n_roto} | Sano ~{n_sano} (con negativos) | razon {razon:.2f} -> cls_pw={CLS_PW}")
""")

code(r"""
# @title 10. Módulos de entrenamiento e inferencia
import v3_entrenamiento as V, v3_inferencia as VI
importlib.reload(V); importlib.reload(VI)
print("aumentaciones:", ", ".join(type(t).__name__ for t in V.aumentaciones_fotometricas()))
""")

code(r"""
# @title 11. Entrenar (o reanudar): mínimo 150 épocas, cierre ordenado tras la meseta
# Si Colab se desconecto: corre 1-10 (recrean el mismo dataset: todo usa SEMILLA), pega aqui el nombre
# de la carpeta del run en MyDrive/eggdetection/runs_v3/ y sigue desde esta celda.
REANUDAR = ""  # @param {type:"string"}
from ultralytics import YOLO
parada = V.crear_parada_ordenada(MIN_EPOCAS, PACIENCIA, MEJORA_MINIMA, EPOCAS_FINALES)
if REANUDAR:
    RUN = CARPETA_RUNS / REANUDAR
    modelo = YOLO(str(RUN / "weights" / "last.pt"))
    V.registrar(modelo, parada)   # al inicio (reanudar en plena fase final) y al final de cada epoca
    modelo.train(resume=True)
else:
    modelo = YOLO(MODELO_BASE)
    V.registrar(modelo, parada)   # al inicio (reanudar en plena fase final) y al final de cada epoca
    modelo.train(
        data=str(data_yaml), imgsz=IMGSZ, batch=BATCH, epochs=EPOCAS_MAX,
        patience=0, close_mosaic=EPOCAS_FINALES,             # la regla y el cierre los maneja el callback
        optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True, warmup_epochs=3, weight_decay=0.0005,
        box=7.5, cls=0.7, dfl=1.5, cls_pw=CLS_PW,
        # geometria del huevo intacta: sin shear, perspectiva ni mixup (igual que el modelo 1)
        degrees=10.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.0, copy_paste=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        augmentations=V.aumentaciones_fotometricas(),
        cache="disk", workers=2, seed=SEMILLA, deterministic=False, amp=True,
        project=str(CARPETA_RUNS), name=NOMBRE_RUN, exist_ok=True, plots=True,
    )
    RUN = Path(modelo.trainer.save_dir)
print("\npesos:", RUN / "weights" / "best.pt")
""")

code(r"""
# @title 12. Curvas de entrenamiento
df = pd.read_csv(RUN / "results.csv"); df.columns = [c.strip() for c in df.columns]
c50 = next(c for c in df.columns if "mAP50(B)" in c); c95 = next(c for c in df.columns if "mAP50-95" in c)
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4))
a.plot(df["epoch"], df[c50], label="mAP50"); a.plot(df["epoch"], df[c95], label="mAP50-95"); a.legend(); a.set_title("val")
for k in ("train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss"):
    if k in df: b.plot(df["epoch"], df[k], label=k)
b.legend(); b.set_title("pérdidas"); plt.tight_layout(); plt.show()
i = int(df[c95].idxmax())
print(f"{len(df)} epocas | mejor mAP50-95 {df.loc[i, c95]:.4f} (mAP50 {df.loc[i, c50]:.4f}) en la epoca {int(df.loc[i, 'epoch'])}")
""")

code(r"""
# @title 13. Evaluación como en producción: umbrales calibrados en VAL, medidos en TEST y BENCHMARK
import numpy as np, torch
DISP = "cuda" if torch.cuda.is_available() else "cpu"
modelo = YOLO(str(RUN / "weights" / "best.pt"))
red = modelo.model.float().eval().to(DISP)

def predecir(im):
    x, r, dw, dh = VI.letterbox(im, IMGSZ)
    with torch.no_grad():
        o = red(torch.from_numpy(x).to(DISP)); o = o[0] if isinstance(o, (list, tuple)) else o
    return VI.postprocesar(o.float().cpu().numpy(), r, dw, dh, im.width, im.height, CONF_MIN, IOU_NMS)

def huevos_de(it, im):
    W, H = im.size
    return [(c[0], (c[1] - c[3] / 2) * W, (c[2] - c[4] / 2) * H, (c[1] + c[3] / 2) * W, (c[2] + c[4] / 2) * H, c)
            for c in it["cajas"]]

def recolectar(split, transform=None, solo=None):
    filas = []
    for it in final.get(split, []):
        im = Image.open(it["ruta_final"]).convert("RGB")
        gts = huevos_de(it, im)
        if solo and not any(g[0] == solo for g in gts): continue
        im_t = transform(im, gts) if transform else im
        preds = predecir(im_t)
        for g, p in zip(gts, VI.emparejar(gts, preds)):
            if solo and g[0] != solo: continue
            nit = VI.nitidez_recorte(im_t, *p[:4]) if p else None
            filas.append({"clase": g[0], "fuente": it["fuente"], "color": color_de(im, g[5]),
                          "p_roto": p[4] if p else None, "p_sano": p[5] if p else None, "nitidez": nit})
    return filas

VAL = recolectar("val")
roto = [f for f in VAL if f["clase"] == "Roto"]; sano = [f for f in VAL if f["clase"] == "Sano"]
mejor = None
for tau in np.round(np.arange(0.05, 0.96, 0.01), 2):
    fn = sum(f["p_roto"] is None or f["p_roto"] < tau for f in roto)
    fp = sum(f["p_roto"] is not None and f["p_roto"] >= tau for f in sano)
    c = fn * COSTO_FN + fp * COSTO_FP
    if mejor is None or c < mejor[0]: mejor = (c, float(tau))
TAU_ROTO = mejor[1]
ps = sorted(f["p_sano"] for f in sano if f["p_sano"] is not None and f["p_roto"] < TAU_ROTO)
TAU_SANO = float(min(0.95, max(0.50, np.percentile(ps, 100 * MAX_REVISION_SANOS)))) if ps else 0.85
print(f"calibrado en VAL: TAU_ROTO={TAU_ROTO:.2f} (costo minimo {COSTO_FN:g}:{COSTO_FP:g}) | TAU_SANO={TAU_SANO:.2f}")

def rutear(f):
    if f["p_roto"] is None: return "NO_DETECTADO"
    return VI.decidir(f["p_roto"], f["p_sano"], f["nitidez"], TAU_ROTO, TAU_SANO, NITIDEZ_MIN)

def tabla(filas, titulo):
    print(f"\n{titulo}")
    print(f"  {'real':<6}{'DESCARTE':>10}{'REVISION':>10}{'ACEPTADO':>10}{'no det.':>9}{'n':>6}")
    res = {}
    for cl in ("Roto", "Sano"):
        fs = [f for f in filas if f["clase"] == cl]; r = Counter(rutear(f) for f in fs)
        res[cl] = {k: r[k] for k in ("DESCARTE", "REVISION", "ACEPTADO", "NO_DETECTADO")} | {"n": len(fs)}
        print(f"  {cl:<6}{r['DESCARTE']:>10}{r['REVISION']:>10}{r['ACEPTADO']:>10}{r['NO_DETECTADO']:>9}{len(fs):>6}")
    ro = res["Roto"]; lo, hi = L.ic95(ro["ACEPTADO"], ro["n"])
    print(f"  rotos al EMPAQUE: {ro['ACEPTADO']}/{ro['n']} = {100*ro['ACEPTADO']/max(1,ro['n']):.2f}% IC95 [{100*lo:.1f}, {100*hi:.1f}]"
          f" | sanos botados: {res['Sano']['DESCARTE']}/{res['Sano']['n']}")
    return res

METRICAS = {"tau_roto": TAU_ROTO, "tau_sano": TAU_SANO, "conf_min": CONF_MIN, "nitidez_min": NITIDEZ_MIN}
for s in ("test", "benchmark"):
    filas = recolectar(s)
    METRICAS[s] = tabla(filas, f"=== {s.upper()} ({len(final.get(s, []))} imagenes) ===")
    if s == "test":
        print("\n  por fuente (rotos al empaque / rotos | sanos botados / sanos):")
        for fu in sorted({f["fuente"] for f in filas}):
            ro = [f for f in filas if f["fuente"] == fu and f["clase"] == "Roto"]; sa = [f for f in filas if f["fuente"] == fu and f["clase"] == "Sano"]
            print(f"    {fu:<8} {sum(rutear(f)=='ACEPTADO' for f in ro):>3}/{len(ro):<4} | {sum(rutear(f)=='DESCARTE' for f in sa):>3}/{len(sa)}")
        if COLOR:
            print("  por color de cascara:")
            for col in ("blanco", "marron"):
                ro = [f for f in filas if f["color"] == col and f["clase"] == "Roto"]; sa = [f for f in filas if f["color"] == col and f["clase"] == "Sano"]
                print(f"    {col:<7} rotos al empaque {sum(rutear(f)=='ACEPTADO' for f in ro)}/{len(ro)} | sanos botados {sum(rutear(f)=='DESCARTE' for f in sa)}/{len(sa)}")
for s in ("test", "benchmark"):
    yml = DATASET / f"data_{s}.yaml"
    yml.write_text(f"path: {DATASET.as_posix()}\ntrain: train/images\nval: {s}/images\nnc: 2\nnames: ['Roto', 'Sano']\n")
    m = modelo.val(data=str(yml), split="val", imgsz=IMGSZ, batch=BATCH, plots=False, verbose=False)
    METRICAS[s + "_map"] = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)}
    print(f"{s}: mAP50={m.box.map50:.4f}  mAP50-95={m.box.map:.4f}")
""")

code(r"""
# @title 14. Robustez: las pruebas que tumbaron al modelo 1
from PIL import ImageFilter, ImageEnhance, ImageFont

def rayar(im, gts):   # EXACTAMENTE la prueba de diagnostico_campo.py (no es el generador de train)
    im = im.copy(); d = ImageDraw.Draw(im); rng = random.Random(3)
    for g in gts:
        if g[0] != "Sano": continue
        x1, y1, x2, y2 = g[1:5]; cx, alto = (x1 + x2) / 2, y2 - y1
        pts = [(cx + (x2 - x1) * 0.12 * np.sin(t * 3.1) + rng.uniform(-3, 3), y1 + alto * (0.12 + 0.76 * t)) for t in (i / 13 for i in range(14))]
        d.line(pts, fill=(35, 25, 20), width=3, joint="curve")
    return im

def sellar(im, gts):  # sello con fuente, texto y tintas DISTINTOS a los de train
    im = im.copy(); d = ImageDraw.Draw(im)
    for g in gts:
        if g[0] != "Sano": continue
        x1, y1, x2, y2 = g[1:5]
        fnt = ImageFont.load_default(size=max(10, int((y2 - y1) * 0.13)))
        d.text(((x1 + x2) / 2, (y1 + y2) / 2), "EXP 12-26 F7", fill=(200, 110, 20), font=fnt, anchor="mm")
    return im

PRUEBAS = [("raya de lapicero en SANOS", rayar, "Sano"), ("sello de tinta en SANOS", sellar, "Sano"),
           ("desenfoque r=3 en ROTOS", lambda im, g: im.filter(ImageFilter.GaussianBlur(3)), "Roto"),
           ("sobreexposicion 2.4x en ROTOS", lambda im, g: ImageEnhance.Brightness(im).enhance(2.4), "Roto"),
           ("subexposicion 0.35x en ROTOS", lambda im, g: ImageEnhance.Brightness(im).enhance(0.35), "Roto")]
ROBUSTEZ = {}
for s in ("test", "benchmark"):
    print(f"\n=== {s.upper()} ===")
    print(f"  {'prueba':<32}{'DESCARTE':>10}{'REVISION':>10}{'ACEPTADO':>10}{'no det.':>9}{'n':>6}")
    for nombre, tf, cl in PRUEBAS:
        r = Counter(rutear(f) for f in recolectar(s, tf, cl)); n = sum(r.values())
        ROBUSTEZ[f"{s}|{nombre}"] = dict(r) | {"n": n}
        print(f"  {nombre:<32}{r['DESCARTE']:>10}{r['REVISION']:>10}{r['ACEPTADO']:>10}{r['NO_DETECTADO']:>9}{n:>6}")
print("\nModelo 1 en el BENCHMARK (docs/COMPARACION_MODELOS.md):")
print("  raya en sanos -> DESCARTE 24/43 | desenfoque r3 -> 52 desc / 63 rev / 3 empaque / 22 no det (de 140)")
print("  sobreexposicion 2.4x -> 45 desc / 17 rev / 8 empaque / 70 no det (de 140)")
""")

code(r"""
# @title 15. Exportar ONNX para el backend y verificar que da lo mismo
import onnxruntime as ort
onnx_path = modelo.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True, dynamic=False, nms=False)
s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
forma = s.get_outputs()[0].shape; nombres = s.get_modelmeta().custom_metadata_map.get("names")
print("salida:", forma, "| clases:", nombres)
assert list(forma) == [1, 6, 8400] and nombres == "{0: 'Roto', 1: 'Sano'}", "No es compatible con el backend"
dif = 0.0
for it in final["benchmark"][:25]:
    im = Image.open(it["ruta_final"]).convert("RGB"); x, r, dw, dh = VI.letterbox(im, IMGSZ)
    a = VI.postprocesar(s.run(None, {s.get_inputs()[0].name: x})[0], r, dw, dh, im.width, im.height, CONF_MIN, IOU_NMS)
    b = predecir(im)
    dif = max([dif] + [abs(p[4] - q[4]) for p, q in zip(a, b)] + [abs(p[5] - q[5]) for p, q in zip(a, b)])
print(f"ONNX (CPU, como el servidor) vs PyTorch: dif. maxima de probabilidad = {dif:.2e}")
assert dif < 5e-3, "El ONNX no reproduce al modelo"
""")

code(r"""
# @title 16. Guardar el paquete en Drive
import json
PAQ = TRABAJO / RUN.name
if PAQ.exists(): shutil.rmtree(PAQ)
(PAQ / "weights").mkdir(parents=True)
shutil.copy2(RUN / "weights" / "best.pt", PAQ / "weights" / "best.pt"); shutil.copy2(onnx_path, PAQ / "weights" / "best.onnx")
for f in ("results.csv", "results.png", "args.yaml", "confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png"):
    if (RUN / f).exists(): shutil.copy2(RUN / f, PAQ / f)
shutil.copy2(TRABAJO / "manifiesto_v3.csv", PAQ / "manifiesto_v3.csv"); shutil.copy2(data_yaml, PAQ / "data.yaml")
(PAQ / "metricas_v3.json").write_text(json.dumps({**METRICAS, "robustez": ROBUSTEZ, "color": COLOR,
    "negativos_duros": dict(NEG), "auditoria": {k: {kk: vv for kk, vv in v.items() if kk != "clases"} for k, v in auditoria.items()},
    "backend_env": {"TAU_ROTO": TAU_ROTO, "TAU_SANO": TAU_SANO, "CONF_MIN_DETECCION": CONF_MIN}}, indent=2, default=str))
zip_path = shutil.make_archive(str(PAQ), "zip", PAQ)
dest = Path("/content/drive/MyDrive/eggdetection"); shutil.copy2(zip_path, dest)
print("guardado:", dest / Path(zip_path).name, f"({Path(zip_path).stat().st_size / 1e6:.0f} MB)")
""")

code(r"""
# @title 17. Resumen: v3 contra el modelo 1 en el BENCHMARK común
b = METRICAS["benchmark"]; r, s_ = b["Roto"], b["Sano"]
print("=" * 72); print(f"MODELO {RUN.name}"); print("=" * 72)
print(f"{'en el benchmark (mismos 183 huevos)':<40}{'modelo 1':>12}{'v3':>12}")
print(f"{'rotos al empaque':<40}{'0/140':>12}{str(r['ACEPTADO']) + '/' + str(r['n']):>12}")
print(f"{'rotos no detectados':<40}{'8/140':>12}{str(r['NO_DETECTADO']) + '/' + str(r['n']):>12}")
print(f"{'sanos botados':<40}{'0/43':>12}{str(s_['DESCARTE']) + '/' + str(s_['n']):>12}")
rb = ROBUSTEZ.get("benchmark|raya de lapicero en SANOS", {})
print(f"{'sanos con raya -> DESCARTE':<40}{'24/43':>12}{str(rb.get('DESCARTE', 0)) + '/' + str(rb.get('n', 0)):>12}")
bl = ROBUSTEZ.get("benchmark|desenfoque r=3 en ROTOS", {})
print(f"{'rotos desenfocados al empaque':<40}{'3/140':>12}{str(bl.get('ACEPTADO', 0)) + '/' + str(bl.get('n', 0)):>12}")
ov = ROBUSTEZ.get("benchmark|sobreexposicion 2.4x en ROTOS", {})
print(f"{'rotos sobreexpuestos no detectados':<40}{'70/140':>12}{str(ov.get('NO_DETECTADO', 0)) + '/' + str(ov.get('n', 0)):>12}")
print("=" * 72)
print(f"Para el backend (.env): TAU_ROTO={TAU_ROTO:.2f}  TAU_SANO={TAU_SANO:.2f}  CONF_MIN_DETECCION={CONF_MIN}")
""")

code(r"""
# @title 18. Prueba con TUS fotos
import io
from PIL import ImageOps
from google.colab import files
COL = {"DESCARTE": "#d03b3b", "REVISION": "#fab219", "ACEPTADO": "#0ca30c"}
ETQ = {"DESCARTE": "✗ DAÑADO", "REVISION": "⚠ REVISIÓN", "ACEPTADO": "✓ BUENO"}
subidas = files.upload()
imgs = [ImageOps.exif_transpose(Image.open(io.BytesIO(b))).convert("RGB") for b in subidas.values()]
if imgs:
    fig, ejes = plt.subplots(-(-len(imgs) // 3), min(3, len(imgs)), figsize=(6 * min(3, len(imgs)), 6.5 * -(-len(imgs) // 3)), squeeze=False)
    for ax in ejes.flat: ax.set_visible(False)
    for ax, (nombre, im) in zip(ejes.flat, zip(subidas, imgs)):
        d = ImageDraw.Draw(im); lw = max(3, im.width // 150)
        for x1, y1, x2, y2, pr, psn in predecir(im):
            v = VI.decidir(pr, psn, VI.nitidez_recorte(im, x1, y1, x2, y2), TAU_ROTO, TAU_SANO, NITIDEZ_MIN)
            d.rectangle([x1, y1, x2, y2], outline=COL[v], width=lw)
            d.text((x1 + lw, max(0, y1 - 14 * lw // 3)), f"{ETQ[v]}  roto {pr:.0%}", fill=COL[v], font=ImageFont.load_default(size=6 * lw))
        ax.set_visible(True); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(nombre[:40], fontsize=9)
    plt.tight_layout(); plt.show()
""")

md(r"""
## Integración con la app y el backend

1. Descarga el zip de `MyDrive/eggdetection/` y copia `weights/best.onnx` a `backend/modelos/best.onnx`
   (guarda antes el actual como `best_v1.onnx`). **No hay que tocar código**: la salida `[1, 6, 8400]`
   y el orden de clases `{0: Roto, 1: Sano}` son los mismos del modelo 1 (verificado en la celda 15).
2. En `backend/.env` pon los umbrales que imprimió la celda 17 (`TAU_ROTO`, `TAU_SANO`).
3. YOLOv8l es ~2× más pesado que el modelo 1 en CPU: en tu `t2.large` calcula **~1,3 s por foto**.
   Si la banda necesita respuesta más rápida, reentrena con `MODELO_BASE = "yolov8m.pt"`.

## Fuentes (hay que citarlas en el informe; revisa la licencia de cada una en su página)

`indra-bayu/egg-shell-cracked-detections` · `360digitmg-swnfx/egg-crack-detection-yjhsy` ·
`digit360mg/egg-crack-detection-fahfj` · `new-workspace-jqutx/egg-crack-detection-ek7qj` ·
`carlos-oliveira-5vnxi/crack-egg-8wlgl` · `aau-nupnz/egg-crack-9lbqk` · `crackedeggs/cracked-eggs` ·
`ducks-sfodd/egg-quality-o7xyt` — todas en https://universe.roboflow.com
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "cells": celdas}
salida = AQUI / "entrenar_v3_yolov8.ipynb"
salida.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"generado: {salida} ({len(celdas)} celdas)")
