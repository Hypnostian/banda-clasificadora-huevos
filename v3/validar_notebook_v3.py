"""Valida entrenar_v3_yolov8.ipynb de PUNTA A PUNTA con ultralytics real.

Simula solo lo que no existe fuera de Colab (Roboflow, Drive, subir archivos) y
ejecuta TODO lo demas: auditoria, dedup, split, color, negativos duros,
ENTRENAMIENTO real (yolov8n, 160 px, pocas epocas), evaluacion, robustez,
export ONNX, guardado, resumen y la celda de fotos propias.

    <venv con ultralytics>/python v3/validar_notebook_v3.py
"""
import ast
import io
import json
import random
import shutil
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

AQUI = Path(__file__).parent
NB = AQUI / "entrenar_v3_yolov8.ipynb"
fallos = 0


def check(cond, msg, extra=""):
    global fallos
    print(f"  {'[OK]  ' if cond else '[FALLO]'} {msg}{(' -> ' + str(extra)) if extra != '' else ''}", flush=True)
    fallos += 0 if cond else 1


def sin_magias(src):
    out = []
    for ln in src.splitlines():
        s = ln.lstrip()
        if s.startswith("%%"):
            continue
        out.append(ln[: len(ln) - len(s)] + "pass" if s.startswith(("%", "!")) else ln)
    return "\n".join(out)


nb = json.loads(NB.read_text(encoding="utf-8"))
fuentes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

print("=" * 76); print("1. ESTRUCTURA E INTEGRIDAD"); print("=" * 76)
for i, src in enumerate(fuentes):
    malas = [k for k, ln in enumerate(src.splitlines()) if ln.lstrip().startswith("%%") and k != 0]
    if malas:
        check(False, f"celda {i}: magia de celda fuera de la linea 1", malas)
    try:
        ast.parse(sin_magias(src))
    except SyntaxError as e:
        check(False, f"celda {i} compila", e)
check(True, f"{len(fuentes)} celdas de codigo: magias bien ubicadas y todas compilan")
for mod in ("v3_datos_lib", "v3_entrenamiento", "v3_inferencia"):
    src = next(s for s in fuentes if s.startswith(f"%%writefile {mod}.py"))
    check(src.split("\n", 1)[1].rstrip("\n") == (AQUI / f"{mod}.py").read_text(encoding="utf-8").rstrip("\n"),
          f"{mod}.py incrustado identico al archivo probado")

print("\n" + "=" * 76); print("2. ENSAYO DE PUNTA A PUNTA (ultralytics real, datos sinteticos)"); print("=" * 76)
tmp = Path(tempfile.mkdtemp(prefix="v3_nb_"))
CONTENT = tmp / "content"
import os  # noqa: E402
os.chdir(tmp)                     # ultralytics descarga yolov8n.pt en la carpeta actual: no ensuciar el proyecto
rng = random.Random(0)


def escena(semilla, n, roto_prob=0.45):
    r = random.Random(semilla)
    im = Image.new("RGB", (320, 320), tuple(r.randint(60, 200) for _ in range(3)))
    d = ImageDraw.Draw(im)
    for _ in range(8):
        x, y = r.randint(0, 320), r.randint(0, 320)
        d.rectangle([x, y, x + r.randint(10, 60), y + r.randint(10, 60)], fill=tuple(r.randint(0, 255) for _ in range(3)))
    huevos = []
    for k in range(n):
        cx, cy = 80 + (k % 2) * 160 + r.randint(-10, 10), 90 + (k // 2) * 150 + r.randint(-10, 10)
        w, h = r.randint(60, 80), r.randint(80, 100)
        col = r.choice([(190, 125, 80), (238, 232, 222)])
        roto = r.random() < roto_prob
        d.ellipse([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill=col)
        grieta = None
        if roto:
            pts = [(cx + r.uniform(-8, 8), cy - h * .3 + j * h * .12) for j in range(6)]
            d.line(pts, fill=(40, 28, 22), width=3)
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            grieta = (min(xs) - 3, min(ys) - 3, max(xs) + 3, max(ys) + 3)
        huevos.append((cx, cy, w, h, col, roto, grieta))
    return im, huevos


def yl(c, x1, y1, x2, y2, S=320):
    return f"{c} {(x1 + x2) / 2 / S:.6f} {(y1 + y2) / 2 / S:.6f} {(x2 - x1) / S:.6f} {(y2 - y1) / S:.6f}"


def export_sintetico(destino: Path, nombre: str, mapa: dict, n: int, semilla: int, color: dict | None = None):
    """Export YOLO 'como Roboflow' con las clases reales de la fuente."""
    clases = list(mapa)
    roto_c = [c for c in clases if mapa[c] == "Roto"]
    sano_c = [c for c in clases if mapa[c] == "Sano"]
    extra = [c for c in clases if mapa[c] not in ("Roto", "Sano")]
    region = nombre == "aau"
    for i in range(n):
        im, hs = escena(semilla + i, rng.randint(1, 3))
        lin = []
        for cx, cy, w, h, col, roto, grieta in hs:
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if region:
                lin.append(yl(clases.index(sano_c[0]), *box))
                if roto:
                    lin.append(yl(clases.index(roto_c[0]), *grieta))
                continue
            cands = roto_c if roto else sano_c
            if color:
                quiere = "marron" if col[0] < 220 else "blanco"
                cands = [c for c in cands if color.get(c) == quiere] or cands
            c = clases.index(cands[0])
            if nombre == "ducks":                          # poligono de segmentacion
                poly = " ".join(f"{(cx + w / 2 * np.cos(t)) / 320:.5f} {(cy + h / 2 * np.sin(t)) / 320:.5f}"
                                for t in np.linspace(0, 2 * np.pi, 10, endpoint=False))
                lin.append(f"{c} {poly}")
            else:
                lin.append(yl(c, *box))
        if extra and i % 9 == 0:
            lin.append(yl(clases.index(extra[0]), 10, 10, 60, 70))      # clase EXCLUIR: saca la imagen
        split = "train" if i < 0.75 * n else ("valid" if i < 0.88 * n else "test")
        base = f"{nombre}_{i:03d}_jpg.rf.{rng.getrandbits(40):010x}"
        for sub in ("images", "labels"):
            (destino / split / sub).mkdir(parents=True, exist_ok=True)
        im.save(destino / split / "images" / f"{base}.jpg", quality=92)
        (destino / split / "labels" / f"{base}.txt").write_text("".join(x + "\n" for x in lin))
    (destino / "data.yaml").write_text(f"nc: {len(clases)}\nnames: {clases}\n")


# --- google.colab y roboflow simulados
_colab = types.ModuleType("google.colab")
_colab.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
_colab.userdata = types.SimpleNamespace(get=lambda k: "clave-falsa")
_SUBIDAS = {}
_colab.files = types.SimpleNamespace(upload=lambda: dict(_SUBIDAS), download=lambda *a, **k: None)
# En Colab, google.colab y google.protobuf (que usa onnx) conviven en el MISMO paquete de nombres:
# se cuelga colab del paquete real en vez de reemplazarlo.
import google.protobuf  # noqa: E402,F401
sys.modules["google"].colab = _colab
sys.modules["google.colab"] = _colab
CONFIG_FUENTES = {}


class _Version:
    def __init__(self, nombre):
        self.nombre = nombre

    def download(self, fmt, location, overwrite=True):
        f = CONFIG_FUENTES[self.nombre]
        export_sintetico(Path(location), self.nombre, f["mapa"], 34, 7000 + 1000 * list(CONFIG_FUENTES).index(self.nombre), f.get("color"))


class _Proyecto:
    license = "CC BY 4.0 (simulado)"

    def __init__(self, nombre):
        self.nombre = nombre

    def version(self, v):
        return _Version(self.nombre)


class Roboflow:
    def __init__(self, api_key):
        pass

    def workspace(self, ws):
        return types.SimpleNamespace(project=lambda p: _Proyecto(next(k for k, f in CONFIG_FUENTES.items() if f.get("proyecto") == p)))


sys.modules["roboflow"] = types.SimpleNamespace(Roboflow=Roboflow)

ns = {"__name__": "__main__", "display": lambda df: print(df.to_string())}
sys.path.insert(0, str(tmp))


def correr(titulo_inicio, parches=()):
    src = next(s for s in fuentes if s.splitlines()[0].strip().startswith(f"# @title {titulo_inicio}"))
    src = src.replace('"/content', f'"{CONTENT.as_posix()}')
    for a, b in parches:
        assert a in src, f"parche no aplica: {a}"
        src = src.replace(a, b)
    try:
        exec(compile(sin_magias(src), titulo_inicio, "exec"), ns)
        check(True, f"ejecuta celda {titulo_inicio}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        check(False, f"ejecuta celda {titulo_inicio}", f"{type(e).__name__}: {e}")
        raise


import matplotlib.pyplot as plt  # noqa: E402
plt.show = lambda *a, **k: plt.close("all")

# celdas de modulos (%%writefile) -> archivos en tmp
for mod in ("v3_datos_lib", "v3_entrenamiento", "v3_inferencia"):
    src = next(s for s in fuentes if s.startswith(f"%%writefile {mod}.py"))
    (tmp / f"{mod}.py").write_text(src.split("\n", 1)[1], encoding="utf-8")

correr("2.", parches=[('MODELO_BASE = "yolov8l.pt"', 'MODELO_BASE = "yolov8n.pt"'),
                      ('"yolov8m.pt": 32,', '"yolov8n.pt": 8, "yolov8m.pt": 32,'),
                      ("IMGSZ = 640", "IMGSZ = 160")])
ns.update(MIN_EPOCAS=2, PACIENCIA=1, MEJORA_MINIMA=1.0, EPOCAS_FINALES=1, EPOCAS_MAX=6)   # meseta forzada: termina en la 3
for f in ns["FUENTES"]:
    CONFIG_FUENTES[f["nombre"]] = f

# huevos.zip sintetico en la ruta de Drive (su test = benchmark)
hv = tmp / "hv_src" / "huevos"
export_sintetico(hv, "huevos", {"Roto": "Roto", "Sano": "Sano"}, 60, 9100)
zdest = CONTENT / "drive" / "MyDrive" / "eggdetection" / "huevos.zip"
zdest.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zdest, "w") as z:
    for p in hv.rglob("*"):
        z.write(p, p.relative_to(hv.parent))

correr("3.")
correr("4.")
au = ns["auditoria"]
check(au["aau"]["convencion_usada"] == "region", "auditoria: 'aau' detectada como convencion REGION", au["aau"]["ioa_roto_en_sano"])
check(au["360a"]["convencion_usada"] == "huevo", "auditoria: '360a' detectada como convencion HUEVO")
check(au["ducks"]["descartes"].get("imagenes_excluidas", 0) > 0, "auditoria: clases EXCLUIR sacan imagenes (ducks)")
correr("5.")
check(ns["PROPORCIONES"]["train"] == 0.70, "con < 3.000 originales elige 70/15/15 (la regla de tamano)")
check("benchmark" in ns["final"] and all(it["fuente"] == "huevos" for it in ns["final"]["benchmark"]),
      "benchmark = test de 'huevos', aparte")
correr("6.")
check(ns["COLOR"] is not None and ns["COLOR"]["exactitud_prueba"] > 0.9, "color calibrado con las fuentes 360",
      f"{100 * ns['COLOR']['exactitud_prueba']:.0f}%")
correr("7.")
check(ns["NEG"]["imagenes"] > 0, "negativos duros generados", ns["NEG"]["imagenes"])
correr("8.")
correr("9.")
correr("10.")
correr("11.", parches=[("workers=2", "workers=0")])   # en Windows los workers relanzan el script; en Colab (Linux) no
import csv  # noqa: E402
filas = list(csv.DictReader(open(ns["RUN"] / "results.csv", encoding="utf-8")))
check(len(filas) == 3, "entreno real y el cierre ordenado termino en la epoca 3 (tope 6)", f"{len(filas)} epocas")
correr("12.")
correr("13.", parches=[("plots=False, verbose=False)", "plots=False, verbose=False, workers=0)")])
m = ns["METRICAS"]
check({"tau_roto", "tau_sano", "test", "benchmark", "test_map", "benchmark_map"} <= set(m), "metricas de test y benchmark")
correr("14.")
check(len(ns["ROBUSTEZ"]) == 10, "5 pruebas de robustez x (test, benchmark)")
correr("15.", parches=[("[1, 6, 8400]", "[1, 6, 525]")])        # 160 px -> 525 anclas (a 640 son 8400)
correr("16.")
paq = list((CONTENT / "drive" / "MyDrive" / "eggdetection").glob("v3_*.zip"))
check(len(paq) == 1, "paquete zip guardado en Drive", paq[0].name if paq else "")
if paq:
    with zipfile.ZipFile(paq[0]) as z:
        nombres = z.namelist()
        met = json.loads(z.read("metricas_v3.json"))
    check({"weights/best.onnx", "weights/best.pt", "metricas_v3.json", "manifiesto_v3.csv"} <= set(nombres),
          "el zip trae onnx, pt, metricas y manifiesto")
    check(set(met["backend_env"]) == {"TAU_ROTO", "TAU_SANO", "CONF_MIN_DETECCION"}, "umbrales listos para el .env del backend",
          met["backend_env"])
correr("17.")
b1 = io.BytesIO(); escena(1, 2)[0].save(b1, "JPEG")
_SUBIDAS.update({"mi_huevo.jpg": b1.getvalue()})
correr("18.")

os.chdir(AQUI)
shutil.rmtree(tmp, ignore_errors=True)
print("\n" + "=" * 76)
print("RESULTADO:", "TODO OK" if fallos == 0 else f"{fallos} FALLO(S)")
print("=" * 76)
print("Simulado (no existe fuera de Colab): descarga de Roboflow, Drive y el boton de subir fotos.")
print("Real: todo lo demas, incluido el entrenamiento con ultralytics 8.4.157.")
sys.exit(1 if fallos else 0)
