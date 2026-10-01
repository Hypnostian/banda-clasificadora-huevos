"""Valida el notebook ANTES de mandarlo a Colab.

Tres niveles:
  1. Estructura: JSON valido y magias de celda (%%) en la primera linea.
  2. Integridad: la biblioteca incrustada es byte a byte el archivo probado.
  3. Ensayo en seco: ejecuta DE VERDAD las celdas que no necesitan GPU
     (config, biblioteca, inventario, agrupamiento, split, vista previa)
     sobre un dataset sintetico con fugas plantadas. Atrapa errores de
     enlace entre celdas (nombres de variables, rutas) que ast.parse no ve.

    python colab/validar_notebook.py
"""
import ast
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                  # sin ventanas: plt.show() no bloquea

from PIL import Image, ImageDraw

AQUI = Path(__file__).parent
NB = AQUI / "entrenar_cdd_eggs_yolov8cls.ipynb"
fallos = 0


def check(cond, msg, extra=""):
    global fallos
    print(f"  {'[OK]  ' if cond else '[FALLO]'} {msg}{(' -> ' + str(extra)) if extra != '' else ''}")
    if not cond:
        fallos += 1


def sin_magias(src: str) -> str:
    out = []
    for ln in src.splitlines():
        s = ln.lstrip()
        if s.startswith("%%"):
            continue
        out.append(ln[: len(ln) - len(s)] + "pass" if s.startswith(("%", "!")) else ln)
    return "\n".join(out)


nb = json.loads(NB.read_text(encoding="utf-8"))
codigo = [c for c in nb["cells"] if c["cell_type"] == "code"]
fuentes = ["".join(c["source"]) for c in codigo]

print("=" * 72); print("1. ESTRUCTURA"); print("=" * 72)
check(nb["nbformat"] == 4, "nbformat 4")
for i, src in enumerate(fuentes):
    lineas = src.splitlines()
    mal = [k for k, ln in enumerate(lineas) if ln.lstrip().startswith("%%") and k != 0]
    check(not mal, f"celda de codigo {i}: magias de celda solo en la linea 1", f"lineas {mal}" if mal else "")
    try:
        ast.parse(sin_magias(src))
    except SyntaxError as e:
        check(False, f"celda de codigo {i} compila", e)

print("\n" + "=" * 72); print("2. INTEGRIDAD DE LA BIBLIOTECA"); print("=" * 72)
celda_lib = next(s for s in fuentes if s.startswith("%%writefile cdd_split_lib.py"))
incrustada = celda_lib.split("\n", 1)[1]
original = (AQUI / "cdd_split_lib.py").read_text(encoding="utf-8")
# Las celdas de Jupyter no guardan el salto de linea final; es la unica
# diferencia admitida (no cambia el comportamiento de Python).
check(incrustada.rstrip("\n") == original.rstrip("\n"),
      "la biblioteca del notebook es identica a la probada (salvo el salto final)",
      f"{len(original.rstrip())} bytes")

print("\n" + "=" * 72); print("3. ENSAYO EN SECO DE LAS CELDAS SIN GPU"); print("=" * 72)
tmp = Path(tempfile.mkdtemp(prefix="nb_dry_"))
ns: dict = {"__name__": "__main__"}

# google.colab falso: drive.mount no hace nada, files.upload devuelve lo que pongamos.
import types  # noqa: E402

_colab = types.ModuleType("google.colab")
_colab.drive = types.SimpleNamespace(mount=lambda *a, **k: print("  (drive.mount simulado)"))
_colab.files = types.SimpleNamespace(upload=lambda: dict(_SUBIDAS), download=lambda *a, **k: None)
_colab.userdata = types.SimpleNamespace(get=lambda k: "clave-falsa")
_google = types.ModuleType("google")
_google.colab = _colab
sys.modules["google"], sys.modules["google.colab"] = _google, _colab
_SUBIDAS: dict = {}


def celda(titulo_inicio: str) -> str:
    return next(s for s in fuentes if titulo_inicio in s.splitlines()[0])


def correr(nombre: str, src: str) -> None:
    # Todas las rutas /content (Colab) van a una carpeta temporal: el ensayo no
    # debe crear nada en el disco del usuario.
    src = src.replace('"/content', f'"{tmp.as_posix()}/content')
    try:
        exec(compile(sin_magias(src), nombre, "exec"), ns)
        check(True, f"ejecuta: {nombre}")
    except Exception as e:  # noqa: BLE001
        check(False, f"ejecuta: {nombre}", f"{type(e).__name__}: {e}")
        raise


# config
correr("2. Configuracion", celda("2. Configuraci"))
check(str(ns["CARPETA_RUNS"]).startswith(str(tmp)), "las rutas de Colab quedan en la carpeta temporal")
check(ns["CARPETA_RUNS"].is_dir(), "crea la carpeta de runs (en Drive en Colab)")
ns["EPOCHS"] = 1

# biblioteca: lo que haria %%writefile
(tmp / "cdd_split_lib.py").write_text(incrustada, encoding="utf-8")
sys.path.insert(0, str(tmp))

# en vez de descargar de Roboflow: export sintetico con la misma estructura y fugas plantadas
rng = random.Random(0)


def img(semilla):
    r = random.Random(semilla)
    im = Image.new("RGB", (224, 224), tuple(r.randint(0, 255) for _ in range(3)))
    d = ImageDraw.Draw(im)
    for _ in range(7):
        x, y = r.randint(-40, 200), r.randint(-40, 200)
        d.ellipse([x, y, x + r.randint(30, 140), y + r.randint(30, 140)], fill=tuple(r.randint(0, 255) for _ in range(3)))
    return im


def guardar(split, clase, base, im):
    p = ns["ORIGEN"] / split / clase
    p.mkdir(parents=True, exist_ok=True)
    im.save(p / f"{base}_jpg.rf.{rng.getrandbits(64):016x}.jpg", "JPEG", quality=90)


clases = ["good"] * 120 + ["broken"] * 60 + ["dirty"] * 24
rng.shuffle(clases)
for i, c in enumerate(clases):
    split = "train" if i < 163 else ("valid" if i < 184 else "test")
    guardar(split, c, f"img_{i:04d}", img(i))
    if split == "train":
        guardar(split, c, f"img_{i:04d}", img(i).transpose(Image.FLIP_LEFT_RIGHT))
for k, (s, f) in enumerate([("train", 3), ("test", 7), ("valid", 28), ("train", 55)]):
    guardar(s, "broken", f"20251002_174109_{f:03d}", img(9000 + k))
(ns["ORIGEN"] / "README.dataset.txt").write_text("export de prueba")      # Roboflow deja archivos sueltos

correr("5. Inventario y mapeo", celda("5. Inventario"))
check(len(ns["items"]) > 0, "hay items mapeados", len(ns["items"]))
check({it["clase"] for it in ns["items"]} == {"Roto", "Sano", "Sucio"}, "mapeo a clases del proyecto")

correr("6. Fuga en el split publicado", celda("6. "))
check(ns["fuga"]["grupos_cruzados"] >= 1, "detecta la fuga plantada del video", ns["fuga"])

correr("7. Split nuevo + verificacion", celda("7. Split"))
check(ns["sin_fuga"] is True, "split nuevo sin fuga")
check((ns["LIMPIO"] / "manifiesto_split.csv").exists(), "escribe el manifiesto")
for s in ("train", "val", "test"):
    check(all((ns["LIMPIO"] / s / c).is_dir() for c in ("Roto", "Sano", "Sucio")),
          f"{s}/ tiene las 3 carpetas de clase (lo que exige ultralytics)")

import matplotlib.pyplot as plt  # noqa: E402
plt.show = lambda *a, **k: None
correr("8. Vista previa", celda("8. Vista previa"))

# el modo binario tambien debe funcionar
for modo, esperadas in [("binario", {"Roto", "Sano"}), ("binario_sin_sucios", {"Roto", "Sano"})]:
    ns["MODO_CLASES"] = modo
    correr(f"5. Inventario (modo {modo})", celda("5. Inventario"))
    check({it["clase"] for it in ns["items"]} == esperadas, f"modo {modo}: clases {esperadas}")
    if modo == "binario_sin_sucios":
        check(all(it["clase_orig"] != "dirty" for it in ns["items"]), "descarta los sucios")

# =============================================================================
print("\n" + "=" * 72); print("4. ENSAYO CON MODELO SIMULADO (celdas 10, 11, 12 y 15)"); print("=" * 72)
# Las celdas posteriores al entrenamiento son las mas largas, y un error ahi
# aparece despues de 40 min de GPU. Se ejecutan con un modelo falso que imita
# la interfaz de ultralytics (predict -> r.probs.data.float().cpu().numpy()).
import hashlib  # noqa: E402

import numpy as np  # noqa: E402

ns["MODO_CLASES"] = "3clases"
correr("5. Inventario (volver a 3clases)", celda("5. Inventario"))
correr("6. (reagrupar)", celda("6. "))
correr("7. (resplit)", celda("7. Split"))


class _T:
    def __init__(self, a): self.a = a
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.a


class _Res:
    def __init__(self, a):
        self.probs = type("Probs", (), {"data": _T(a)})()


class ModeloFalso:
    names = {0: "Roto", 1: "Sano", 2: "Sucio"}

    def predict(self, fuente, **kw):
        assert "batch" not in kw or kw["batch"] <= 64
        assert len(fuente) <= 64, "predecir() debe partir en lotes de <=64"
        for f in fuente:
            if isinstance(f, str):
                real = list(self.names.values()).index(Path(f).parent.name)
                semilla = int(hashlib.md5(f.encode()).hexdigest()[:8], 16)
            else:
                real = None
                semilla = int(np.asarray(f).mean() * 1000)
            r = np.random.default_rng(semilla)
            p = r.dirichlet([1, 1, 1]).astype(np.float32)
            if real is not None and r.random() < 0.85:          # mayormente acierta
                p = p * 0.2; p[real] += 0.8
            yield _Res(p / p.sum())

    def export(self, **kw):
        raise RuntimeError("no se ejecuta en el ensayo")


RUN = tmp / "runs" / "falso"
RUN.mkdir(parents=True)
# results.csv con las columnas reales de ultralytics 8.4.157 para clasificacion
with open(RUN / "results.csv", "w") as fh:
    fh.write("epoch,time,train/loss,metrics/accuracy_top1,metrics/accuracy_top5,val/loss,lr/pg0,lr/pg1,lr/pg2\n")
    for e in range(1, 11):
        fh.write(f"{e},{e*30},{1.2/e:.4f},{0.6+0.03*e:.4f},1.0,{1.1/e+0.05:.4f},0.0005,0.0005,0.0005\n")
ns["RUN"] = RUN
ns["YOLO"] = lambda *a, **k: ModeloFalso()
ns["modelo"] = ModeloFalso()

correr("10. Curvas", celda("10. Curvas"))
correr("11. Evaluacion en TEST", celda("11. Evaluaci"))
tr, td = ns.get("TAU_ROTO"), ns.get("TAU_DUDA")
check(tr is not None and 0.05 <= tr <= 0.95, "elige TAU_ROTO en val", tr)
check(td is not None and 0.0 < td <= tr, "elige TAU_DUDA en val, por debajo de TAU_ROTO", td)
check((RUN / "confusion_test.png").exists(), "guarda la matriz de confusion")
check(set(ns["METRICAS"]) >= {"top1_test", "fuga_pct", "fuga_a_empaque_pct", "tau_roto", "tau_duda",
                              "buenos_a_revision_pct"}, "arma METRICAS")

# --- semantica de los 4 veredictos, con vectores fabricados [Roto, Sano, Sucio] ---
V = ns["veredicto"]
check(V(np.array([0.99, 0.005, 0.005])) == "DANADO", "p(Roto) alto -> DANADO")
check(V(np.array([0.0, 0.95, 0.05])) == "BUENO", "limpio seguro -> BUENO")
check(V(np.array([0.0, 0.10, 0.90])) == "BUENO_SUCIO", "sucio seguro -> BUENO_SUCIO")
if td < tr:
    medio = (td + tr) / 2
    check(V(np.array([medio, 1 - medio, 0.0])) == "REVISION", "p(Roto) en la banda -> REVISION",
          f"p(Roto)={medio:.3f} en [{td:.2f}, {tr:.2f})")
# jerarquia: un "sucio" con sospecha de rotura NO se acepta como bueno
check(V(np.array([tr, 0.0, 1 - tr])) == "DANADO", "sucio pero con p(Roto)>=TAU_ROTO -> DANADO, no BUENO_SUCIO")
# en modo binario nunca aparece BUENO_SUCIO
idx_3 = dict(ns["IDX"])
ns["IDX"] = {"Roto": 0, "Sano": 1}
check(V(np.array([0.0, 0.2, 0.8])) == "BUENO", "modo binario: nunca BUENO_SUCIO")
ns["IDX"] = idx_3

correr("12. Robustez", celda("12. Robustez"))
check(set(ns["ROBUSTEZ"]["blur"]) == {0, 1, 2, 3, 4, 6}, "robustez: 6 niveles de desenfoque")
check(2.4 in ns["ROBUSTEZ"]["brillo"], "robustez: incluye brillo 2.4x")
check(set(ns["ROBUSTEZ"]["raya_veredictos"]) == {"BUENO", "BUENO_SUCIO", "REVISION", "DANADO"},
      "raya: reporta los 4 veredictos")
correr("15. Resumen", celda("15. Resumen"))

# --- celda 16: subir fotos propias (con una foto girada por EXIF, como las del celular) ---
import io  # noqa: E402

def _jpg(im, exif_orient=None):
    b = io.BytesIO()
    if exif_orient:
        ex = Image.Exif(); ex[0x0112] = exif_orient
        im.save(b, "JPEG", exif=ex.tobytes())
    else:
        im.save(b, "JPEG")
    return b.getvalue()

_SUBIDAS.update({"huevo_1.jpg": _jpg(img(1).resize((480, 640))),
                 "huevo_girado.jpg": _jpg(img(2).resize((640, 480)), exif_orient=6),
                 "huevo_3.png": _jpg(img(3))})
correr("16. Tus fotos (3 fotos, una girada por EXIF)", celda("16. Prueba"))
_SUBIDAS.clear()
correr("16. Tus fotos (sin subir nada)", celda("16. Prueba"))

# =============================================================================
print("\n" + "=" * 72); print("5. REGLA DE PARADA (callback de la celda 9) con historiales simulados"); print("=" * 72)
src9 = celda("9. Entrenar")
arbol = ast.parse(sin_magias(src9))
fdef = next(n for n in arbol.body if isinstance(n, ast.FunctionDef) and n.name == "parada_con_minimo")
ns5 = {"Path": Path, "csv": __import__("csv"), "MIN_EPOCAS": 150, "PACIENCIA": 30,
       "MEJORA_MINIMA": 0.005, "print": lambda *a, **k: None}
exec(compile(ast.Module(body=[fdef], type_ignores=[]), "celda9", "exec"), ns5)
parada = ns5["parada_con_minimo"]
EPOCAS_MAX = 200


class TrainerFalso:
    def __init__(self, ruta):
        self.csv, self.stop = ruta, False


def simular(curva, desde=1, previo=None):
    """Escribe results.csv epoca a epoca como ultralytics y llama al callback."""
    ruta = tmp / f"results_{random.random():.6f}.csv"
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("epoch,time,train/loss,metrics/accuracy_top1,metrics/accuracy_top5,val/loss\n")
        for e in range(1, desde):
            fh.write(f"{e},{e*30},0.1,{curva(e):.6g},1,0.3\n")
    t = TrainerFalso(ruta)
    for e in range(desde, EPOCAS_MAX + 1):
        with open(ruta, "a", encoding="utf-8") as fh:
            fh.write(f"{e},{e*30},0.1,{curva(e):.6g},1,0.3\n")
        parada(t)
        if t.stop:
            return e
    return None   # llego al tope sin que el callback lo detuviera


tu_run = lambda e: 0.868 if e == 12 else (0.80 + 0.005 * min(e, 11) if e < 12 else 0.75)   # noqa: E731
check(simular(tu_run) == 150, "tu run (pico en la epoca 12): no para antes de la 150, y para justo en 150")

mejora_hasta_160 = lambda e: 0.70 + 0.002 * min(e, 160)                                    # noqa: E731
e_b = simular(mejora_hasta_160)
check(e_b is not None and 185 <= e_b <= 190, "si sigue mejorando hasta la 160, para ~30 despues", e_b)

check(simular(lambda e: 0.60 + 0.0015 * e) is None, "si mejora hasta el final, llega al tope (200) sin cortar")

ruido = lambda e: 0.85 + 0.004 * ((-1) ** e) if e > 40 else 0.60 + 0.00625 * e             # noqa: E731
check(simular(ruido) == 150, "oscilaciones de +-0.004 (menores a la mejora minima) NO reinician la cuenta")

e_ref = simular(mejora_hasta_160)
check(simular(mejora_hasta_160, desde=91) == e_ref, "al reanudar en la epoca 91 para en la misma epoca", e_ref)

shutil.rmtree(tmp, ignore_errors=True)
print("\n" + "=" * 72)
print("RESULTADO:", "TODO OK" if fallos == 0 else f"{fallos} FALLO(S)")
print("=" * 72)
print("Sin ejecutar aqui (necesitan GPU/Colab/red): instalacion, descarga, entrenamiento,")
print("export ONNX y guardado. Su sintaxis quedo verificada en el punto 1.")
sys.exit(1 if fallos else 0)
