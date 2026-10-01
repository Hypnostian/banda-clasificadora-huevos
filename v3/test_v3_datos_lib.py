"""Prueba v3_datos_lib sobre fuentes SINTETICAS que reproducen los problemas reales.

    python v3/test_v3_datos_lib.py
"""
import random
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import v3_datos_lib as L  # noqa: E402

rng = random.Random(0)
fallos = 0


def check(cond, msg, extra=""):
    global fallos
    print(f"  {'[OK]  ' if cond else '[FALLO]'} {msg}{(' -> ' + str(extra)) if extra != '' else ''}")
    fallos += 0 if cond else 1


MARRON, BLANCO = (190, 125, 80), (238, 232, 222)


def escena(semilla, huevos, fondo=None):
    """huevos: lista de (cx, cy, w, h, color, roto). Devuelve imagen 480x640 y cajas px."""
    r = random.Random(semilla)
    im = Image.new("RGB", (480, 640), fondo or tuple(r.randint(40, 200) for _ in range(3)))
    d = ImageDraw.Draw(im)
    for _ in range(12):                                # textura de fondo unica por imagen
        x, y = r.randint(0, 480), r.randint(0, 640)
        d.rectangle([x, y, x + r.randint(10, 90), y + r.randint(10, 90)], fill=tuple(r.randint(0, 255) for _ in range(3)))
    grietas = []
    for cx, cy, w, h, col, roto in huevos:
        tono = tuple(max(0, min(255, c + r.randint(-12, 12))) for c in col)
        d.ellipse([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill=tono)
        if roto:
            pts = [(cx + r.uniform(-w * .12, w * .12), cy - h * .3 + k * h * .12) for k in range(6)]
            d.line(pts, fill=(45, 30, 25), width=3)
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            grietas.append((min(xs) - 3, min(ys) - 3, max(xs) + 3, max(ys) + 3))
    return im, grietas


def rf(nombre):
    return f"{nombre}_jpg.rf.{rng.getrandbits(48):012x}"


def guardar(raiz, split, nombre, im, lineas, nombres):
    (raiz / split / "images").mkdir(parents=True, exist_ok=True)
    (raiz / split / "labels").mkdir(parents=True, exist_ok=True)
    im.save(raiz / split / "images" / f"{nombre}.jpg", quality=92)
    (raiz / split / "labels" / f"{nombre}.txt").write_text("".join(ln + "\n" for ln in lineas))
    (raiz / "data.yaml").write_text(f"nc: {len(nombres)}\nnames: {nombres}\n")


def yolo(c, x1, y1, x2, y2, W=480, H=640):
    return f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def huevos_aleatorios(r, n):
    out = []
    for k in range(n):
        cx, cy = 90 + (k % 3) * 150 + r.randint(-15, 15), 140 + (k // 3) * 220 + r.randint(-15, 15)
        out.append((cx, cy, r.randint(80, 110), r.randint(110, 140), r.choice([MARRON, BLANCO]), r.random() < 0.4))
    return out


tmp = Path(tempfile.mkdtemp(prefix="v3_test_"))
print("=" * 74); print("PRUEBAS DE v3_datos_lib"); print("=" * 74)

# ---- fuente A: convencion "huevo", clases con color (tipo 360digitmg), train x3 aumentado
A = tmp / "A"
nombresA = ["Cracked Egg Brown", "Cracked Egg White", "Normal Egg Brown", "Normal Egg White"]
imgs_A = {}
for i in range(120):
    r = random.Random(1000 + i)
    hs = huevos_aleatorios(r, r.randint(1, 3))
    im, _ = escena(1000 + i, hs)
    imgs_A[i] = im
    lineas = []
    for cx, cy, w, h, col, roto in hs:
        c = (0 if col == MARRON else 1) if roto else (2 if col == MARRON else 3)
        lineas.append(yolo(c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    split = "train" if i < 96 else ("valid" if i < 108 else "test")
    copias = [im, im.transpose(Image.FLIP_LEFT_RIGHT), im] if split == "train" else [im]
    for k, cim in enumerate(copias):
        ls = lineas
        if k == 1:                                      # espejo: x -> 1-x
            ls = [f"{l.split()[0]} {1 - float(l.split()[1]):.6f} {' '.join(l.split()[2:])}" for l in lineas]
        guardar(A, split, rf(f"a_{i:03d}"), cim, ls, nombresA)

# ---- fuente B: convencion "region" (cajas de grieta DENTRO de cajas de huevo)
B = tmp / "B"
for i in range(80):
    r = random.Random(2000 + i)
    hs = huevos_aleatorios(r, r.randint(1, 3))
    im, grietas = escena(2000 + i, hs)
    lineas = [yolo(0, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for cx, cy, w, h, _, _ in hs]
    lineas += [yolo(1, *g) for g in grietas]
    guardar(B, "train" if i < 60 else ("valid" if i < 70 else "test"), rf(f"b_{i:03d}"), im, lineas, ["egg", "crack"])

# ---- fuente C: copias de 20 imagenes de A (algunas rotadas 90 y espejadas) -> duplicados entre fuentes
C = tmp / "C"
for k, i in enumerate(range(0, 40, 2)):
    im = imgs_A[i]
    if k % 3 == 1:
        im = im.transpose(Image.ROTATE_90)
    elif k % 3 == 2:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    guardar(C, "train", rf(f"copia_{k}"), im, ["0 0.5 0.5 0.3 0.3"], ["crack", "normal"])

# ---- fuente "huevos": su TEST es el benchmark; una de esas fotos esta copiada en la fuente D
HV = tmp / "huevos"
imgs_hv = {}
for i in range(50):
    r = random.Random(3000 + i)
    hs = huevos_aleatorios(r, 1)
    im, _ = escena(3000 + i, hs)
    imgs_hv[i] = im
    lin = [yolo(0 if hs[0][5] else 1, hs[0][0] - hs[0][2] / 2, hs[0][1] - hs[0][3] / 2,
                hs[0][0] + hs[0][2] / 2, hs[0][1] + hs[0][3] / 2)]
    guardar(HV, "train" if i < 30 else ("valid" if i < 40 else "test"), rf(f"damaged_{i}"), im, lin, ["Roto", "Sano"])

# ---- fuente D: segmentacion (poligonos) + clase de pato que debe EXCLUIR la imagen + una foto del benchmark
D = tmp / "D"
for i in range(40):
    r = random.Random(4000 + i)
    hs = huevos_aleatorios(r, 2)
    im, _ = escena(4000 + i, hs)
    lineas = []
    for j, (cx, cy, w, h, _, roto) in enumerate(hs):
        c = 2 if (i % 10 == 0 and j == 0) else (1 if roto else 0)       # 2 = duck-clean
        poly = " ".join(f"{(cx + w / 2 * np.cos(t)) / 480:.5f} {(cy + h / 2 * np.sin(t)) / 640:.5f}"
                        for t in np.linspace(0, 2 * np.pi, 12, endpoint=False))
        lineas.append(f"{c} {poly}")
    guardar(D, "train", rf(f"d_{i}"), im, lineas, ["egg-clean", "egg-cracked", "duck-clean"])
guardar(D, "train", rf("robada_del_benchmark"), imgs_hv[45], ["0 0.5 0.5 0.2 0.2"], ["egg-clean", "egg-cracked", "duck-clean"])
# otra "toma" del mismo huevo del benchmark: un poco mas clara y desplazada (no identica)
from PIL import ImageEnhance  # noqa: E402
casi = ImageEnhance.Brightness(imgs_hv[46]).enhance(1.06).crop((6, 8, 480, 640)).resize((480, 640))
guardar(D, "train", rf("casi_copia_del_benchmark"), casi, ["0 0.5 0.5 0.2 0.2"], ["egg-clean", "egg-cracked", "duck-clean"])

# =========================================================================== pruebas
FUENTES = {
    "huevos": dict(ruta=HV, mapa={"Roto": "Roto", "Sano": "Sano"}),
    "A": dict(ruta=A, mapa={"Cracked Egg Brown": "Roto", "Cracked Egg White": "Roto",
                            "Normal Egg Brown": "Sano", "Normal Egg White": "Sano"},
              color={"Cracked Egg Brown": "marron", "Normal Egg Brown": "marron",
                     "Cracked Egg White": "blanco", "Normal Egg White": "blanco"}),
    "B": dict(ruta=B, mapa={"egg": "Sano", "crack": "Roto"}),
    "C": dict(ruta=C, mapa={"crack": "Roto", "normal": "Sano"}),
    "D": dict(ruta=D, mapa={"egg-clean": "Sano", "egg-cracked": "Roto", "duck-clean": L.EXCLUIR}),
}
todos, auditorias = [], {}
for nombre, f in FUENTES.items():
    its = L.inventariar(f["ruta"], nombre)
    au = L.auditar(its, f["mapa"])
    auditorias[nombre] = au
    L.mapear(its, f["mapa"], au["convencion"], f.get("color"))
    todos += [it for it in its if not it["excluir"]]

check(auditorias["B"]["convencion"] == "region", "detecta la convencion 'region' (grieta dentro del huevo)",
      f"IoA={auditorias['B']['ioa_roto_en_sano']:.2f}, area={auditorias['B']['area_roto_sobre_sano']:.2f}")
check(auditorias["A"]["convencion"] == "huevo", "detecta la convencion 'huevo' en la fuente A")
itsB = [it for it in todos if it["fuente"] == "B"]
check(all(c[0] in ("Roto", "Sano") and c[3] > 0.1 for it in itsB for c in it["cajas"]),
      "B convertida: solo cajas de HUEVO (las de grieta se vuelven 'Roto' en su huevo)")
check(sum(c[0] == "Roto" for it in itsB for c in it["cajas"]) > 0, "B conserva huevos rotos tras convertir")
itsD = L.inventariar(D, "D")
L.mapear(itsD, FUENTES["D"]["mapa"], "huevo")
check(sum(it["excluir"] for it in itsD) == 4, "clase de pato (EXCLUIR) saca la imagen entera", sum(it["excluir"] for it in itsD))
check(all(0 < c[3] < 1 for it in itsD for c in it["cajas"]), "poligonos de segmentacion -> cajas validas")
check(any(c[5] == "marron" for it in todos if it["fuente"] == "A" for c in it["cajas"]), "color real guardado desde la clase")

L.calcular_hashes(todos, log=lambda *_: None)
n0 = len(todos)
todos = L.deduplicar_entre_fuentes(todos, prioridad=list(FUENTES), log=print)
quitadas_C = sum(1 for it in todos if it["fuente"] == "C")
check(quitadas_C == 0, "las 20 copias de C (incluidas rotadas 90 y espejadas) se eliminan", f"quedan {quitadas_C}")
check(not any(it["base_corta"].startswith("robada") for it in todos),
      "copia EXACTA del benchmark en otra fuente: se elimina en la dedup (gana la del benchmark)")

info = L.agrupar(todos)
es_bench = lambda it: it["fuente"] == "huevos" and it["split_orig"] == "test"   # noqa: E731
asig = L.repartir(todos, {"train": 0.8, "val": 0.1, "test": 0.1}, benchmark=es_bench)
casi = [it for it in todos if it["base_corta"].startswith("casi_copia")]
check(all(asig[it["grupo"]] == "benchmark" for it in casi),
      "CASI-copia del benchmark: nunca queda en train/val/test",
      "eliminada por dedup" if not casi else f"agrupada -> {asig[casi[0]['grupo']]}")
real = L.revisar_reparto(todos, asig, {"train": 0.8, "val": 0.1, "test": 0.1})
check(abs(real["train"] - 0.8) < 0.06 and min(real["val"], real["test"]) > 0.07, "reparto cerca de 80/10/10",
      {k: f"{100 * v:.1f}%" for k, v in real.items()})

dest = tmp / "final"
final = L.materializar(todos, asig, dest)
L.escribir_data_yaml(dest, list(final))
check(L.verificar_sin_fuga(final, log=print), "verificacion independiente: sin fuga (incluye benchmark)")
check((dest / "data.yaml").read_text().count("names: ['Roto', 'Sano']") == 1, "data.yaml con el orden del backend")
lbl = next((dest / "train" / "labels").glob("*.txt")).read_text().split()
check(lbl and lbl[0] in ("0", "1"), "etiquetas escritas con ids 0=Roto 1=Sano")
bases_val = [it["base"] for it in final.get("val", [])]
check(len(bases_val) == len(set(bases_val)), "una imagen por original")

n_img = len(list((dest / "train" / "images").glob("*")))
neg = L.generar_negativos_duros(final["train"], dest, fraccion=0.5, log=print)
nuevos = list((dest / "train" / "images").glob("*__neg.jpg"))
check(len(nuevos) == neg["imagenes"] > 0 and len(list((dest / "train" / "images").glob("*"))) == n_img + len(nuevos),
      "negativos duros: imagenes nuevas solo en train", len(nuevos))
check(not any(next((dest / s / "images").glob("*__neg*"), None) for s in ("val", "test", "benchmark") if (dest / s).exists()),
      "val/test/benchmark sin negativos sinteticos")
ejemplo = nuevos[0]
orig = dest / "train" / "images" / (ejemplo.stem.replace("__neg", "") + ".jpg")
check((dest / "train" / "labels" / (ejemplo.stem + ".txt")).read_text()
      == (dest / "train" / "labels" / (orig.stem + ".txt")).read_text(), "la etiqueta del negativo es la del original (sigue Sano)")

# La marca se mide SIN recomprimir: solo recomprimir en JPEG ya cambia pixeles en los bordes del fondo.
fuera_tot, px_tot = 0, 0
for it in final["train"][:40]:
    sanos = [c for c in it["cajas"] if c[0] == "Sano"]
    if not sanos:
        continue
    im0 = Image.open(it["ruta_final"]).convert("RGB")
    W, H = im0.size
    for s, estilo in enumerate(("sello", "lapicero_color", "lapicero_oscuro", "suciedad")):
        c = sanos[s % len(sanos)]
        dif = np.abs(np.asarray(L.marcar_huevo(im0, c, random.Random(s), estilo), np.int16)
                     - np.asarray(im0, np.int16)).sum(axis=2) > 25
        x1, y1 = int((c[1] - c[3] / 2) * W), int((c[2] - c[4] / 2) * H)
        x2, y2 = int((c[1] + c[3] / 2) * W) + 1, int((c[2] + c[4] / 2) * H) + 1
        dentro = np.zeros_like(dif)
        dentro[max(0, y1):y2, max(0, x1):x2] = True
        fuera_tot += int((dif & ~dentro).sum())
        px_tot += int(dif.sum())
check(px_tot > 1000 and fuera_tot == 0, "las 4 clases de marca caen 100 % dentro de la caja del huevo sano",
      f"{px_tot} px marcados, {fuera_tot} fuera")

muestras = []
for it in final["train"]:
    im = Image.open(it["ruta_final"]).convert("RGB")
    for c in it["cajas"]:
        if c[5]:
            s, v = L.rasgos_color(im, c)
            muestras.append((s, v, c[5]))
cal = L.calibrar_color(muestras)
check(cal["exactitud_prueba"] > 0.9, "color de cascara calibrado y medido en mitad no vista",
      f"umbral S={cal['umbral_saturacion']:.2f}, exactitud prueba={100 * cal['exactitud_prueba']:.0f}% (n={cal['n_prueba']})")

it_n = [{"cajas_src": [("Cracked_Egg-Brown ", .5, .5, .2, .3), ("normal egg WHITE", .2, .2, .1, .1)]}]
L.mapear(it_n, {"Cracked Egg Brown": "Roto", "Normal Egg White": "Sano"}, "huevo",
         {"Cracked Egg Brown": "marron", "Normal Egg White": "blanco"})
check([c[0] for c in it_n[0]["cajas"]] == ["Roto", "Sano"] and not it_n[0]["excluir"] and it_n[0]["cajas"][0][5] == "marron",
      "nombres de clase con otra capitalizacion/separadores se mapean igual")

shutil.rmtree(tmp, ignore_errors=True)
print("=" * 74)
print("RESULTADO:", "TODO OK" if fallos == 0 else f"{fallos} FALLO(S)")
print("=" * 74)
sys.exit(1 if fallos else 0)
