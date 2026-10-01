"""Detector (modelo 1) vs clasificador (modelo 2) sobre LOS MISMOS huevos.

Terreno comun: test del dataset 'huevos' (145 imgs, 183 huevos con caja y clase).
Sesgo a declarar: ese test es del MISMO dominio que entreno el detector (fuera de
su train, pero mismo tipo de fotos). El clasificador nunca vio este dominio.
"""
import sys, time, random
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import onnxruntime as ort

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "backend"))
from app.config import CONFIG                      # noqa: E402
from app.detector import DetectorHuevosONNX        # noqa: E402
from app.routing import MotorDecision              # noqa: E402

det = DetectorHuevosONNX(CONFIG.modelo_path, 640, 0)
motor = MotorDecision(CONFIG)
MODELO2 = RAIZ / "colab" / "entrenamientomodel2" / "weights" / "best.onnx"
cls = ort.InferenceSession(str(MODELO2),
                           providers=["CPUExecutionProvider"])
TAU_ROTO_C, TAU_DUDA_C = 0.37, 0.21                 # calibrados en val (metricas.json)
TEST = RAIZ / "huevos" / "test"
RUTA_DET = {"DESCARTE": "DANADO", "ACEPTADO": "BUENO", "REVISION": "REVISION"}


def preproc_cls(im):
    """Replica exacta de classify_transforms (ultralytics 8.4.157)."""
    w, h = im.size
    nw, nh = (224, int(224 * h / w)) if w <= h else (int(224 * w / h), 224)
    im = im.resize((nw, nh), Image.BILINEAR)
    l, t = int(round((nw - 224) / 2)), int(round((nh - 224) / 2))
    return (np.asarray(im.crop((l, t, l + 224, t + 224)), np.float32) / 255.0).transpose(2, 0, 1)[None]


def p_cls(im):
    return cls.run(None, {"images": preproc_cls(im)})[0][0]


def v_cls(p):
    if p[0] >= TAU_ROTO_C:
        return "DANADO"
    return "REVISION" if p[0] >= TAU_DUDA_C else "BUENO"


def recorte(im, b, margen=0.15):
    x1, y1, x2, y2 = b
    cx, cy, lado = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * (1 + 2 * margen)
    return im.crop(tuple(int(round(v)) for v in (cx - lado / 2, cy - lado / 2, cx + lado / 2, cy + lado / 2)))


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def huevos_de(ruta_img):
    im = Image.open(ruta_img).convert("RGB")
    W, H = im.size
    gts = []
    for ln in (TEST / "labels" / (ruta_img.stem + ".txt")).read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
            gts.append(("Roto" if c == 0 else "Sano", [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]))
    return im, gts


def evaluar(im, gts, transform=None, solo=None):
    """Para cada huevo real: veredicto de cada estrategia."""
    im_t = transform(im, gts) if transform else im
    buf = __import__("io").BytesIO(); im_t.save(buf, "JPEG", quality=92)
    cajas, _, _ = det.detectar(buf.getvalue(), CONFIG.conf_min_deteccion, CONFIG.iou_nms)
    usadas, out = set(), []
    for clase, gt in gts:
        if solo and clase != solo:
            continue
        mejor, mi = None, 0.5
        for k, c in enumerate(cajas):
            v = iou(gt, [c.x1, c.y1, c.x2, c.y2])
            if v >= mi and k not in usadas:
                mejor, mi = k, v
        r = {"clase": clase}
        r["cls_gt"] = v_cls(p_cls(recorte(im_t, gt)))                   # techo: recorte perfecto
        if mejor is None:
            r["det"] = r["dos_etapas"] = r["cualquiera"] = r["coinciden"] = "NO_DETECTADO"
        else:
            usadas.add(mejor)
            c = cajas[mejor]
            vd = RUTA_DET[motor.decidir(c.p_roto, c.p_sano, c.nitidez).ruta.value]
            vc = v_cls(p_cls(recorte(im_t, [c.x1, c.y1, c.x2, c.y2])))  # recorte del detector
            r["det"], r["dos_etapas"] = vd, vc
            r["cualquiera"] = "DANADO" if "DANADO" in (vd, vc) else ("BUENO" if vd == vc == "BUENO" else "REVISION")
            r["coinciden"] = vd if vd == vc else "REVISION"
        out.append(r)
    return out


ESTRATEGIAS = [("det", "1. Detector solo (en produccion hoy)"),
               ("cls_gt", "2. Clasificador, recorte perfecto (techo)"),
               ("dos_etapas", "3. Dos etapas: detector recorta, clasificador juzga"),
               ("cualquiera", "4. Combinado: DANADO si cualquiera sospecha"),
               ("coinciden", "5. Combinado: si no coinciden -> REVISION")]


def tabla(resultados, titulo):
    print("=" * 96); print(titulo); print("=" * 96)
    for clase in ("Roto", "Sano"):
        rs = [r for r in resultados if r["clase"] == clase]
        if not rs:
            continue
        print(f"\n  huevos {clase.upper()} reales: {len(rs)}")
        print(f"  {'estrategia':<52}{'DANADO':>9}{'REVISION':>10}{'BUENO':>8}{'no det.':>9}")
        for k, nombre in ESTRATEGIAS:
            c = Counter(r[k] for r in rs)
            print(f"  {nombre:<52}{c['DANADO']:>9}{c['REVISION']:>10}{c['BUENO']:>8}{c['NO_DETECTADO']:>9}")


imgs = sorted((TEST / "images").glob("*.jpg"))
base = []
for f in imgs:
    im, gts = huevos_de(f)
    base += evaluar(im, gts)
tabla(base, "A) TEST LIMPIO (145 imagenes, mismo dominio que el detector)")

# resumen critico
print("\n  RESUMEN (lo que importa en la banda):")
print(f"  {'estrategia':<52}{'roto->EMPAQUE':>14}{'roto no desc.':>14}{'sano botado':>12}{'sano a rev.':>12}")
for k, nombre in ESTRATEGIAS:
    ro = [r for r in base if r["clase"] == "Roto"]; sa = [r for r in base if r["clase"] == "Sano"]
    emp = sum(r[k] == "BUENO" for r in ro); nd = sum(r[k] != "DANADO" for r in ro)
    bot = sum(r[k] == "DANADO" for r in sa); rev = sum(r[k] == "REVISION" for r in sa)
    print(f"  {nombre:<52}{emp:>7}/{len(ro)} {100*emp/len(ro):4.1f}%{nd:>6}/{len(ro)}"
          f"{bot:>7}/{len(sa)} {100*bot/len(sa):3.0f}%{rev:>5}/{len(sa)} {100*rev/len(sa):3.0f}%")

# ------------------------------------------------------------------ robustez
def rayar(im, gts):
    im = im.copy(); d = ImageDraw.Draw(im); rng = random.Random(3)
    for clase, (x1, y1, x2, y2) in gts:
        if clase != "Sano":
            continue
        cx, alto = (x1 + x2) / 2, y2 - y1
        pts = [(cx + (x2 - x1) * 0.12 * np.sin(t * 3.1) + rng.uniform(-3, 3), y1 + alto * (0.12 + 0.76 * t))
               for t in (i / 13 for i in range(14))]
        d.line(pts, fill=(35, 25, 20), width=3, joint="curve")
    return im


print("\n")
for titulo, tf, solo, clave in [
        ("B) RAYA DE LAPICERO sobre los huevos SANOS (la falla que encontraste en campo)", rayar, "Sano", "DANADO"),
        ("C) DESENFOQUE radio 3 sobre los huevos ROTOS", lambda im, g: im.filter(ImageFilter.GaussianBlur(3)), "Roto", None),
        ("D) SOBREEXPOSICION 2.4x sobre los huevos ROTOS", lambda im, g: ImageEnhance.Brightness(im).enhance(2.4), "Roto", None)]:
    rs = []
    for f in imgs:
        im, gts = huevos_de(f)
        if any(c == solo for c, _ in gts):
            rs += evaluar(im, gts, tf, solo)
    print("=" * 96); print(titulo); print("=" * 96)
    print(f"  {'estrategia':<52}{'DANADO':>9}{'REVISION':>10}{'BUENO':>8}{'no det.':>9}")
    for k, nombre in ESTRATEGIAS:
        c = Counter(r[k] for r in rs)
        print(f"  {nombre:<52}{c['DANADO']:>9}{c['REVISION']:>10}{c['BUENO']:>8}{c['NO_DETECTADO']:>9}   (n={len(rs)})")
    print()

# ------------------------------------------------------------------ latencia
x = preproc_cls(Image.open(imgs[0]).convert("RGB"))
for hilos in (2, 0):
    so = ort.SessionOptions(); so.intra_op_num_threads = hilos
    s = ort.InferenceSession(str(MODELO2), so,
                             providers=["CPUExecutionProvider"])
    for _ in range(3):
        s.run(None, {"images": x})
    t = time.perf_counter()
    for _ in range(30):
        s.run(None, {"images": x})
    print(f"latencia clasificador ({'2 hilos' if hilos else 'todos los hilos'}): {(time.perf_counter()-t)/30*1000:.1f} ms por huevo")
