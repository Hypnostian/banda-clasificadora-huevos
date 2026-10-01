"""Comparacion CRITICA: modelo 1 (YOLOv8m, solo `huevos`) contra v3 (YOLOv8l, 8 fuentes).

Reglas para que sea justa:
- Mismo codigo de PRODUCCION para los dos (app.detector: letterbox, sesion ONNX, NMS, nitidez).
- Mismas fotos: huevos/test (145 fotos, 183 huevos). Ninguno entreno con ellas: el modelo 1 las
  tuvo de test y la v3 de benchmark (su manifiesto lo confirma).
- Mismo emparejamiento y ruteo que la celda 13 del notebook (v3_inferencia), cada modelo con SUS
  umbrales: el modelo 1 con los de produccion (.env) y la v3 con los calibrados en su val.
- Tambien SIN umbrales (AUC, tabla por umbral), porque un umbral puede mover errores de lado.
- Pruebas pareadas (McNemar exacto) e intervalos (Wilson, bootstrap): con 43 sanos, una
  diferencia de 2 huevos no es una diferencia.

Es la cancha del modelo 1 (entreno solo con fotos de este dominio): si la v3 gana aqui, gana.

    <venv>/python v3/comparar_v1_v3.py
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

RAIZ = Path(__file__).resolve().parents[1]
AQUI = Path(__file__).resolve().parent
for p in (RAIZ / "backend", RAIZ / "backend" / "scripts", AQUI):
    sys.path.insert(0, str(p))
from app.config import CONFIG                                  # noqa: E402
from app.detector import DetectorHuevosONNX, nitidez_recorte   # noqa: E402
from evaluar_test import IOUV, ap_interpolada, iou_matriz      # noqa: E402
import v3_datos_lib as L                                       # noqa: E402
import v3_inferencia as VI                                     # noqa: E402

RUN = AQUI / "entrenamientov3"
MET = json.loads((RUN / "metricas_v3.json").read_text(encoding="utf-8"))
MODELOS = {
    "modelo 1": dict(onnx=RAIZ / "backend" / "modelos" / "best.onnx",
                     pt=RAIZ / "entrenamientos" / "yolov8m_huevos_v1" / "weights" / "best.pt",
                     tau_roto=CONFIG.tau_roto, tau_sano=CONFIG.tau_sano),
    "v3": dict(onnx=RUN / "weights" / "best.onnx", pt=RUN / "weights" / "best.pt",
               tau_roto=float(MET["tau_roto"]), tau_sano=float(MET["tau_sano"])),
}
NOMBRES = list(MODELOS)
CONF, IOU, NIT = CONFIG.conf_min_deteccion, CONFIG.iou_nms, CONFIG.nitidez_minima
TEST = RAIZ / "huevos" / "test" / "images"
CLASES = ["Roto", "Sano"]
RUTAS = ("DESCARTE", "REVISION", "ACEPTADO", "NO_DETECTADO")
BARRA = "=" * 96
SALIDA = {}


def titulo(t):
    print(f"\n{BARRA}\n{t}\n{BARRA}", flush=True)


def pct(k, n):
    return f"{100 * k / n:5.1f}%" if n else "  -  "


def ic(k, n):
    lo, hi = L.ic95(k, n)
    return f"[{100 * lo:.1f}-{100 * hi:.1f}]"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def cargar_gt(img: Path, W: int, H: int) -> list[tuple]:
    out = []
    for ln in (img.parent.parent / "labels" / (img.stem + ".txt")).read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
            out.append((CLASES[c], (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    return out


def inferir(det: DetectorHuevosONNX, im: Image.Image):
    """Una sola pasada de la red; el posproceso es el del backend. Filtrar la salida de NMS a
    conf>=0.15 da exactamente lo que el backend obtiene con conf_min=0.15 (NMS solo suprime de
    mayor a menor confianza)."""
    tensor, r, dw, dh = det._letterbox(im)
    salida = det.sesion.run(None, {det.nombre_entrada: tensor})[0]
    todas = det._postprocesar(salida, r, dw, dh, im.width, im.height, 0.001, IOU)
    ruteo = [c for c in todas if c.confianza >= CONF]
    for c in ruteo:
        c.nitidez = nitidez_recorte(im, c.x1, c.y1, c.x2, c.y2)
    return todas, ruteo


def stats_map(todas, gts):
    pred = np.array([[c.x1, c.y1, c.x2, c.y2, c.confianza, c.clase_idx] for c in todas], dtype=np.float32).reshape(-1, 6)
    gt_c = np.array([g[1:5] for g in gts], dtype=np.float32).reshape(-1, 4)
    gt_k = np.array([CLASES.index(g[0]) for g in gts], dtype=int)
    acierto = np.zeros((len(pred), len(IOUV)), dtype=bool)
    if len(pred) and len(gt_c):
        m = iou_matriz(pred[:, :4], gt_c)
        m = np.where(pred[:, 5][:, None] == gt_k[None, :], m, 0)
        for k, t in enumerate(IOUV):
            pares = np.argwhere(m >= t)
            if len(pares):
                v = m[pares[:, 0], pares[:, 1]]
                pares = pares[v.argsort()[::-1]]
                pares = pares[np.unique(pares[:, 1], return_index=True)[1]]
                pares = pares[np.unique(pares[:, 0], return_index=True)[1]]
                acierto[pares[:, 0], k] = True
    return acierto, pred[:, 4], pred[:, 5], gt_k


def map_de(stats):
    tp = np.concatenate([s[0] for s in stats], 0)
    conf = np.concatenate([s[1] for s in stats], 0)
    pcls = np.concatenate([s[2] for s in stats], 0)
    tcls = np.concatenate([s[3] for s in stats], 0)
    orden = conf.argsort()[::-1]
    tp, pcls = tp[orden], pcls[orden]
    res = {}
    for ci, nombre in enumerate(CLASES):
        n_gt, k = int((tcls == ci).sum()), pcls == ci
        if not n_gt or not k.sum():
            res[nombre] = (0.0, 0.0, n_gt)
            continue
        tpc, fpc = tp[k].cumsum(0), (1 - tp[k]).cumsum(0)
        rec, pre = tpc / n_gt, tpc / (tpc + fpc + 1e-9)
        ap = [ap_interpolada(rec[:, j], pre[:, j]) for j in range(len(IOUV))]
        res[nombre] = (ap[0], float(np.mean(ap)), n_gt)
    return float(np.mean([v[0] for v in res.values()])), float(np.mean([v[1] for v in res.values()])), res


def rutear(fila, tau_roto, tau_sano):
    if fila["p_roto"] is None:
        return "NO_DETECTADO"
    return VI.decidir(fila["p_roto"], fila["p_sano"], fila["nitidez"], tau_roto, tau_sano, NIT)


def huevos_de(ruteo, gts, img_nombre):
    preds = [(c.x1, c.y1, c.x2, c.y2, c.p_roto, c.p_sano, c.nitidez) for c in ruteo]
    filas = []
    for i, (g, p) in enumerate(zip(gts, VI.emparejar(gts, preds))):
        filas.append({"img": img_nombre, "i": i, "clase": g[0], "caja": g[1:5],
                      "p_roto": p[4] if p else None, "p_sano": p[5] if p else None, "nitidez": p[6] if p else None})
    return filas


def mcnemar(a: list[bool], b: list[bool]):
    """Exacto (binomial) sobre los pares discordantes. a, b = acierto por huevo de cada modelo."""
    solo_a = sum(1 for x, y in zip(a, b) if x and not y)
    solo_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = solo_a + solo_b
    if n == 0:
        return solo_a, solo_b, 1.0
    p = 2 * sum(math.comb(n, i) for i in range(min(solo_a, solo_b) + 1)) / 2 ** n
    return solo_a, solo_b, min(1.0, p)


def auc(pos: list[float], neg: list[float]) -> float:
    """P(p_roto de un roto > p_roto de un sano), empates a la mitad."""
    if not pos or not neg:
        return float("nan")
    a, b = np.array(pos)[:, None], np.array(neg)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))


# ------------------------------------------------------------------ pruebas de robustez
def rayar(im, gts):          # EXACTAMENTE la de diagnostico_campo.py y la celda 14 del notebook
    im = im.copy(); d = ImageDraw.Draw(im); rng = random.Random(3)
    for g in gts:
        if g[0] != "Sano":
            continue
        x1, y1, x2, y2 = g[1:5]; cx, alto = (x1 + x2) / 2, y2 - y1
        pts = [(cx + (x2 - x1) * 0.12 * np.sin(t * 3.1) + rng.uniform(-3, 3), y1 + alto * (0.12 + 0.76 * t))
               for t in (i / 13 for i in range(14))]
        d.line(pts, fill=(35, 25, 20), width=3, joint="curve")
    return im


def sellar(im, gts):         # la de la celda 14: fuente, texto y tinta distintos a los de train
    im = im.copy(); d = ImageDraw.Draw(im)
    for g in gts:
        if g[0] != "Sano":
            continue
        x1, y1, x2, y2 = g[1:5]
        fnt = ImageFont.load_default(size=max(10, int((y2 - y1) * 0.13)))
        d.text(((x1 + x2) / 2, (y1 + y2) / 2), "EXP 12-26 F7", fill=(200, 110, 20), font=fnt, anchor="mm")
    return im


def jpeg(im, gts, q=20):
    import io
    b = io.BytesIO(); im.save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


PRUEBAS = [("raya de lapicero en SANOS", rayar, "Sano"),
           ("sello de tinta en SANOS", sellar, "Sano"),
           ("desenfoque r=3 en ROTOS", lambda im, g: im.filter(ImageFilter.GaussianBlur(3)), "Roto"),
           ("sobreexposicion 2.4x en ROTOS", lambda im, g: ImageEnhance.Brightness(im).enhance(2.4), "Roto"),
           ("subexposicion 0.35x en ROTOS", lambda im, g: ImageEnhance.Brightness(im).enhance(0.35), "Roto"),
           ("JPEG calidad 20 en ROTOS", jpeg, "Roto"),
           ("JPEG calidad 20 en SANOS", jpeg, "Sano")]


def main():
    imagenes = sorted(TEST.glob("*.jpg"))
    dets = {n: DetectorHuevosONNX(m["onnx"], 640, 0) for n, m in MODELOS.items()}

    titulo("0) QUE SE COMPARA")
    for n, m in MODELOS.items():
        s = dets[n].sesion
        print(f"  {n:<9} {m['onnx'].relative_to(RAIZ)}  sha {sha(m['onnx'])}  {m['onnx'].stat().st_size / 1e6:.0f} MB"
              f" | salida {s.get_outputs()[0].shape} | clases {s.get_modelmeta().custom_metadata_map.get('names')}"
              f" | TAU_ROTO={m['tau_roto']:.2f} TAU_SANO={m['tau_sano']:.2f}")
    print(f"  comun: CONF_MIN={CONF} IOU_NMS={IOU} NITIDEZ_MIN={NIT} | {len(imagenes)} fotos de {TEST.relative_to(RAIZ)}")
    prod = RAIZ / "entrenamientos" / "yolov8m_huevos_v1" / "weights" / "best.onnx"
    print(f"  el ONNX del backend es el del entrenamiento del modelo 1: {sha(prod) == sha(MODELOS['modelo 1']['onnx'])}")

    # -------------------------------------------------------------- pasada base
    titulo("1) PASADA BASE (una inferencia por foto y modelo)")
    base = {n: {"stats": [], "huevos": []} for n in NOMBRES}
    gts_de = {}
    t0 = time.time()
    for k, img in enumerate(imagenes, 1):
        im = Image.open(img).convert("RGB")
        gts = cargar_gt(img, *im.size)
        gts_de[img.name] = gts
        for n in NOMBRES:
            todas, ruteo = inferir(dets[n], im)
            base[n]["stats"].append(stats_map(todas, gts))
            base[n]["huevos"].extend(huevos_de(ruteo, gts, img.name))
        if k % 30 == 0 or k == len(imagenes):
            print(f"  {k}/{len(imagenes)} fotos ({time.time() - t0:.0f} s)", flush=True)

    # -------------------------------------------------------------- deteccion
    titulo("2) DETECCION (protocolo COCO, conf 0.001, NMS del backend) + bootstrap por foto")
    print(f"  {'modelo':<10}{'mAP50':>8}{'mAP50-95':>10}   {'Roto AP50 / AP50-95':>22}   {'Sano AP50 / AP50-95':>22}")
    maps = {}
    for n in NOMBRES:
        m50, m95, por = map_de(base[n]["stats"])
        maps[n] = (m50, m95)
        print(f"  {n:<10}{m50:>8.4f}{m95:>10.4f}   {por['Roto'][0]:>10.4f} / {por['Roto'][1]:.4f}   {por['Sano'][0]:>10.4f} / {por['Sano'][1]:.4f}")
    rng = np.random.default_rng(0)
    d50, d95 = [], []
    idx_n = len(imagenes)
    for _ in range(1000):
        idx = rng.integers(0, idx_n, idx_n)
        a = map_de([base["modelo 1"]["stats"][i] for i in idx]); b = map_de([base["v3"]["stats"][i] for i in idx])
        d50.append(b[0] - a[0]); d95.append(b[1] - a[1])
    for nom, d in (("mAP50", d50), ("mAP50-95", d95)):
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  v3 - modelo 1 en {nom:<9}: {np.mean(d):+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]"
              f"  -> {'v3 mejor' if lo > 0 else ('modelo 1 mejor' if hi < 0 else 'sin diferencia demostrable')}")
    SALIDA["deteccion"] = {n: {"mAP50": maps[n][0], "mAP50-95": maps[n][1]} for n in NOMBRES}
    SALIDA["deteccion"]["dif_IC95"] = {"mAP50": list(np.percentile(d50, [2.5, 97.5])), "mAP50-95": list(np.percentile(d95, [2.5, 97.5]))}

    # -------------------------------------------------------------- ruteo con sus umbrales
    titulo("3) RUTEO POR HUEVO, cada modelo con SUS umbrales (lo que pasaria en la banda)")
    tablas = {}
    for n in NOMBRES:
        m = MODELOS[n]
        for f in base[n]["huevos"]:
            f["ruta"] = rutear(f, m["tau_roto"], m["tau_sano"])
        tablas[n] = {cl: Counter(f["ruta"] for f in base[n]["huevos"] if f["clase"] == cl) for cl in CLASES}
        print(f"\n  {n}  (TAU_ROTO={m['tau_roto']:.2f}, TAU_SANO={m['tau_sano']:.2f})")
        print(f"    {'real':<6}" + "".join(f"{r:>14}" for r in RUTAS) + f"{'n':>6}")
        for cl in CLASES:
            c = tablas[n][cl]; tot = sum(c.values())
            print(f"    {cl:<6}" + "".join(f"{c[r]:>5} {pct(c[r], tot):>8}" for r in RUTAS) + f"{tot:>6}")
    print(f"\n  {'KPI (IC95 Wilson)':<44}" + "".join(f"{n:>26}" for n in NOMBRES))
    kpis = [("rotos al EMPAQUE (aceptados)", "Roto", ("ACEPTADO",)),
            ("rotos NO detectados (siguen de largo)", "Roto", ("NO_DETECTADO",)),
            ("rotos que se escapan (las dos anteriores)", "Roto", ("ACEPTADO", "NO_DETECTADO")),
            ("rotos a revision", "Roto", ("REVISION",)),
            ("sanos BOTADOS (descarte)", "Sano", ("DESCARTE",)),
            ("sanos a revision", "Sano", ("REVISION",)),
            ("sanos aceptados (lo correcto)", "Sano", ("ACEPTADO",))]
    SALIDA["ruteo"] = {}
    for nom, cl, rs in kpis:
        linea = f"  {nom:<44}"
        for n in NOMBRES:
            c = tablas[n][cl]; tot = sum(c.values()); k = sum(c[r] for r in rs)
            linea += f"{k:>5}/{tot:<4}{pct(k, tot):>7} {ic(k, tot):>11}"
            SALIDA["ruteo"].setdefault(nom, {})[n] = [k, tot]
        print(linea)
    # pareado
    h1 = {(f["img"], f["i"]): f for f in base["modelo 1"]["huevos"]}
    h3 = {(f["img"], f["i"]): f for f in base["v3"]["huevos"]}
    claves = sorted(h1)
    rot = [k for k in claves if h1[k]["clase"] == "Roto"]; san = [k for k in claves if h1[k]["clase"] == "Sano"]
    a, b, p = mcnemar([h1[k]["ruta"] in ("DESCARTE", "REVISION") for k in rot], [h3[k]["ruta"] in ("DESCARTE", "REVISION") for k in rot])
    print(f"\n  McNemar ROTOS atrapados (descarte o revision): solo modelo 1 acierta {a} | solo v3 acierta {b} | p={p:.3f}")
    a2, b2, p2 = mcnemar([h1[k]["ruta"] == "ACEPTADO" for k in san], [h3[k]["ruta"] == "ACEPTADO" for k in san])
    print(f"  McNemar SANOS aceptados:                     solo modelo 1 acierta {a2} | solo v3 acierta {b2} | p={p2:.3f}")
    SALIDA["mcnemar"] = {"rotos": [a, b, p], "sanos": [a2, b2, p2]}

    # -------------------------------------------------------------- sin umbrales
    titulo("4) SIN UMBRALES: separa bien rotos de sanos? (solo huevos detectados por AMBOS, pareado)")
    ambos = [k for k in claves if h1[k]["p_roto"] is not None and h3[k]["p_roto"] is not None]
    for n, h in (("modelo 1", h1), ("v3", h3)):
        pos = [h[k]["p_roto"] for k in ambos if h[k]["clase"] == "Roto"]; neg = [h[k]["p_roto"] for k in ambos if h[k]["clase"] == "Sano"]
        det_r = sum(h[k]["p_roto"] is not None for k in rot); det_s = sum(h[k]["p_roto"] is not None for k in san)
        print(f"  {n:<9} AUC p(Roto) = {auc(pos, neg):.4f}  (n={len(pos)} rotos, {len(neg)} sanos)"
              f" | detecta a conf>=0.15: rotos {det_r}/{len(rot)}, sanos {det_s}/{len(san)}")
        SALIDA.setdefault("auc", {})[n] = auc(pos, neg)
    print(f"\n  Si el umbral de descarte fuera TAU (sobre TODOS los huevos; no detectado = no atrapado):")
    print(f"  {'TAU':>6}" + "".join(f"{n + ': rotos atrapados / sanos botados':>44}" for n in NOMBRES))
    for tau in (0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.70):
        linea = f"  {tau:>6.2f}"
        for h in (h1, h3):
            at = sum(h[k]["p_roto"] is not None and h[k]["p_roto"] >= tau for k in rot)
            bo = sum(h[k]["p_roto"] is not None and h[k]["p_roto"] >= tau for k in san)
            linea += f"{at:>18}/{len(rot)} {pct(at, len(rot))}  {bo:>6}/{len(san)} {pct(bo, len(san))}"
        print(linea)

    # -------------------------------------------------------------- robustez
    titulo("5) ROBUSTEZ: las fallas de campo, mismas fotos y transformaciones para los dos")
    print(f"  {'prueba':<32}{'modelo':<10}" + "".join(f"{r:>13}" for r in RUTAS) + f"{'n':>5}{'p(Roto) medio':>15}")
    SALIDA["robustez"] = {}
    for nombre, tf, cl in PRUEBAS:
        filas = {n: [] for n in NOMBRES}
        for img in imagenes:
            gts = gts_de[img.name]
            if not any(g[0] == cl for g in gts):
                continue
            im = tf(Image.open(img).convert("RGB"), gts)
            for n in NOMBRES:
                _, ruteo = inferir(dets[n], im)
                filas[n].extend(f for f in huevos_de(ruteo, gts, img.name) if f["clase"] == cl)
        for n in NOMBRES:
            m = MODELOS[n]; c = Counter(rutear(f, m["tau_roto"], m["tau_sano"]) for f in filas[n]); tot = sum(c.values())
            pr = [f["p_roto"] for f in filas[n] if f["p_roto"] is not None]
            print(f"  {nombre if n == NOMBRES[0] else '':<32}{n:<10}" + "".join(f"{c[r]:>5} {pct(c[r], tot):>7}" for r in RUTAS)
                  + f"{tot:>5}{np.mean(pr) if pr else float('nan'):>15.3f}")
            SALIDA["robustez"].setdefault(nombre, {})[n] = dict(c) | {"n": tot}
        print(flush=True)

    # -------------------------------------------------------------- desacuerdos
    titulo("6) DONDE NO COINCIDEN (base, sin transformar)")
    peores = []
    for k in claves:
        f1, f3 = h1[k], h3[k]
        mal1 = (f1["clase"] == "Roto" and f1["ruta"] in ("ACEPTADO", "NO_DETECTADO")) or (f1["clase"] == "Sano" and f1["ruta"] == "DESCARTE")
        mal3 = (f3["clase"] == "Roto" and f3["ruta"] in ("ACEPTADO", "NO_DETECTADO")) or (f3["clase"] == "Sano" and f3["ruta"] == "DESCARTE")
        if mal1 or mal3:
            peores.append((k, f1, f3, mal1, mal3))
    print(f"  errores graves (roto que se escapa / sano botado): modelo 1 {sum(x[3] for x in peores)} | v3 {sum(x[4] for x in peores)}"
          f" | de ambos {sum(x[3] and x[4] for x in peores)}")
    def fmt(f):
        p = "-" if f["p_roto"] is None else f"{f['p_roto']:.2f}"
        return f"{f['ruta']:<12} p_roto={p}"

    for (img, i), f1, f3, m1, m3 in peores:
        quien = "AMBOS" if m1 and m3 else ("solo m1" if m1 else "solo v3")
        print(f"  {quien:<8} {img[:44]:<45} #{i} real {f1['clase']:<5} | m1 {fmt(f1)} | v3 {fmt(f3)}")
    SALIDA["errores_graves"] = [{"img": img, "i": i, "real": f1["clase"], "m1": f1["ruta"], "v3": f3["ruta"],
                                 "p_roto_m1": f1["p_roto"], "p_roto_v3": f3["p_roto"]} for (img, i), f1, f3, _, _ in peores]
    hoja_de_errores(peores, RUN / "comparacion_errores.jpg")

    # -------------------------------------------------------------- latencia
    titulo("7) LATENCIA del camino de produccion (detectar: decodificar + red + NMS + nitidez)")
    muestra = [p.read_bytes() for p in imagenes[:40]]
    for hilos in (0, 2):
        linea = f"  hilos={hilos if hilos else 'todos'}:"
        for n in NOMBRES:
            d = DetectorHuevosONNX(MODELOS[n]["onnx"], 640, hilos)
            t = []
            for b in muestra:
                s = time.perf_counter(); d.detectar(b, CONF, IOU); t.append((time.perf_counter() - s) * 1000)
            linea += f"   {n}: media {np.mean(t):.0f} ms p95 {np.percentile(t, 95):.0f} ms"
            SALIDA.setdefault("latencia_ms", {})[f"{n}|hilos={hilos}"] = float(np.mean(t))
        print(linea, flush=True)
    r = SALIDA["latencia_ms"]["v3|hilos=2"] / SALIDA["latencia_ms"]["modelo 1|hilos=2"]
    print(f"  la v3 es {r:.2f}x mas lenta con 2 hilos (el t2.large tiene 2 vCPU)")

    # -------------------------------------------------------------- integridad v3
    titulo("8) INTEGRIDAD DE LOS ARCHIVOS DE LA v3")
    integridad_v3(dets["v3"], imagenes[:8])

    # -------------------------------------------------------------- val oficial
    titulo("9) CONTROL CRUZADO: `model.val()` de ultralytics sobre las mismas 145 fotos (.pt)")
    val_ultralytics(imagenes)

    (RUN / "comparacion_v1_v3.json").write_text(json.dumps(SALIDA, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nguardado: {(RUN / 'comparacion_v1_v3.json').relative_to(RAIZ)} y {(RUN / 'comparacion_errores.jpg').relative_to(RAIZ)}")


def hoja_de_errores(peores, destino: Path, lado=220, por_fila=5):
    if not peores:
        return
    celdas = []
    for (img, i), f1, f3, m1, m3 in peores:
        im = Image.open(TEST / img).convert("RGB")
        x1, y1, x2, y2 = f1["caja"]; mw, mh = (x2 - x1) * 0.25, (y2 - y1) * 0.25
        rec = im.crop((max(0, x1 - mw), max(0, y1 - mh), min(im.width, x2 + mw), min(im.height, y2 + mh)))
        rec.thumbnail((lado, lado - 58)); c = Image.new("RGB", (lado, lado), (255, 255, 255)); c.paste(rec, ((lado - rec.width) // 2, 0))
        d = ImageDraw.Draw(c); fnt = ImageFont.load_default(size=12)
        pr = lambda f: "-" if f["p_roto"] is None else f"{f['p_roto']:.2f}"
        d.text((4, lado - 56), f"REAL {f1['clase']}  {img[:22]}", fill=(0, 0, 0), font=fnt)
        d.text((4, lado - 38), f"m1 {f1['ruta'][:9]} p={pr(f1)}", fill=(180, 0, 0) if m1 else (0, 120, 0), font=fnt)
        d.text((4, lado - 20), f"v3 {f3['ruta'][:9]} p={pr(f3)}", fill=(180, 0, 0) if m3 else (0, 120, 0), font=fnt)
        celdas.append(c)
    filas = math.ceil(len(celdas) / por_fila)
    hoja = Image.new("RGB", (lado * por_fila, lado * filas), (230, 230, 230))
    for k, c in enumerate(celdas):
        hoja.paste(c, ((k % por_fila) * lado, (k // por_fila) * lado))
    hoja.save(destino, quality=90)


def integridad_v3(det, muestra):
    import torch
    from ultralytics import YOLO
    ck = torch.load(MODELOS["v3"]["pt"], map_location="cpu", weights_only=False)
    tm = ck.get("train_metrics") or {}
    print(f"  best.pt: ultralytics {ck.get('version')} | fecha {ck.get('date')} | epoch guardada {ck.get('epoch')}"
          f" | mAP50-95 de val al guardarlo {tm.get('metrics/mAP50-95(B)', float('nan')):.4f}")
    ta = ck.get("train_args") or {}
    import re
    aum = re.findall(r"__class_fullname__:\s*(\w+)", (RUN / "args.yaml").read_text(encoding="utf-8"))
    print(f"  train_args: modelo {ta.get('model')} imgsz {ta.get('imgsz')} epochs {ta.get('epochs')} close_mosaic {ta.get('close_mosaic')}"
          f" | cls_pw {ta.get('cls_pw')} | aumentaciones en args.yaml: {', '.join(aum) or 'NINGUNA'}")
    red = YOLO(str(MODELOS["v3"]["pt"])).model.float().eval()
    dmax_p, dmax_b = 0.0, 0.0
    for img in muestra:
        x, *_ = det._letterbox(Image.open(img).convert("RGB"))
        with torch.no_grad():
            o = red(torch.from_numpy(x)); o = (o[0] if isinstance(o, (list, tuple)) else o).numpy()
        q = det.sesion.run(None, {det.nombre_entrada: x})[0]
        dmax_p = max(dmax_p, float(np.abs(o[:, 4:] - q[:, 4:]).max())); dmax_b = max(dmax_b, float(np.abs(o[:, :4] - q[:, :4]).max()))
    print(f"  ONNX vs PyTorch en {len(muestra)} fotos: dif. max probabilidad {dmax_p:.2e} | cajas {dmax_b:.3f} px")
    SALIDA["integridad"] = {"dif_prob": dmax_p, "dif_caja_px": dmax_b, "epoch": ck.get("epoch"), "version": ck.get("version")}


def val_ultralytics(imagenes):
    from ultralytics import YOLO
    tmp = Path(tempfile.mkdtemp(prefix="cmp_val_"))
    try:
        for sub in ("images", "labels"):
            (tmp / "test" / sub).mkdir(parents=True)
        for img in imagenes:
            shutil.copy2(img, tmp / "test" / "images" / img.name)
            shutil.copy2(img.parent.parent / "labels" / (img.stem + ".txt"), tmp / "test" / "labels" / (img.stem + ".txt"))
        yml = tmp / "data.yaml"
        yml.write_text(f"path: {tmp.as_posix()}\ntrain: test/images\nval: test/images\nnc: 2\nnames: ['Roto', 'Sano']\n")
        for n in NOMBRES:
            m = YOLO(str(MODELOS[n]["pt"])).val(data=str(yml), split="val", imgsz=640, batch=8, device="cpu", plots=False,
                                               verbose=False, project=str(tmp / "runs"), name=n.replace(" ", "_"))
            print(f"  {n:<9} mAP50 {m.box.map50:.4f} | mAP50-95 {m.box.map:.4f} | P {m.box.mp:.4f} R {m.box.mr:.4f}"
                  f" | por clase mAP50-95 Roto {m.box.maps[0]:.4f} Sano {m.box.maps[1]:.4f}", flush=True)
            SALIDA.setdefault("val_ultralytics", {})[n] = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
