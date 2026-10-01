"""Evalúa el modelo desplegado sobre un split etiquetado.

Usa EXACTAMENTE el mismo detector que corre en producción
(`app.detector.DetectorHuevosONNX`), no una copia. Es la evidencia de que las
métricas del informe son las del sistema que se despliega, no las de otro
pipeline que casualmente se le parece.

    python scripts/evaluar_test.py                       # split test
    python scripts/evaluar_test.py --split valid
    python scripts/evaluar_test.py --dir /ruta/a/images  # carpeta suelta

Reporta:
  * P / R / mAP50 / mAP50-95 global y por clase (protocolo COCO, 101 puntos)
  * tabla de FUGA de rotos por umbral (el KPI que decide la política de ruteo)
  * latencia del pipeline completo
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from app.config import CLASES, CONFIG          # noqa: E402
from app.detector import DetectorHuevosONNX    # noqa: E402

IOUV = np.linspace(0.5, 0.95, 10)
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iou_matriz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def ap_interpolada(rec: np.ndarray, pre: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], pre, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    # numpy 2.x renombro trapz -> trapezoid; el default de getattr se evalua
    # siempre, asi que no sirve como fallback.
    trapecio = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapecio(np.interp(x, mrec, mpre), x))


def cargar_gt(ruta_img: Path, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    partes = list(ruta_img.parts)
    for i in range(len(partes) - 2, -1, -1):   # el ultimo componente es el archivo
        if partes[i] == "images":
            partes[i] = "labels"
            break
    etiqueta = Path(*partes).with_suffix(".txt")
    cajas, clases = [], []
    if etiqueta.exists():
        for linea in etiqueta.read_text().splitlines():
            p = linea.split()
            if len(p) < 5:
                continue
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
            cajas.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
            clases.append(c)
    return np.array(cajas, dtype=np.float32).reshape(-1, 4), np.array(clases, dtype=int)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["test", "valid", "train"])
    p.add_argument("--dir", default=None, help="carpeta de imágenes (ignora --split)")
    p.add_argument("--conf", type=float, default=0.001, help="umbral para las métricas COCO")
    args = p.parse_args()

    carpeta = Path(args.dir) if args.dir else RAIZ.parent / "huevos" / args.split / "images"
    imagenes = sorted(f for f in carpeta.iterdir() if f.suffix.lower() in EXT)
    if not imagenes:
        raise SystemExit(f"No hay imágenes en {carpeta}")

    det = DetectorHuevosONNX(CONFIG.modelo_path, CONFIG.modelo_imgsz, CONFIG.hilos_inferencia)
    print(f"Modelo: {CONFIG.modelo_path.name} · {det.proveedor}")
    print(f"Split : {carpeta}  ({len(imagenes)} imágenes)\n")

    registros, estadisticas, latencias = [], [], []
    for i, img in enumerate(imagenes, 1):
        datos = img.read_bytes()
        t0 = time.perf_counter()
        cajas, lat, (W, H) = det.detectar(datos, args.conf, CONFIG.iou_nms)
        latencias.append((time.perf_counter() - t0) * 1000)
        if i % 25 == 0 or i == len(imagenes):
            print(f"  procesadas {i}/{len(imagenes)}")

        pred = np.array([[c.x1, c.y1, c.x2, c.y2, c.confianza, c.clase_idx] for c in cajas],
                        dtype=np.float32).reshape(-1, 6)
        pred = pred[pred[:, 4].argsort()[::-1]] if len(pred) else pred
        gt_cajas, gt_clases = cargar_gt(img, W, H)
        registros.append((gt_cajas, gt_clases, pred))

        acierto = np.zeros((len(pred), len(IOUV)), dtype=bool)
        if len(pred) and len(gt_cajas):
            m = iou_matriz(pred[:, :4], gt_cajas)
            m = np.where(pred[:, 5][:, None] == gt_clases[None, :], m, 0)
            for k, t in enumerate(IOUV):
                pares = np.argwhere(m >= t)
                if len(pares):
                    v = m[pares[:, 0], pares[:, 1]]
                    pares = pares[v.argsort()[::-1]]
                    pares = pares[np.unique(pares[:, 1], return_index=True)[1]]
                    pares = pares[np.unique(pares[:, 0], return_index=True)[1]]
                    acierto[pares[:, 0], k] = True
        estadisticas.append((acierto,
                             pred[:, 4] if len(pred) else np.zeros(0),
                             pred[:, 5] if len(pred) else np.zeros(0),
                             gt_clases))

    # ------------------------------------------------------------ métricas AP
    tp = np.concatenate([s[0] for s in estadisticas], 0)
    conf = np.concatenate([s[1] for s in estadisticas], 0)
    pcls = np.concatenate([s[2] for s in estadisticas], 0)
    tcls = np.concatenate([s[3] for s in estadisticas], 0)
    orden = conf.argsort()[::-1]
    tp, pcls = tp[orden], pcls[orden]

    barra = "=" * 76
    print(f"\n{barra}\nMÉTRICAS (protocolo COCO, conf={args.conf}, IoU NMS={CONFIG.iou_nms})\n{barra}")
    print(f"{'clase':<8}{'inst':>7}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>11}")
    ps, rs, ap50s, aps = [], [], [], []
    for ci, nombre in enumerate(CLASES):
        n_gt = int((tcls == ci).sum())
        k = pcls == ci
        if not n_gt or not k.sum():
            continue
        tpc, fpc = tp[k].cumsum(0), (1 - tp[k]).cumsum(0)
        rec, pre = tpc / n_gt, tpc / (tpc + fpc + 1e-9)
        ap = [ap_interpolada(rec[:, j], pre[:, j]) for j in range(len(IOUV))]
        f1 = 2 * pre[:, 0] * rec[:, 0] / (pre[:, 0] + rec[:, 0] + 1e-9)
        b = int(f1.argmax())
        print(f"{nombre:<8}{n_gt:>7}{pre[b,0]:>10.4f}{rec[b,0]:>10.4f}{ap[0]:>10.4f}{np.mean(ap):>11.4f}")
        ps.append(pre[b, 0]); rs.append(rec[b, 0]); ap50s.append(ap[0]); aps.append(np.mean(ap))
    print(f"{'todas':<8}{len(tcls):>7}{np.mean(ps):>10.4f}{np.mean(rs):>10.4f}"
          f"{np.mean(ap50s):>10.4f}{np.mean(aps):>11.4f}")

    # --------------------------------------------------- fuga por umbral
    print(f"\n{barra}\nFUGA DE ROTOS POR UMBRAL (IoU 0.5) — el KPI de la banda\n{barra}")
    print(f"{'conf':>6}{'RotoOK':>9}{'Roto>Sano':>11}{'NoDetect':>10}{'FUGA%':>9}"
          f"{'SanoOK':>9}{'FPfondo':>9}{'costo':>9}")
    c_fn, c_fp = CONFIG.costo_falso_negativo, CONFIG.costo_falso_positivo
    mejor = (float("inf"), None)
    for umbral in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.60, 0.70):
        cm = np.zeros((3, 3), dtype=int)
        for gt_cajas, gt_clases, pred in registros:
            d = pred[pred[:, 4] >= umbral] if len(pred) else pred
            usados_gt, usados_d = set(), set()
            if len(d) and len(gt_cajas):
                m = iou_matriz(d[:, :4], gt_cajas)
                pares = np.argwhere(m >= 0.5)
                if len(pares):
                    v = m[pares[:, 0], pares[:, 1]]
                    for di, gi in pares[v.argsort()[::-1]]:
                        if di in usados_d or gi in usados_gt:
                            continue
                        usados_d.add(di); usados_gt.add(gi)
                        cm[int(d[di, 5]), gt_clases[gi]] += 1
            for gi in range(len(gt_cajas)):
                if gi not in usados_gt:
                    cm[2, gt_clases[gi]] += 1
            for di in range(len(d)):
                if di not in usados_d:
                    cm[int(d[di, 5]), 2] += 1
        roto_ok, roto_sano, roto_no = cm[0, 0], cm[1, 0], cm[2, 0]
        n = max(1, roto_ok + roto_sano + roto_no)
        fugas, fp = roto_sano + roto_no, cm[0, 2] + cm[1, 2]
        # Costo total bajo la matriz asimetrica del .env: cada roto que se cuela
        # pesa c_fn, cada deteccion espuria pesa c_fp.
        costo = fugas * c_fn + fp * c_fp
        mejor = min(mejor, (costo, umbral))
        print(f"{umbral:>6.2f}{roto_ok:>9}{roto_sano:>11}{roto_no:>10}"
              f"{100*fugas/n:>9.2f}{cm[1,1]:>9}{fp:>9}{costo:>9.0f}")
    print(f"\n  Umbral de costo minimo con c_fn:c_fp = {c_fn:g}:{c_fp:g}  ->  "
          f"CONF_MIN_DETECCION={mejor[1]:.2f}  (costo {mejor[0]:.0f})")

    lat = sorted(latencias[3:]) or latencias
    print(f"\n{barra}\nLATENCIA DEL PIPELINE COMPLETO\n{barra}")
    print(f"media={np.mean(lat):.1f} ms · p50={lat[len(lat)//2]:.1f} ms · "
          f"p95={lat[int(len(lat)*0.95)]:.1f} ms · techo {3600/(np.mean(lat)/1000):,.0f} huevos/h")


if __name__ == "__main__":
    main()
