# =============================================================================
#  v3 — inferencia IDENTICA a la del backend (backend/app/detector.py + routing.py)
#
#  "Evaluar lo que se despliega": el notebook mide con el mismo letterbox, el
#  mismo NMS en numpy, las mismas dos probabilidades por huevo y la misma
#  compuerta de nitidez que corren en el servidor. Probado contra el backend
#  real con v3/validar_notebook_v3.py (mismas cajas, mismas probabilidades).
# =============================================================================
from __future__ import annotations

import numpy as np
from PIL import Image


def letterbox(im: Image.Image, imgsz: int = 640):
    w, h = im.size
    r = min(imgsz / w, imgsz / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    lienzo = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    dw, dh = (imgsz - nw) // 2, (imgsz - nh) // 2
    lienzo.paste(im.resize((nw, nh), Image.BILINEAR), (dw, dh))
    x = np.asarray(lienzo, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x), r, dw, dh


def _nms(cajas, scores, iou_thr):
    x1, y1, x2, y2 = cajas.T
    areas = (x2 - x1) * (y2 - y1)
    orden = scores.argsort()[::-1]
    keep = []
    while orden.size:
        i = orden[0]
        keep.append(int(i))
        if orden.size == 1:
            break
        xx1, yy1 = np.maximum(x1[i], x1[orden[1:]]), np.maximum(y1[i], y1[orden[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[orden[1:]]), np.minimum(y2[i], y2[orden[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        orden = orden[1:][inter / (areas[i] + areas[orden[1:]] - inter + 1e-9) <= iou_thr]
    return np.asarray(keep, dtype=int)


def postprocesar(salida: np.ndarray, r, dw, dh, W, H, conf_min: float, iou_nms: float) -> list[tuple]:
    """-> [(x1, y1, x2, y2, p_roto, p_sano)] en pixeles del original. Igual que el backend."""
    p = salida[0]
    if p.shape[0] < p.shape[1]:
        p = p.T
    xywh, scores = p[:, :4], p[:, 4:]
    mejor = scores.max(1)
    m = mejor >= conf_min
    if not m.any():
        return []
    xywh, scores, mejor = xywh[m], scores[m], mejor[m]
    cx, cy, w, h = xywh.T
    b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    b[:, [0, 2]] = ((b[:, [0, 2]] - dw) / r).clip(0, W)
    b[:, [1, 3]] = ((b[:, [1, 3]] - dh) / r).clip(0, H)
    idx = scores.argmax(1)
    keep = []
    for c in np.unique(idx):
        k = np.flatnonzero(idx == c)
        keep.extend(k[_nms(b[k], mejor[k], iou_nms)].tolist())
    keep.sort(key=lambda i: -mejor[i])
    return [(float(b[i, 0]), float(b[i, 1]), float(b[i, 2]), float(b[i, 3]), float(scores[i, 0]), float(scores[i, 1]))
            for i in keep]


def nitidez_recorte(im: Image.Image, x1, y1, x2, y2, lado: int = 192) -> float:
    a, b = max(0, int(x1)), max(0, int(y1))
    c, d = min(im.width, int(x2)), min(im.height, int(y2))
    if c - a < 8 or d - b < 8:
        return 0.0
    g = np.asarray(im.crop((a, b, c, d)).convert("L").resize((lado, lado), Image.BILINEAR), dtype=np.float32)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def decidir(p_roto, p_sano, nitidez, tau_roto, tau_sano, nitidez_min=4.0) -> str:
    """Politica del backend (routing.MotorDecision): DESCARTE / ACEPTADO / REVISION."""
    if nitidez is not None and nitidez < nitidez_min and p_roto < tau_roto:
        return "REVISION"
    if p_roto >= tau_roto:
        return "DESCARTE"
    if p_sano >= tau_sano:
        return "ACEPTADO"
    return "REVISION"


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def emparejar(gts: list, preds: list, umbral: float = 0.5) -> list:
    """Cada huevo real -> la deteccion con mayor IoU (>= umbral), sin repetir. None si no se detecto."""
    usadas, out = set(), []
    for g in gts:
        mejor, mi = None, umbral
        for k, p in enumerate(preds):
            v = iou(g[1:5], p[:4])
            if v >= mi and k not in usadas:
                mejor, mi = k, v
        if mejor is not None:
            usadas.add(mejor)
        out.append(None if mejor is None else preds[mejor])
    return out
