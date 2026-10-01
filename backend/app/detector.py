"""Inferencia YOLOv8 sobre ONNX Runtime, sin PyTorch y sin Ultralytics.

El .onnx exportado NO trae NMS embebido (end2end=False, nms=False), asi que el
pre y el post-procesamiento van aqui, en numpy puro. Esta implementacion fue
validada contra el split de test reproduciendo las metricas del run original
(mAP50-95 test = 0.9238).

Salida del grafo: output0 con forma (1, 6, 8400)
    filas 0..3 -> cx, cy, w, h en el espacio letterboxed de 640x640
    filas 4..5 -> score de la clase 0 (Roto) y clase 1 (Sano), ya con sigmoide
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .config import CLASES


@dataclass(slots=True)
class Caja:
    """Una deteccion cruda, antes de aplicar politica de negocio."""

    x1: float
    y1: float
    x2: float
    y2: float
    p_roto: float
    p_sano: float
    nitidez: float = 0.0
    """Varianza del laplaciano del recorte del huevo. Baja = foto movida."""

    @property
    def clase_idx(self) -> int:
        return 0 if self.p_roto >= self.p_sano else 1

    @property
    def clase(self) -> str:
        return CLASES[self.clase_idx]

    @property
    def confianza(self) -> float:
        return max(self.p_roto, self.p_sano)

    @property
    def bbox(self) -> list[float]:
        return [round(self.x1, 1), round(self.y1, 1), round(self.x2, 1), round(self.y2, 1)]


def nitidez_recorte(im: Image.Image, x1: float, y1: float, x2: float, y2: float,
                    lado: int = 192) -> float:
    """Varianza del laplaciano del recorte, normalizado a `lado`x`lado`.

    Se mide sobre el HUEVO y no sobre el frame completo: la textura del mantel
    o de la servilleta no dice nada sobre si la cascara quedo enfocada.
    Normalizar el tamano hace que el numero no dependa de que tan cerca este
    la camara.
    """
    a, b = max(0, int(x1)), max(0, int(y1))
    c, d = min(im.width, int(x2)), min(im.height, int(y2))
    if c - a < 8 or d - b < 8:
        return 0.0
    g = np.asarray(
        im.crop((a, b, c, d)).convert("L").resize((lado, lado), Image.BILINEAR),
        dtype=np.float32,
    )
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def _nms(cajas: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    x1, y1, x2, y2 = cajas.T
    areas = (x2 - x1) * (y2 - y1)
    orden = scores.argsort()[::-1]
    conservar: list[int] = []
    while orden.size:
        i = orden[0]
        conservar.append(int(i))
        if orden.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[orden[1:]])
        yy1 = np.maximum(y1[i], y1[orden[1:]])
        xx2 = np.minimum(x2[i], x2[orden[1:]])
        yy2 = np.minimum(y2[i], y2[orden[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[orden[1:]] - inter + 1e-9)
        orden = orden[1:][iou <= iou_thr]
    return np.asarray(conservar, dtype=int)


class DetectorHuevosONNX:
    """Envuelve la sesion de ONNX Runtime. No sabe nada de negocio ni de rutas."""

    def __init__(self, modelo: Path, imgsz: int = 640, hilos: int = 0) -> None:
        if not Path(modelo).exists():
            raise FileNotFoundError(
                f"No encuentro el modelo en {modelo}. "
                "Copia best.onnx a backend/modelos/ (ver scripts/setup_aws.sh)."
            )
        opciones = ort.SessionOptions()
        opciones.intra_op_num_threads = hilos
        opciones.inter_op_num_threads = 1
        opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sesion = ort.InferenceSession(
            str(modelo), opciones, providers=["CPUExecutionProvider"]
        )
        self.nombre_entrada = self.sesion.get_inputs()[0].name
        self.imgsz = imgsz
        self.modelo_path = str(modelo)
        self.proveedor = self.sesion.get_providers()[0]
        self._calentar()

    def _calentar(self) -> None:
        """La primera inferencia de ORT es 3-5x mas lenta. La quemamos al arrancar."""
        vacio = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
        for _ in range(2):
            self.sesion.run(None, {self.nombre_entrada: vacio})

    # ------------------------------------------------------------- pipeline
    def _letterbox(self, im: Image.Image) -> tuple[np.ndarray, float, int, int]:
        w, h = im.size
        r = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        lienzo = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))
        dw, dh = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        lienzo.paste(im.resize((nw, nh), Image.BILINEAR), (dw, dh))
        tensor = np.asarray(lienzo, dtype=np.float32).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(tensor / 255.0), r, dw, dh

    def detectar(
        self, imagen_bytes: bytes, conf_min: float, iou_nms: float
    ) -> tuple[list[Caja], dict[str, float], tuple[int, int]]:
        t0 = time.perf_counter()
        im = Image.open(io.BytesIO(imagen_bytes))
        if im.mode != "RGB":
            im = im.convert("RGB")
        W, H = im.size
        tensor, r, dw, dh = self._letterbox(im)
        t1 = time.perf_counter()

        salida = self.sesion.run(None, {self.nombre_entrada: tensor})[0]
        t2 = time.perf_counter()

        cajas = self._postprocesar(salida, r, dw, dh, W, H, conf_min, iou_nms)
        for caja in cajas:
            caja.nitidez = nitidez_recorte(im, caja.x1, caja.y1, caja.x2, caja.y2)
        t3 = time.perf_counter()

        latencias = {
            "preproceso_ms": round((t1 - t0) * 1000, 2),
            "inferencia_ms": round((t2 - t1) * 1000, 2),
            "postproceso_ms": round((t3 - t2) * 1000, 2),
            "total_ms": round((t3 - t0) * 1000, 2),
        }
        return cajas, latencias, (W, H)

    def _postprocesar(
        self,
        salida: np.ndarray,
        r: float,
        dw: int,
        dh: int,
        W: int,
        H: int,
        conf_min: float,
        iou_nms: float,
    ) -> list[Caja]:
        p = salida[0]
        if p.shape[0] < p.shape[1]:   # (6, 8400) -> (8400, 6)
            p = p.T
        xywh, scores = p[:, :4], p[:, 4:]
        mejor = scores.max(1)
        mascara = mejor >= conf_min
        if not mascara.any():
            return []
        xywh, scores, mejor = xywh[mascara], scores[mascara], mejor[mascara]

        cx, cy, w, h = xywh.T
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        # deshacer el letterbox para volver a pixeles del frame original
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dw) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dh) / r
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)

        idx_clase = scores.argmax(1)
        conservados: list[int] = []
        for c in np.unique(idx_clase):
            k = np.flatnonzero(idx_clase == c)
            conservados.extend(k[_nms(xyxy[k], mejor[k], iou_nms)].tolist())

        conservados.sort(key=lambda i: -mejor[i])
        return [
            Caja(
                x1=float(xyxy[i, 0]),
                y1=float(xyxy[i, 1]),
                x2=float(xyxy[i, 2]),
                y2=float(xyxy[i, 3]),
                p_roto=float(scores[i, 0]),
                p_sano=float(scores[i, 1]),
            )
            for i in conservados
        ]
