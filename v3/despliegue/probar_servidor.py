"""Prueba la API desplegada como lo hace la app: sube fotos y muestra la ruta de cada huevo.

    python v3/despliegue/probar_servidor.py                          # v3 en el servidor
    python v3/despliegue/probar_servidor.py --url http://127.0.0.1:18001   # p. ej. modelo 1 por tunel SSH

Casos: huevo sano, el MISMO sano con una raya de lapicero (la falla de campo del modelo 1),
un sano marron en la mano (el punto debil de la v3, lo recoge la banda de revision) y rotos.
"""
from __future__ import annotations

import argparse
import io
import random
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance

RAIZ = Path(__file__).resolve().parents[2]
TEST = RAIZ / "huevos" / "test" / "images"


def rayar(im: Image.Image, caja) -> Image.Image:
    """La misma raya de lapicero de backend/scripts/diagnostico_campo.py."""
    im = im.copy(); d = ImageDraw.Draw(im); rng = random.Random(3)
    x1, y1, x2, y2 = caja; cx, alto = (x1 + x2) / 2, y2 - y1
    pts = [(cx + (x2 - x1) * 0.12 * np.sin(t * 3.1) + rng.uniform(-3, 3), y1 + alto * (0.12 + 0.76 * t))
           for t in (i / 13 for i in range(14))]
    d.line(pts, fill=(35, 25, 20), width=3, joint="curve")
    return im


def caja_gt(img: Path, im: Image.Image):
    c, cx, cy, w, h = (float(v) for v in (img.parent.parent / "labels" / (img.stem + ".txt")).read_text().split()[:5])
    W, H = im.size
    return (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H


def jpg(im: Image.Image) -> bytes:
    b = io.BytesIO(); im.save(b, "JPEG", quality=90); return b.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://34.236.46.110:8000")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    salud = requests.get(f"{url}/salud", timeout=10).json()
    politica = requests.get(f"{url}/politica", timeout=10).json()
    print(f"{url}  ->  modelo: {salud['modelo']} | tau_roto={politica['tau_roto']} "
          f"tau_sano={politica['tau_sano']} tau_revision={politica.get('tau_revision')}")

    sano = next(TEST.glob("not_damaged_1*.jpg"))
    marron = next(TEST.glob("IMG_3346*.jpg"))
    rotos = sorted(TEST.glob("damaged_1*.jpg"))[:3]
    im_sano = Image.open(sano).convert("RGB")
    casos = [("sano (limpio)", "Sano", jpg(im_sano)),
             ("el MISMO sano con raya de lapicero", "Sano", jpg(rayar(im_sano, caja_gt(sano, im_sano)))),
             ("sano marron en la mano", "Sano", marron.read_bytes())]
    casos += [(f"roto {p.name[:14]}", "Roto", p.read_bytes()) for p in rotos]
    im_r = Image.open(rotos[0]).convert("RGB")
    casos.append(("roto sobreexpuesto 1.8x", "Roto", jpg(ImageEnhance.Brightness(im_r).enhance(1.8))))

    print(f"\n{'caso':<38}{'real':<6}{'ruta':<11}{'p(Roto)':>8}{'p(Sano)':>8}{'servidor':>10}{'ida y vuelta':>14}")
    for nombre, real, datos in casos:
        t0 = time.perf_counter()
        r = requests.post(f"{url}/inspeccion", files={"imagen": ("frame.jpg", datos, "image/jpeg")},
                          data={"estacion": "prueba-despliegue"}, timeout=30)
        r.raise_for_status(); res = r.json(); total = (time.perf_counter() - t0) * 1000
        d = max(res["detecciones"], key=lambda x: x["confianza"]) if res["detecciones"] else None
        if d is None:
            print(f"{nombre:<38}{real:<6}{'NO DETECTA':<11}{'':>16}{res['latencia_ms']['total_ms']:>8.0f}ms{total:>12.0f}ms")
        else:
            print(f"{nombre:<38}{real:<6}{d['ruta']:<11}{d['p_roto']:>8.2f}{d['p_sano']:>8.2f}"
                  f"{res['latencia_ms']['total_ms']:>8.0f}ms{total:>12.0f}ms")
        time.sleep(2)      # > seguimiento_ttl_s: que el tracker no junte un caso con el siguiente


if __name__ == "__main__":
    sys.exit(main())
