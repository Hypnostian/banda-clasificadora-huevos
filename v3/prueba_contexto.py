"""Prueba del ATAJO de contexto: el modelo mira el huevo o mira la foto?

En `huevos` (lo unico que vio el modelo 1) la fuente de la foto casi define la etiqueta:
  damaged: 656 Roto / 0 Sano | not_damaged: 0 / 124 | IMG (celular, en la mano): 2 / 86
Un modelo puede acertar reconociendo el TIPO DE FOTO en vez de la cascara. Si es asi:
  - un huevo ROTO en una foto de celular (contexto "sano") se escapa,
  - un huevo SANO en una foto tipo `damaged` (contexto "roto") se bota.

Diseno 2x2, iguales para los dos modelos: se pega un huevo (roto o sano) en el lugar del huevo
de una foto de contexto (celular o damaged), con mascara eliptica difuminada. Los huevos pegados
salen de fotos DISTINTAS a la del contexto. Mismo ruteo y umbrales que produccion.

Ademas: desglose del test base por tipo de foto y por color de cascara.

    <venv>/python v3/prueba_contexto.py
"""
from __future__ import annotations

import json
import random
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparar_v1_v3 as C   # noqa: E402  (mismos detectores, emparejamiento, ruteo y umbrales)

TEST = C.TEST
MIN_AREA = 0.04          # huevo entero, no fragmentos de cascara


def tipo_de(nombre: str) -> str:
    m = re.match(r"^([A-Za-z_]+?)(?:_?\d|$)", nombre.split(".rf.")[0])
    return {"IMG": "celular", "not_damaged": "not_damaged", "damaged": "damaged", "Egg_X": "Egg_X"}.get(m.group(1) if m else "", "?")


def etiquetas(img: Path):
    """[(clase, cx, cy, w, h)] normalizadas, como las usa rasgos_color."""
    out = []
    for ln in (img.parent.parent / "labels" / (img.stem + ".txt")).read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            out.append((C.CLASES[int(p[0])], *map(float, p[1:5])))
    return out


def a_pixeles(e, W, H):
    _, cx, cy, w, h = e
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


def pegar(contexto: Image.Image, caja, huevo: Image.Image) -> Image.Image:
    x1, y1, x2, y2 = (int(round(v)) for v in caja)
    w, h = max(2, x2 - x1), max(2, y2 - y1)
    parche = huevo.resize((w, h), Image.LANCZOS)
    mascara = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mascara).ellipse((1, 1, w - 2, h - 2), fill=255)
    mascara = mascara.filter(ImageFilter.GaussianBlur(max(1.0, 0.03 * min(w, h))))
    out = contexto.copy()
    out.paste(parche, (x1, y1), mascara)
    return out


def main():
    dets = {n: C.DetectorHuevosONNX(m["onnx"], 640, 0) for n, m in C.MODELOS.items()}
    color = json.loads((C.RUN / "metricas_v3.json").read_text(encoding="utf-8"))["color"]["umbral_saturacion"]
    imgs = sorted(TEST.glob("*.jpg"))
    resultado = {}

    # ------------------------------------------------------ 1) desglose del test base
    C.titulo("1) TEST BASE POR TIPO DE FOTO Y COLOR DE CASCARA (cada modelo con sus umbrales)")
    filas = []
    for img in imgs:
        im = Image.open(img).convert("RGB")
        es = etiquetas(img)
        gts = [(e[0], *a_pixeles(e, *im.size)) for e in es]
        for n in C.NOMBRES:
            _, ruteo = C.inferir(dets[n], im)
            for f, e in zip(C.huevos_de(ruteo, gts, img.name), es):
                f.update(modelo=n, tipo=tipo_de(img.name),
                         color="marron" if C.L.rasgos_color(im, e)[0] >= color else "blanco",
                         ruta=C.rutear(f, C.MODELOS[n]["tau_roto"], C.MODELOS[n]["tau_sano"]))
                filas.append(f)
    mal = lambda f: (f["clase"] == "Roto" and f["ruta"] in ("ACEPTADO", "NO_DETECTADO")) or (f["clase"] == "Sano" and f["ruta"] == "DESCARTE")
    for eje in ("tipo", "color"):
        print(f"\n  por {eje}: errores graves (roto que se escapa / sano botado) sobre huevos")
        for (valor, cl) in sorted({(f[eje], f["clase"]) for f in filas}):
            linea = f"    {valor:<12}{cl:<6}"
            for n in C.NOMBRES:
                fs = [f for f in filas if f["modelo"] == n and f[eje] == valor and f["clase"] == cl]
                k = sum(map(mal, fs)); linea += f"   {n}: {k:>3}/{len(fs):<4}{C.pct(k, len(fs))}"
                resultado.setdefault("desglose", {})[f"{eje}={valor}|{cl}|{n}"] = [k, len(fs)]
            print(linea)

    # ------------------------------------------------------ 2) cambio de contexto
    C.titulo("2) CAMBIO DE CONTEXTO: el mismo huevo pegado en fotos de celular vs fotos tipo `damaged`")
    rng = random.Random(7)
    contextos = {"celular (la fuente dice SANO)": [], "damaged (la fuente dice ROTO)": []}
    donantes = {"Roto": [], "Sano": []}
    for img in imgs:
        es = etiquetas(img); t = tipo_de(img.name)
        grandes = [e for e in es if e[3] * e[4] >= MIN_AREA and 0.5 <= (e[3] / max(e[4], 1e-6)) <= 2.0]
        if len(es) == 1 and grandes:
            if t == "celular":
                contextos["celular (la fuente dice SANO)"].append((img, grandes[0]))
            elif t == "damaged":
                contextos["damaged (la fuente dice ROTO)"].append((img, grandes[0]))
        for e in grandes:
            if t in ("damaged", "Egg_X") and e[0] == "Roto":
                donantes["Roto"].append((img, e))
            elif t == "not_damaged" and e[0] == "Sano":
                donantes["Sano"].append((img, e))
    rng.shuffle(contextos["damaged (la fuente dice ROTO)"])
    n_ctx = min(len(v) for v in contextos.values())
    contextos = {k: v[:n_ctx] for k, v in contextos.items()}
    POR_CONTEXTO = 6
    print(f"  {n_ctx} fotos de contexto por tipo | donantes: {len(donantes['Roto'])} rotos enteros, "
          f"{len(donantes['Sano'])} sanos | {POR_CONTEXTO} huevos de cada clase por contexto")
    muestras_vista = []
    tabla = {}
    for nombre_ctx, lista in contextos.items():
        for img, e in lista:
            base = Image.open(img).convert("RGB"); caja = a_pixeles(e, *base.size)
            for cl in ("Roto", "Sano"):
                cand = [d for d in donantes[cl] if d[0] != img]
                for dimg, de in random.Random(zlib.crc32(f"{img.name}|{cl}".encode())).sample(cand, min(POR_CONTEXTO, len(cand))):
                    dim = Image.open(dimg).convert("RGB")
                    huevo = dim.crop(tuple(int(round(v)) for v in a_pixeles(de, *dim.size)))
                    comp = pegar(base, caja, huevo)
                    gts = [(cl, *caja)]
                    for n in C.NOMBRES:
                        _, ruteo = C.inferir(dets[n], comp)
                        f = C.huevos_de(ruteo, gts, img.name)[0]
                        ruta = C.rutear(f, C.MODELOS[n]["tau_roto"], C.MODELOS[n]["tau_sano"])
                        tabla.setdefault((nombre_ctx, cl, n), []).append((ruta, f["p_roto"]))
                    if len(muestras_vista) < 12 and rng.random() < 0.15:
                        muestras_vista.append((comp.crop(tuple(int(v) for v in caja)), nombre_ctx, cl))
    print(f"\n  {'contexto':<32}{'huevo':<6}{'modelo':<10}{'DESCARTE':>10}{'REVISION':>10}{'ACEPTADO':>10}{'no det.':>9}"
          f"{'n':>5}{'p(Roto) med.':>14}")
    for (ctx, cl, n), v in sorted(tabla.items()):
        c = Counter(r for r, _ in v); tot = len(v); pr = [p for _, p in v if p is not None]
        print(f"  {ctx:<32}{cl:<6}{n:<10}{C.pct(c['DESCARTE'], tot):>10}{C.pct(c['REVISION'], tot):>10}"
              f"{C.pct(c['ACEPTADO'], tot):>10}{C.pct(c['NO_DETECTADO'], tot):>9}{tot:>5}"
              f"{(np.median(pr) if pr else float('nan')):>14.3f}")
        resultado.setdefault("contexto", {})[f"{ctx}|{cl}|{n}"] = dict(c) | {"n": tot}
    print("\n  Lectura: si un modelo mira el huevo, cada fila de ROTO deberia verse igual en los dos contextos,")
    print("  y lo mismo las de SANO. Si mira la foto, el roto se escapa en el contexto 'celular' y el sano")
    print("  se bota en el contexto 'damaged'.")
    for cl, malo in (("Roto", ("ACEPTADO", "NO_DETECTADO")), ("Sano", ("DESCARTE",))):
        for n in C.NOMBRES:
            tasas = []
            for ctx in contextos:
                v = tabla[(ctx, cl, n)]; tasas.append(sum(r in malo for r, _ in v) / len(v))
            print(f"  {n:<9} {cl}: error grave en contexto celular {100 * tasas[0]:5.1f}% | en contexto damaged "
                  f"{100 * tasas[1]:5.1f}% | efecto del contexto {100 * (tasas[0] - tasas[1]):+6.1f} pts")

    if muestras_vista:
        lado = 180; hoja = Image.new("RGB", (lado * 6, lado * 2 + 40), (255, 255, 255))
        fnt = ImageFont.load_default(size=11); d = ImageDraw.Draw(hoja)
        for k, (im, ctx, cl) in enumerate(muestras_vista[:12]):
            im = im.copy(); im.thumbnail((lado, lado - 22))
            x, y = (k % 6) * lado, (k // 6) * (lado + 20)
            hoja.paste(im, (x, y)); d.text((x + 2, y + lado - 20), f"{cl} en {ctx.split()[0]}", fill=(0, 0, 0), font=fnt)
        hoja.save(C.RUN / "prueba_contexto_muestras.jpg", quality=90)
    (C.RUN / "prueba_contexto.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nguardado: {(C.RUN / 'prueba_contexto.json').relative_to(C.RAIZ)} y prueba_contexto_muestras.jpg")


if __name__ == "__main__":
    main()
