"""Diagnostico de los fallos reportados en campo.

Tres hipotesis, tres experimentos controlados sobre el split de test:

  H1  El modelo aprendio el atajo "linea oscura y fina sobre la cascara = Roto".
      -> dibujo una linea de lapicero sobre huevos SANOS y veo si se voltean.

  H2  El desenfoque (foto de cerca con poca luz) borra la fisura y empuja los
      ROTOS hacia Sano.
      -> aplico desenfoque gaussiano creciente a huevos ROTOS.

  H3  La exposicion (poca luz / sobreexposicion) degrada la deteccion.
      -> escalo el brillo hacia arriba y hacia abajo.
"""
import sys, glob, random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

RAIZ = Path(r'D:\eggdetection\backend')
sys.path.insert(0, str(RAIZ))
from app.config import CONFIG              # noqa: E402
from app.detector import DetectorHuevosONNX  # noqa: E402

import io

det = DetectorHuevosONNX(CONFIG.modelo_path, 640, 0)
TEST = Path(r'D:\eggdetection\huevos\test\images')
random.seed(11)


def jpg(im, q=88):
    b = io.BytesIO(); im.save(b, 'JPEG', quality=q); return b.getvalue()


def mejor(cajas, clase=None):
    """Caja de mayor confianza, opcionalmente de una clase dada."""
    c = [k for k in cajas if clase is None or k.clase == clase]
    return max(c, key=lambda k: k.confianza) if c else None


def etiqueta_de(p: Path) -> str:
    return 'Roto' if p.name.startswith('damaged') else ('Sano' if p.name.startswith('not_damaged') else '?')


sanos = [p for p in sorted(TEST.glob('*.jpg')) if etiqueta_de(p) == 'Sano']
rotos = [p for p in sorted(TEST.glob('*.jpg')) if etiqueta_de(p) == 'Roto']
print(f'test: {len(sanos)} sanos identificables, {len(rotos)} rotos identificables\n')

bar = '=' * 78

# --------------------------------------------------------------------------
# H1: la raya de lapicero
# --------------------------------------------------------------------------
print(bar); print('H1  UNA RAYA DE LAPICERO SOBRE UN HUEVO SANO'); print(bar)


def rayar(im: Image.Image, caja, grosor=3, color=(35, 25, 20), curvatura=0.12):
    """Traza una linea oscura irregular a lo largo del huevo, como un lapicero."""
    im = im.copy()
    d = ImageDraw.Draw(im)
    x1, y1, x2, y2 = caja.x1, caja.y1, caja.x2, caja.y2
    cx = (x1 + x2) / 2
    alto = y2 - y1
    puntos = []
    rng = random.Random(3)
    for i in range(14):
        t = i / 13
        y = y1 + alto * (0.12 + 0.76 * t)
        x = cx + (x2 - x1) * curvatura * np.sin(t * 3.1) + rng.uniform(-3, 3)
        puntos.append((x, y))
    d.line(puntos, fill=color, width=grosor, joint='curve')
    return im


n, volteados, subio = 0, 0, []
print(f"{'imagen':<34}{'p(Roto) limpio':>16}{'p(Roto) rayado':>16}{'ruta':>12}")
for p in sanos[:20]:
    im = Image.open(p).convert('RGB')
    c0 = mejor(det.detectar(jpg(im), 0.05, 0.45)[0])
    if c0 is None or c0.clase != 'Sano':
        continue
    im2 = rayar(im, c0)
    c1 = mejor(det.detectar(jpg(im2), 0.05, 0.45)[0])
    if c1 is None:
        continue
    n += 1
    subio.append(c1.p_roto - c0.p_roto)
    ruta = 'DESCARTE' if c1.p_roto >= CONFIG.tau_roto else ('ACEPTADO' if c1.p_sano >= CONFIG.tau_sano else 'REVISION')
    if ruta == 'DESCARTE':
        volteados += 1
    if n <= 12:
        print(f'{p.name[:33]:<34}{c0.p_roto:>16.3f}{c1.p_roto:>16.3f}{ruta:>12}')
print(f'\n  {volteados}/{n} huevos SANOS pasaron a DESCARTE solo por la raya')
print(f'  subida media de p(Roto): {np.mean(subio):+.3f}  (mediana {np.median(subio):+.3f})')

# --------------------------------------------------------------------------
# H2: desenfoque
# --------------------------------------------------------------------------
print('\n' + bar); print('H2  DESENFOQUE SOBRE HUEVOS ROTOS  (foto movida / mal enfocada)'); print(bar)
radios = [0, 1, 2, 3, 4, 6]
print(f"{'radio blur':>11}{'p(Roto) medio':>15}{'detectados':>12}{'-> DESCARTE':>13}{'-> REVISION':>13}{'-> ACEPTADO':>13}")
muestra_r = rotos[:25]
for r in radios:
    ps, nd, rutas = [], 0, {'DESCARTE': 0, 'REVISION': 0, 'ACEPTADO': 0}
    for p in muestra_r:
        im = Image.open(p).convert('RGB')
        if r:
            im = im.filter(ImageFilter.GaussianBlur(r))
        c = mejor(det.detectar(jpg(im), 0.05, 0.45)[0])
        if c is None:
            continue
        nd += 1
        ps.append(c.p_roto)
        ruta = 'DESCARTE' if c.p_roto >= CONFIG.tau_roto else ('ACEPTADO' if c.p_sano >= CONFIG.tau_sano else 'REVISION')
        rutas[ruta] += 1
    print(f'{r:>11}{np.mean(ps):>15.3f}{nd:>12}{rutas["DESCARTE"]:>13}{rutas["REVISION"]:>13}{rutas["ACEPTADO"]:>13}')

# --------------------------------------------------------------------------
# H3: exposicion
# --------------------------------------------------------------------------
print('\n' + bar); print('H3  EXPOSICION  (poca luz vs sobreexpuesto) SOBRE HUEVOS ROTOS'); print(bar)
factores = [0.35, 0.5, 0.7, 1.0, 1.4, 1.8, 2.4]
print(f"{'brillo x':>10}{'p(Roto) medio':>15}{'detectados':>12}{'-> DESCARTE':>13}{'-> ACEPTADO':>13}")
for f in factores:
    ps, nd, dsc, acp = [], 0, 0, 0
    for p in muestra_r:
        im = ImageEnhance.Brightness(Image.open(p).convert('RGB')).enhance(f)
        c = mejor(det.detectar(jpg(im), 0.05, 0.45)[0])
        if c is None:
            continue
        nd += 1; ps.append(c.p_roto)
        if c.p_roto >= CONFIG.tau_roto: dsc += 1
        elif c.p_sano >= CONFIG.tau_sano: acp += 1
    etq = f'{f:.2f}' + (' (actual)' if f == 1.0 else '')
    print(f'{etq:>10}{np.mean(ps):>15.3f}{nd:>12}{dsc:>13}{acp:>13}')

# --------------------------------------------------------------------------
# H4: cuanto margen hay en la confianza de los Sano correctos
# --------------------------------------------------------------------------
print('\n' + bar); print('H4  DISTRIBUCION DE p(Sano) EN HUEVOS REALMENTE SANOS'); print(bar)
conf_sanos = []
for p in sanos:
    c = mejor(det.detectar(Path(p).read_bytes(), 0.05, 0.45)[0], 'Sano')
    if c:
        conf_sanos.append(c.p_sano)
conf_sanos = np.array(sorted(conf_sanos))
print(f'  n={len(conf_sanos)}  min={conf_sanos.min():.3f}  p10={np.percentile(conf_sanos,10):.3f}  '
      f'mediana={np.median(conf_sanos):.3f}  max={conf_sanos.max():.3f}')
print(f"\n{'tau_sano':>10}{'Sanos que irian a REVISION':>30}{'% de sanos':>14}")
for t in [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    n_rev = int((conf_sanos < t).sum())
    marca = '  <- actual' if t == 0.60 else ''
    print(f'{t:>10.2f}{n_rev:>30}{100*n_rev/len(conf_sanos):>13.1f}%{marca}')
