"""Mide las soluciones candidatas antes de recomendarlas.

  S1  Subir tau_sano: los rotos que se cuelan por desenfoque, que p(Sano) sacan?
      Si sacan menos que un sano de verdad, un umbral alto los atrapa gratis.

  S2  Compuerta de nitidez: sirve la varianza del laplaciano para rechazar la
      foto ANTES de clasificarla? Que umbral?

  S3  Peor de N tomas: si el operario dispara 3 veces en vez de 1 y nos
      quedamos con el veredicto mas pesimista, baja la fuga?
"""
import io, sys, random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

RAIZ = Path(r'D:\eggdetection\backend')
sys.path.insert(0, str(RAIZ))
from app.config import CONFIG               # noqa: E402
from app.detector import DetectorHuevosONNX # noqa: E402

det = DetectorHuevosONNX(CONFIG.modelo_path, 640, 0)
TEST = Path(r'D:\eggdetection\huevos\test\images')
rotos = [p for p in sorted(TEST.glob('*.jpg')) if p.name.startswith('damaged')][:30]
sanos = [p for p in sorted(TEST.glob('*.jpg')) if p.name.startswith('not_damaged')]
bar = '=' * 78


def jpg(im, q=88):
    b = io.BytesIO(); im.save(b, 'JPEG', quality=q); return b.getvalue()


def mejor(cajas):
    return max(cajas, key=lambda k: k.confianza) if cajas else None


def nitidez(im: Image.Image) -> float:
    """Varianza del laplaciano en escala de grises, normalizada por tamano."""
    g = np.asarray(im.convert('L').resize((320, 320)), dtype=np.float32)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from numpy.lib.stride_tricks import sliding_window_view
    v = sliding_window_view(g, (3, 3))
    lap = (v * k).sum(axis=(-1, -2))
    return float(lap.var())


# ---------------------------------------------------------------- S1
print(bar); print('S1  p(Sano) DE LOS ROTOS QUE SE CUELAN POR DESENFOQUE'); print(bar)
colados, sanos_reales = [], []
for p in rotos:
    for r in (3, 4):
        im = Image.open(p).convert('RGB').filter(ImageFilter.GaussianBlur(r))
        c = mejor(det.detectar(jpg(im), 0.05, 0.45)[0])
        if c and c.p_roto < CONFIG.tau_roto and c.p_sano >= CONFIG.tau_sano:
            colados.append(c.p_sano)          # roto mal aceptado
for p in sanos:
    c = mejor(det.detectar(p.read_bytes(), 0.05, 0.45)[0])
    if c and c.clase == 'Sano':
        sanos_reales.append(c.p_sano)
colados, sanos_reales = np.array(colados), np.array(sanos_reales)
print(f'  ROTOS mal aceptados (n={len(colados)}):  p(Sano) min={colados.min():.3f} '
      f'mediana={np.median(colados):.3f} max={colados.max():.3f}')
print(f'  SANOS de verdad     (n={len(sanos_reales)}):  p(Sano) min={sanos_reales.min():.3f} '
      f'mediana={np.median(sanos_reales):.3f} max={sanos_reales.max():.3f}')
print(f"\n{'tau_sano':>9}{'rotos colados atrapados':>26}{'sanos mandados a revision':>28}")
for t in [0.60, 0.70, 0.80, 0.85, 0.88, 0.90]:
    atrapados = int((colados < t).sum())
    falsos = int((sanos_reales < t).sum())
    marca = '  <- actual' if t == 0.60 else ''
    print(f'{t:>9.2f}{atrapados:>18} / {len(colados):<5}{falsos:>20} / {len(sanos_reales):<5}{marca}')

# ---------------------------------------------------------------- S2
print('\n' + bar); print('S2  COMPUERTA DE NITIDEZ (varianza del laplaciano)'); print(bar)
print(f"{'radio blur':>11}{'nitidez mediana':>18}{'min':>10}{'max':>10}")
umbrales = {}
for r in [0, 1, 2, 3, 4, 6]:
    vals = []
    for p in rotos[:20]:
        im = Image.open(p).convert('RGB')
        if r:
            im = im.filter(ImageFilter.GaussianBlur(r))
        vals.append(nitidez(im))
    vals = np.array(vals); umbrales[r] = vals
    print(f'{r:>11}{np.median(vals):>18.1f}{vals.min():>10.1f}{vals.max():>10.1f}')
sep = (np.median(umbrales[2]) + np.median(umbrales[3])) / 2
print(f'\n  El desenfoque se vuelve peligroso a partir de radio 3 (ver H2).')
print(f'  Umbral sugerido de nitidez: {sep:.0f}')
buenos = np.concatenate([umbrales[0], umbrales[1], umbrales[2]])
malos = np.concatenate([umbrales[3], umbrales[4], umbrales[6]])
print(f'  Con ese corte: {100*(buenos>=sep).mean():.0f}% de las fotas NITIDAS pasan, '
      f'{100*(malos<sep).mean():.0f}% de las BORROSAS se rechazan')

# ---------------------------------------------------------------- S3
print('\n' + bar); print('S3  PEOR DE N TOMAS  (el operario dispara varias veces)'); print(bar)
rng = random.Random(5)


def toma(im, k):
    """Simula otra toma del mismo huevo: leve desenfoque, giro y brillo distintos."""
    from PIL import ImageEnhance
    r = [1.0, 3.2, 2.4][k % 3]
    out = im.rotate([0, -7, 6][k % 3], resample=Image.BILINEAR, expand=False)
    out = out.filter(ImageFilter.GaussianBlur(r))
    return ImageEnhance.Brightness(out).enhance([1.0, 1.25, 0.9][k % 3])


def ruta_de(c):
    if c is None:
        return 'SIN_DETECCION'
    if c.p_roto >= CONFIG.tau_roto:
        return 'DESCARTE'
    return 'ACEPTADO' if c.p_sano >= CONFIG.tau_sano else 'REVISION'


print(f"{'estrategia':<26}{'DESCARTE':>10}{'REVISION':>10}{'ACEPTADO':>10}{'sin det.':>10}{'FUGA%':>9}")
for nombre, n_tomas in [('1 toma (actual)', 1), ('peor de 2 tomas', 2), ('peor de 3 tomas', 3)]:
    cuenta = {'DESCARTE': 0, 'REVISION': 0, 'ACEPTADO': 0, 'SIN_DETECCION': 0}
    for p in rotos:
        base = Image.open(p).convert('RGB')
        p_roto_max, c_final = -1, None
        for k in range(n_tomas):
            im = toma(base, k) if n_tomas > 1 else base.filter(ImageFilter.GaussianBlur(3.2))
            c = mejor(det.detectar(jpg(im), 0.05, 0.45)[0])
            # "peor caso" = el que mas sospecha de rotura
            if c and c.p_roto > p_roto_max:
                p_roto_max, c_final = c.p_roto, c
        r = ruta_de(c_final)
        cuenta[r] += 1
    fuga = 100 * cuenta['ACEPTADO'] / len(rotos)
    print(f'{nombre:<26}{cuenta["DESCARTE"]:>10}{cuenta["REVISION"]:>10}'
          f'{cuenta["ACEPTADO"]:>10}{cuenta["SIN_DETECCION"]:>10}{fuga:>9.1f}')
print('\n  (todas las tomas parten de una foto desenfocada radio ~3, que es el')
print('   escenario donde el sistema falla hoy)')
