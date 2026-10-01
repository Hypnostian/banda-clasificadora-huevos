"""Prueba la biblioteca de split sobre un dataset SINTETICO con fugas plantadas.

Cada mecanismo de fuga se planta de forma aislada, para demostrar que cada
regla funciona por si sola:

  copia aumentada   -> mismo nombre antes de '.rf.', contenido volteado
  frame de video    -> contenido DISTINTO, solo el nombre los relaciona
  rafaga            -> contenido DISTINTO, solo la marca de tiempo
  casi-duplicado    -> nombres sin relacion, solo el contenido
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
import cdd_split_lib as L  # noqa: E402

rng = random.Random(0)
fallos = 0


def check(cond, msg, extra=""):
    global fallos
    print(f"  {'[OK]  ' if cond else '[FALLO]'} {msg}{(' -> ' + str(extra)) if extra != '' else ''}")
    if not cond:
        fallos += 1


def imagen_aleatoria(semilla: int) -> Image.Image:
    r = random.Random(semilla)
    im = Image.new("RGB", (224, 224), tuple(r.randint(0, 255) for _ in range(3)))
    d = ImageDraw.Draw(im)
    for _ in range(7):
        x0, y0 = r.randint(-40, 200), r.randint(-40, 200)
        d.ellipse([x0, y0, x0 + r.randint(30, 140), y0 + r.randint(30, 140)],
                  fill=tuple(r.randint(0, 255) for _ in range(3)))
    return im


def rf(nombre: str) -> str:
    """Imita el renombrado de Roboflow: foto.jpg -> foto_jpg.rf.<hash>.jpg"""
    return f"{nombre}_jpg.rf.{rng.getrandbits(64):016x}.jpg"


tmp = Path(tempfile.mkdtemp(prefix="cdd_test_"))
raiz = tmp / "export"
semilla = 1000


def guardar(split: str, clase: str, nombre: str, im: Image.Image, q: int = 90):
    p = raiz / split / clase
    p.mkdir(parents=True, exist_ok=True)
    im.save(p / nombre, "JPEG", quality=q)


# ---------------------------------------------------------------- dataset base
# 300 originales unicos: good 60% / broken 30% / dirty 10%, split original 80/10/10.
# Los de train van DUPLICADOS por aumentacion (como la v2 real: x2).
clases = ["good"] * 180 + ["broken"] * 90 + ["dirty"] * 30
rng.shuffle(clases)
for i, clase in enumerate(clases):
    im = imagen_aleatoria(semilla + i)
    split = "train" if i < 240 else ("valid" if i < 270 else "test")
    if split == "train":
        guardar(split, clase, rf(f"img_{i:04d}"), im)
        guardar(split, clase, rf(f"img_{i:04d}"), im.transpose(Image.FLIP_LEFT_RIGHT))
    else:
        guardar(split, clase, rf(f"img_{i:04d}"), im)

# ---------------------------------------------------------------- fugas plantadas
# (1) video: 6 frames del mismo video, contenido distinto, repartidos en train y test
for k, (split, f) in enumerate([("train", 3), ("train", 17), ("test", 7), ("test", 28),
                                ("valid", 55), ("train", 98)]):
    guardar(split, "broken", rf(f"20251002_174109_{f:03d}"), imagen_aleatoria(50000 + k))

# (2) rafaga: 3 fotos a 7 s entre si, contenido distinto, en splits distintos
for k, (split, t) in enumerate([("train", "20_19_15_972818"), ("test", "20_19_22_111111"),
                                ("valid", "20_19_29_500000")]):
    guardar(split, "good", rf(f"2023-04-09-{t}"), imagen_aleatoria(60000 + k))

# (3) cadena larga: una foto cada 15 s durante 5 min -> NO debe volverse un solo grupo
for k in range(20):
    s = 15 * k
    ts = f"2023-05-01-10_{(s // 60):02d}_{(s % 60):02d}_000000"
    guardar("train", "good", rf(ts), imagen_aleatoria(70000 + k))

# (4) casi-duplicado: MISMO contenido, nombres sin relacion, train vs test
dup = imagen_aleatoria(80000)
guardar("train", "dirty", rf("subida_original"), dup, q=92)
guardar("test", "dirty", rf("otra_cosa_distinta"), dup, q=70)       # recomprimido

# (5) duplicado volteado subido como si fuera otro
volt = imagen_aleatoria(80001)
guardar("train", "good", rf("foto_a"), volt)
guardar("valid", "good", rf("foto_b"), volt.transpose(Image.FLIP_LEFT_RIGHT))

print("=" * 72)
print("PRUEBA DE LA BIBLIOTECA DE SPLIT (dataset sintetico con fugas plantadas)")
print("=" * 72)

items = L.inventariar(raiz)
print(f"  inventario: {len(items)} archivos")
mapeo = {"good": "Sano", "broken": "Roto", "dirty": "Sucio"}
for it in items:
    it["clase"] = mapeo[it["clase_orig"]]

L.calcular_hashes(items, log=lambda *_: None)
info = L.agrupar(items, log=lambda *_: None)
print(f"  grupos: {info['n_grupos']} | razones de union: {dict(info['razones'])}")

grupo_de = lambda base: {it["grupo"] for it in items if it["base"] == base}  # noqa: E731
un_grupo = lambda bases: len(set().union(*(grupo_de(b) for b in bases))) == 1  # noqa: E731

check(all(len(grupo_de(f"img_{i:04d}")) == 1 for i in range(240)),
      "copias aumentadas del mismo original quedan en un grupo")
check(un_grupo([f"20251002_174109_{f:03d}" for f in (3, 17, 7, 28, 55, 98)]),
      "los 6 frames del video quedan en un solo grupo (aunque se vean distintos)")
check(un_grupo(["2023-04-09-20_19_15_972818", "2023-04-09-20_19_22_111111",
                "2023-04-09-20_19_29_500000"]),
      "la rafaga de 3 fotos a 7 s queda en un grupo")
grupos_cadena = set().union(*(grupo_de(f"2023-05-01-10_{(15*k)//60:02d}_{(15*k)%60:02d}_000000")
                              for k in range(20)))
check(len(grupos_cadena) >= 3, "la cadena de 5 min NO colapsa en un grupo gigante",
      f"{len(grupos_cadena)} grupos")
check(un_grupo(["subida_original", "otra_cosa_distinta"]),
      "casi-duplicado recomprimido con otro nombre -> mismo grupo")
check(un_grupo(["foto_a", "foto_b"]), "duplicado volteado con otro nombre -> mismo grupo")

originales_distintos = {it["base"] for it in items}
check(info["n_grupos"] < len(originales_distintos), "hubo uniones",
      f"{len(originales_distintos)} originales -> {info['n_grupos']} grupos")

fuga = L.fuga_en_split_original(items)
check(fuga["grupos_cruzados"] >= 4, "detecta la fuga del split ORIGINAL",
      f"{fuga['grupos_cruzados']} grupos cruzados, {fuga['imgs_eval_contaminadas']}/"
      f"{fuga['total_eval']} imgs de val/test contaminadas")

# ------------------------------------------------------------------ split nuevo
props = {"train": 0.70, "val": 0.15, "test": 0.15}
asignado = L.split_estratificado_por_grupos(items, props, semilla=42)
destino = tmp / "limpio"
final = L.materializar(items, asignado, destino, sobremuestrear_hasta=0.6, log=lambda *_: None)

check(L.verificar_sin_fuga(final, log=print), "verificacion independiente: CERO fuga")

bases_por_split = {s: {it["base"] for it in lst} for s, lst in final.items()}
check(all(len(bases_por_split[a] & bases_por_split[b]) == 0
          for a in bases_por_split for b in bases_por_split if a < b),
      "ningun original aparece en dos splits")

for s in ("val", "test"):
    bases = [it["base"] for it in final[s]]
    check(len(bases) == len(set(bases)), f"{s}: una sola imagen por original (sin copias aumentadas)")

# proporciones por clase (sin contar duplicados de sobremuestreo)
unicos = {s: Counter(it["clase"] for it in {it["base"]: it for it in lst}.values())
          for s, lst in final.items()}
total = Counter()
for c in unicos.values():
    total.update(c)
print(f"\n  {'clase':<7}" + "".join(f"{s:>10}" for s in props))
peor_desvio = 0.0
for clase in sorted(total):
    fila = []
    for s in props:
        frac = unicos[s][clase] / total[clase]
        peor_desvio = max(peor_desvio, abs(frac - props[s]))
        fila.append(f"{100*frac:>9.1f}%")
    print(f"  {clase:<7}" + "".join(fila))
check(peor_desvio < 0.06, "estratificacion: cada clase queda cerca de 70/15/15",
      f"desvio maximo {100*peor_desvio:.1f} pp")

n_train = Counter(it["clase"] for it in final["train"])
check(n_train["Sucio"] >= 0.6 * max(n_train.values()) - 1, "sobremuestreo sube la clase minoritaria",
      dict(n_train))
check(all("dup" not in it for s in ("val", "test") for it in final[s]),
      "el sobremuestreo NO toca val ni test")

archivos = {s: sum(1 for _ in (destino / s).rglob("*.jpg")) for s in props}
check(archivos == {s: len(final[s]) for s in props}, "los archivos en disco coinciden", archivos)

# Reproducibilidad: reanudar un entrenamiento exige que, al volver a correr todo,
# el split salga IDENTICO (mismo archivo en el mismo split, mismo sobremuestreo).
def huella(dest):
    return sorted(str(f.relative_to(dest)) for f in dest.rglob("*.jpg"))

items2 = L.inventariar(raiz)
for it in items2:
    it["clase"] = mapeo[it["clase_orig"]]
L.calcular_hashes(items2, log=lambda *_: None)
L.agrupar(items2, log=lambda *_: None)
asignado2 = L.split_estratificado_por_grupos(items2, props, semilla=42)
destino2 = tmp / "limpio_otra_vez"
L.materializar(items2, asignado2, destino2, sobremuestrear_hasta=0.6, log=lambda *_: None)
check(huella(destino) == huella(destino2), "reproducible: correr dos veces da el mismo split exacto",
      f"{len(huella(destino))} archivos")

lo, hi = L.ic95_proporcion(3, 60)
check(0.0 < lo < 0.05 < hi < 0.15, "IC95 de Wilson razonable para 3/60", f"[{lo:.3f}, {hi:.3f}]")

# =============================================================================
print("\n" + "=" * 72)
print("REGRESION: el bug del primer entrenamiento real (val=227, test=226)")
print("=" * 72)
# Camara fija: huevos DISTINTOS fotografiados uno a uno sobre la misma superficie.
# Con el hash de 64 bits eso se encadenaba en un solo grupo gigante.
raiz_fija = tmp / "camara_fija"
fondo = Image.new("RGB", (224, 224), (168, 160, 150))
dfondo = ImageDraw.Draw(fondo)
for y in range(0, 224, 4):
    dfondo.line([(0, y), (224, y)], fill=(160 + (y % 12), 152 + (y % 9), 143), width=2)


def huevo_camara_fija(semilla: int, clase: str, jitter: int = 10, tonos=None) -> Image.Image:
    r = random.Random(semilla)
    im = fondo.copy()
    d = ImageDraw.Draw(im)
    cx, cy = 112 + r.randint(-jitter, jitter), 112 + r.randint(-jitter, jitter)
    ax, ay = r.randint(34, 44), r.randint(46, 56)
    tono = r.choice(tonos or [(205, 140, 90), (215, 160, 110), (235, 225, 210)])
    d.ellipse([cx - ax, cy - ay, cx + ax, cy + ay], fill=tono)
    d.ellipse([cx - ax // 2, cy - ay + 6, cx, cy - ay // 3], fill=tuple(min(255, c + 25) for c in tono))
    if clase == "broken":
        d.line([(cx + r.randint(-9, 9), cy - ay + 8), (cx + r.randint(-12, 12), cy),
                (cx + r.randint(-9, 9), cy + ay - 10)], fill=(60, 40, 30), width=2)
    elif clase == "dirty":
        for _ in range(r.randint(2, 5)):
            x, y = cx + r.randint(-ax + 8, ax - 8), cy + r.randint(-ay + 8, ay - 8)
            d.ellipse([x - 4, y - 3, x + 4, y + 3], fill=(95, 75, 55))
    ruido = np.random.default_rng(semilla).normal(0, 4, (224, 224, 3))
    return Image.fromarray(np.clip(np.asarray(im, dtype=np.float32) + ruido, 0, 255).astype(np.uint8))


def generar_sesion(raiz, n, jitter, tonos=None):
    cls = ["good"] * int(n * 0.58) + ["broken"] * int(n * 0.29)
    cls += ["dirty"] * (n - len(cls))
    rng.shuffle(cls)
    for i, c in enumerate(cls):
        p = raiz / ("train" if i < 0.8 * n else "valid" if i < 0.9 * n else "test") / c
        p.mkdir(parents=True, exist_ok=True)
        huevo_camara_fija(i, c, jitter, tonos).save(p / rf(f"foto_{i:04d}"), "JPEG", quality=88)


def correr_config(raiz, lado, umbral, tope):
    its = L.inventariar(raiz)
    for it in its:
        it["clase"] = mapeo[it["clase_orig"]]
    L.calcular_hashes(its, lado=lado, log=lambda *_: None)
    inf = L.agrupar(its, umbral_hash=umbral, max_grupo_hash=tope, log=lambda *_: None)
    frac = max(Counter(it["grupo"] for it in its).values()) / len(its)
    asig = L.split_estratificado_por_grupos(its, props, semilla=42)
    try:
        return frac, L.revisar_reparto(its, asig, props, inf), None
    except RuntimeError as e:
        return frac, None, str(e)


VIEJA, NUEVA = dict(lado=8, umbral=6, tope=10**9), dict(lado=16, umbral=16, tope=30)

# A) camara fija con variacion normal: la config vieja ya junta huevos DISTINTOS en masa
generar_sesion(raiz_fija, 1200, jitter=10)
fv, _, _ = correr_config(raiz_fija, **VIEJA)
fn_, real_a, err_a = correr_config(raiz_fija, **NUEVA)
print(f"  A) camara fija, 1200 huevos distintos: grupo mas grande vieja={100*fv:.0f}%  nueva={100*fn_:.1f}%")
check(fv > 0.25, "REPRODUCE el mecanismo: la config vieja encadena huevos distintos", f"{100*fv:.0f}% en un grupo")
check(err_a is None and min(real_a["val"], real_a["test"]) >= 0.12, "la nueva deja val y test cerca del 15%",
      {k: f"{100*v:.1f}%" for k, v in (real_a or {}).items()} or err_a)

# B) montaje uniforme (captura automatica, mismo punto, mismo tipo de huevo): el caso real
raiz_rig = tmp / "montaje_uniforme"
generar_sesion(raiz_rig, 600, jitter=3, tonos=[(205, 140, 90)])
fv, _, err_v = correr_config(raiz_rig, **VIEJA)
fn_, real_b, err_b = correr_config(raiz_rig, **NUEVA)
print(f"  B) montaje uniforme, 600 huevos distintos: grupo mas grande vieja={100*fv:.0f}%  nueva={100*fn_:.1f}%")
# Aqui el sintetico llega a ~50%, no al 91.5% del run real: ese numero se DEDUCE del log
# (453 originales en val+test), no se reproduce. Lo demostrado es el mecanismo.
check(fv > 0.25, "REPRODUCE el mecanismo tambien en un montaje uniforme", f"{100*fv:.0f}% en un grupo")
check(err_b is None and min(real_b["val"], real_b["test"]) >= 0.12, "con la config nueva ese mismo caso reparte bien",
      {k: f"{100*v:.1f}%" for k, v in (real_b or {}).items()} or err_b)

# revisar_reparto, directo: si un grupo domina (el caso del run real), DEBE detener.
its = L.inventariar(raiz_rig)
for it in its:
    it["clase"] = mapeo[it["clase_orig"]]
L.calcular_hashes(its, log=lambda *_: None)
inf = L.agrupar(its, log=lambda *_: None)
dominante = its[0]["grupo"]
for it in its[: int(0.915 * len(its))]:          # el 91.5% del run real en un solo grupo
    it["grupo"] = dominante
inf["causas_por_grupo"][dominante] = Counter({"casi-duplicado visual": 540})
try:
    L.revisar_reparto(its, L.split_estratificado_por_grupos(its, props), props, inf)
    check(False, "con un grupo del 91.5% revisar_reparto debe detener")
except RuntimeError as e:
    check("casi-duplicado visual" in str(e), "con el 91.5% en un grupo DETIENE y senala al hash como causa",
          str(e).splitlines()[0][:74] + "...")

# video legitimo gigante: el reparto no puede cumplirse sin fuga -> debe avisar, no inventar
raiz_video = tmp / "video_gigante"
for k in range(300):
    p = raiz_video / "train" / ("broken" if k % 3 == 0 else "good")
    p.mkdir(parents=True, exist_ok=True)
    imagen_aleatoria(90000 + k).save(p / rf(f"20251002_174109_{k:03d}"), "JPEG")
for k in range(40):
    p = raiz_video / "test" / ("broken" if k % 3 == 0 else "good")
    p.mkdir(parents=True, exist_ok=True)
    imagen_aleatoria(95000 + k).save(p / rf(f"suelta_{k}"), "JPEG")
its = L.inventariar(raiz_video)
for it in its:
    it["clase"] = mapeo[it["clase_orig"]]
L.calcular_hashes(its, log=lambda *_: None)
inf = L.agrupar(its, log=lambda *_: None)
try:
    L.revisar_reparto(its, L.split_estratificado_por_grupos(its, props), props, inf)
    check(False, "un video con el 88% del dataset debe detener el reparto")
except RuntimeError as e:
    check("frame del mismo video" in str(e), "con un video gigante avisa y dice que regla lo formo",
          str(e).splitlines()[0][:70] + "...")

# fotos reales: 968 huevos distintos del primer dataset (si estan en el disco)
reales = Path(__file__).resolve().parents[1] / "huevos"
if reales.exists():
    its = [{"ruta": f, "split_orig": "train", "clase_orig": "x", "clase": "x", "base": L.nombre_base(f.name)}
           for f in sorted(reales.rglob("images/*.jpg"))]
    L.calcular_hashes(its, log=lambda *_: None)
    inf = L.agrupar(its, log=lambda *_: None)
    tam_r = max(Counter(it["grupo"] for it in its).values())
    check(tam_r <= 30, f"fotos reales ({len(its)} huevos distintos): ningun grupo gigante",
          f"mas grande={tam_r}, uniones por hash={inf['razones'].get('casi-duplicado visual', 0)}, "
          f"rechazadas por tope={inf['rechazadas_por_tope']}")

shutil.rmtree(tmp, ignore_errors=True)
print("=" * 72)
print("RESULTADO:", "TODO OK" if fallos == 0 else f"{fallos} FALLO(S)")
print("=" * 72)
sys.exit(1 if fallos else 0)
