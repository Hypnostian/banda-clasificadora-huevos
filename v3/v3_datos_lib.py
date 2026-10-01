# =============================================================================
#  v3 — Curacion multi-fuente para DETECCION de huevos (Roto / Sano)
#  (probado localmente con v3/test_v3_datos_lib.py antes de ir a Colab)
#
#  Lo que hace, y por que:
#   1. Lee exports YOLO de Roboflow (deteccion o segmentacion -> caja).
#   2. AUDITA cada fuente y detecta su convencion de etiquetado:
#        "huevo"  -> la caja de roto rodea el huevo entero   (lo que usa el backend)
#        "region" -> la caja de roto rodea SOLO la grieta, dentro de una caja de huevo
#      Mezclar ambas sin convertir le ensena al modelo dos cosas incompatibles.
#   3. Mapea las clases de cada fuente a {Roto, Sano}. Una clase que es un huevo
#      pero no se puede mapear EXCLUYE la imagen: si solo se borrara la caja, ese
#      huevo quedaria como "fondo" y el modelo aprenderia que un huevo no es huevo.
#   4. Deduplica ENTRE fuentes (muchos proyectos de Roboflow son copias de otros),
#      con hash perceptual invariante a rotaciones de 90 grados y espejos.
#   5. Agrupa "el mismo huevo" (copias, videos, rafagas, casi-duplicados con tope)
#      y reparte GRUPOS estratificando por (fuente, tiene_roto).
#   6. Reserva el test del dataset original como BENCHMARK comun con el modelo 1,
#      junto con todo lo que se le parezca en otras fuentes: sin eso la comparacion
#      v3 vs modelo 1 estaria contaminada.
#   7. Negativos duros sinteticos (el diferencial): sellos de tinta, rayas de
#      lapicero y suciedad dibujados sobre huevos SANOS de train. El modelo 1
#      bota el 56 % de los huevos sanos con una raya de lapicero.
#   8. Color de cascara (blanco / marron) estimado del recorte, calibrado contra
#      las fuentes que traen el color en la etiqueta.
# =============================================================================
from __future__ import annotations

import math
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CARPETAS_SPLIT = {"train": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}
CLASES = ["Roto", "Sano"]               # MISMO orden que el backend: 0=Roto, 1=Sano
IGNORAR, EXCLUIR = "IGNORAR", "EXCLUIR"  # no es huevo -> se borra la caja | es huevo sin mapeo -> fuera la imagen


def norm(nombre: str) -> str:
    """'Cracked_Egg-Brown ' -> 'cracked egg brown': los nombres de clase de Roboflow varian."""
    return re.sub(r"[\s_\-]+", " ", str(nombre).strip().lower())


def normalizar_mapa(mapa: dict | None) -> dict:
    return {norm(k): v for k, v in (mapa or {}).items()}


# ------------------------------------------------------------------ nombres
def nombre_base(nombre: str) -> str:
    """'foto_jpg.rf.ab12.jpg' -> 'foto'. Las copias aumentadas comparten la base."""
    base = nombre.split(".rf.")[0]
    base = re.sub(r"\.(jpe?g|png|bmp|webp)$", "", base, flags=re.I)
    return re.sub(r"_(jpe?g|png|bmp|webp)$", "", base, flags=re.I)


RE_VIDEO = re.compile(r"^(\d{8}_\d{6})_(\d{1,5})$")
RE_FECHA = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(\d{2})_(\d{2})_(\d{2})(?:_(\d+))?$")
RE_FRAME = re.compile(r"^(.+?)[_\-\s]*frame[_\-\s]*\d+$", re.I)


def clave_video(base: str) -> str | None:
    if m := RE_VIDEO.match(base):
        return "video:" + m.group(1)
    if m := RE_FRAME.match(base):
        return "video:" + m.group(1).lower()
    return None


def fecha_de(base: str) -> datetime | None:
    if m := RE_FECHA.match(base):
        y, mo, d, h, mi, s, micro = m.groups()
        return datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), int((micro or "0")[:6].ljust(6, "0")))
    return None


# ------------------------------------------------------------ lectura YOLO
def leer_nombres(data_yaml: Path) -> list[str]:
    """names de un data.yaml de Roboflow (lista o dict), sin depender de PyYAML."""
    txt = data_yaml.read_text(encoding="utf-8", errors="ignore")
    try:
        import yaml
        d = yaml.safe_load(txt)
        n = d.get("names")
        return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
    except Exception:
        m = re.search(r"names:\s*\[(.*?)\]", txt, re.S)
        if m:
            return [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
        return re.findall(r"^\s*\d+:\s*['\"]?(.+?)['\"]?\s*$", txt.split("names:", 1)[-1], re.M)


def leer_etiquetas(ruta_txt: Path, nombres: list[str]) -> list[tuple[str, float, float, float, float]]:
    """Cajas normalizadas (clase, cx, cy, w, h). Poligonos de segmentacion -> caja envolvente."""
    cajas = []
    if not ruta_txt.exists():
        return cajas
    for ln in ruta_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        if c >= len(nombres):
            continue
        v = [float(x) for x in p[1:]]
        if len(v) == 4:
            cx, cy, w, h = v
        else:                                      # poligono x1 y1 x2 y2 ...
            xs, ys = v[0::2], v[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        if w > 0 and h > 0:
            cajas.append((nombres[c], cx, cy, w, h))
    return cajas


def inventariar(raiz: Path, fuente: str) -> list[dict]:
    """Export YOLO de Roboflow: <raiz>/<split>/images + labels + data.yaml."""
    yml = next(iter(sorted(raiz.rglob("data.yaml"))), None)
    if yml is None:
        raise FileNotFoundError(f"{fuente}: no hay data.yaml en {raiz}")
    nombres = leer_nombres(yml)
    items = []
    for carpeta in sorted(p for p in yml.parent.iterdir() if p.is_dir()):
        split = CARPETAS_SPLIT.get(carpeta.name.lower())
        if split is None or not (carpeta / "images").is_dir():
            continue
        for img in sorted((carpeta / "images").iterdir()):
            if img.suffix.lower() in EXT_IMG:
                items.append({"ruta": img, "fuente": fuente, "split_orig": split,
                              "base": f"{fuente}:{nombre_base(img.name)}", "base_corta": nombre_base(img.name),
                              "cajas_src": leer_etiquetas(carpeta / "labels" / (img.stem + ".txt"), nombres)})
    return items


# ------------------------------------------------------------------ auditoria
def _xyxy(c):
    _, cx, cy, w, h = c
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _ioa(a, b) -> float:
    """Fraccion del area de a que cae dentro de b."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))


def auditar(items: list[dict], mapa: dict[str, str]) -> dict:
    """Estadisticas por clase y convencion detectada ('huevo' o 'region')."""
    por_clase = defaultdict(list)
    for it in items:
        for c in it["cajas_src"]:
            por_clase[c[0]].append(c[3] * c[4])
    resumen = {k: {"cajas": len(v), "area_mediana": float(np.median(v))} for k, v in sorted(por_clase.items())}

    mapa = normalizar_mapa(mapa)
    roto_src = {k for k, v in mapa.items() if v == "Roto"}
    sano_src = {k for k, v in mapa.items() if v == "Sano"}
    ioas, razones = [], []
    for it in items:
        sanos = [_xyxy(c) for c in it["cajas_src"] if norm(c[0]) in sano_src]
        if not sanos:
            continue
        for c in it["cajas_src"]:
            if norm(c[0]) in roto_src:
                a = _xyxy(c)
                j = max(range(len(sanos)), key=lambda k: _ioa(a, sanos[k]))
                ioas.append(_ioa(a, sanos[j]))
                s = sanos[j]
                razones.append((c[3] * c[4]) / max(1e-9, (s[2] - s[0]) * (s[3] - s[1])))
    region = bool(ioas) and np.median(ioas) >= 0.8 and np.median(razones) <= 0.5
    return {"clases": resumen, "convencion": "region" if region else "huevo",
            "ioa_roto_en_sano": float(np.median(ioas)) if ioas else None,
            "area_roto_sobre_sano": float(np.median(razones)) if razones else None,
            "sin_clase_sano": not sano_src, "imagenes": len(items),
            "originales": len({it["base"] for it in items})}


def mapear(items: list[dict], mapa: dict[str, str], convencion: str, color_por_clase: dict | None = None) -> dict:
    """Convierte cajas de la fuente a (Roto|Sano). Devuelve contadores de lo descartado."""
    stats = Counter()
    mapa, color_por_clase = normalizar_mapa(mapa), normalizar_mapa(color_por_clase)
    for it in items:
        cajas, excluir, sin_mapa = [], False, set()
        crudas = [(norm(c[0]),) + tuple(c[1:]) for c in it["cajas_src"]]
        if convencion == "region":
            huevos = [c for c in crudas if mapa.get(c[0]) == "Sano"]
            grietas = [c for c in crudas if mapa.get(c[0]) == "Roto"]
            for g in grietas:
                if not any(_ioa(_xyxy(g), _xyxy(h)) >= 0.5 for h in huevos):
                    excluir = True                 # grieta sin huevo que la contenga: ambiguo
            for h in huevos:
                roto = any(_ioa(_xyxy(g), _xyxy(h)) >= 0.5 for g in grietas)
                cajas.append(("Roto" if roto else "Sano", h[1], h[2], h[3], h[4], (color_por_clase or {}).get(h[0])))
            crudas = [c for c in crudas if mapa.get(c[0]) not in ("Roto", "Sano")]
        for c in crudas:
            destino = mapa.get(c[0], EXCLUIR)
            if c[0] not in mapa:
                sin_mapa.add(c[0])
            if destino == EXCLUIR:
                excluir = True
            elif destino in ("Roto", "Sano"):
                cajas.append((destino, c[1], c[2], c[3], c[4], (color_por_clase or {}).get(c[0])))
        it["cajas"], it["excluir"] = cajas, excluir
        stats["imagenes_excluidas"] += excluir
        stats["imagenes_sin_cajas"] += (not excluir and not cajas)
        for k in sin_mapa:
            stats[f"clase sin mapeo: {k}"] += 1
    return dict(stats)


# ------------------------------------------------------------ hash perceptual
def _dhash_de(im: Image.Image, lado: int = 16) -> np.ndarray:
    a = np.asarray(im.resize((lado + 1, lado), Image.BILINEAR), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten()).view(np.uint64).copy()


def hashes_variantes(ruta: Path) -> np.ndarray:
    """8 variantes (4 rotaciones x espejo): (8, 4) uint64. Una sola decodificacion."""
    im = Image.open(ruta).convert("L")
    out = []
    for espejo in (False, True):
        base = im.transpose(Image.FLIP_LEFT_RIGHT) if espejo else im
        for rot in (None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270):
            out.append(_dhash_de(base.transpose(rot) if rot is not None else base))
    return np.stack(out)


def calcular_hashes(items: list[dict], log=print) -> None:
    log(f"  hash perceptual (256 bits x 8 variantes) de {len(items)} imagenes...")
    for it in items:
        it["hv"] = hashes_variantes(it["ruta"])


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).sum(axis=-1, dtype=np.int32)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return lut[x.view(np.uint8)].sum(axis=-1, dtype=np.int32)


def pares_cercanos(items: list[dict], umbral: int) -> list[tuple[int, int, int]]:
    """(distancia, i, j) con distancia invariante a rotacion/espejo <= umbral."""
    H0 = np.stack([it["hv"][0] for it in items])            # (n, 4) variante canonica
    V = np.stack([it["hv"] for it in items])                 # (n, 8, 4)
    pares = []
    for a in range(len(items) - 1):
        d = _popcount(H0[a][None, None, :] ^ V[a + 1:]).min(axis=1)   # vs las 8 variantes del resto
        for b in np.nonzero(d <= umbral)[0]:
            pares.append((int(d[b]), a, a + 1 + int(b)))
    pares.sort()
    return pares


def deduplicar_entre_fuentes(items: list[dict], prioridad: list[str], umbral: int = 4, log=print) -> list[dict]:
    """La MISMA foto en dos fuentes -> se queda la de mayor prioridad (primera en la lista)."""
    rango = {f: i for i, f in enumerate(prioridad)}
    fuera = set()
    for d, i, j in pares_cercanos(items, umbral):
        a, b = items[i], items[j]
        if a["fuente"] == b["fuente"] or i in fuera or j in fuera:
            continue
        perdedor = j if rango.get(a["fuente"], 99) <= rango.get(b["fuente"], 99) else i
        fuera.add(perdedor)
    quitados = Counter(items[k]["fuente"] for k in fuera)
    log(f"  duplicados entre fuentes eliminados: {len(fuera)}  {dict(quitados)}")
    return [it for k, it in enumerate(items) if k not in fuera]


# ------------------------------------------------------------------ grupos
class UnionFind:
    def __init__(self, n: int) -> None:
        self.p, self.tam = list(range(n)), [1] * n
        self.causas: list[Counter] = [Counter() for _ in range(n)]

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int, causa: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.tam[ra] < self.tam[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.tam[ra] += self.tam[rb]
        self.causas[ra] += self.causas[rb]
        self.causas[ra][causa] += 1
        return True


def agrupar(items: list[dict], gap_rafaga_s: float = 20.0, duracion_max_rafaga_s: float = 120.0,
            umbral_hash: int = 16, max_grupo_hash: int = 30) -> dict:
    """'Mismo huevo': copia aumentada, video, rafaga (por fuente) y casi-duplicado visual con tope."""
    n = len(items)
    uf, razones = UnionFind(n), Counter()

    def unir(i, j, causa):
        if uf.union(i, j, causa):
            razones[causa] += 1

    for clave, regla in ((lambda it: it["base"], "copia aumentada"),
                         (lambda it: f"{it['fuente']}|{clave_video(it['base_corta'])}"
                          if clave_video(it["base_corta"]) else None, "frame del mismo video")):
        idx = defaultdict(list)
        for i, it in enumerate(items):
            k = clave(it)
            if k:
                idx[k].append(i)
        for grupo in idx.values():
            for j in grupo[1:]:
                unir(grupo[0], j, regla)

    por_fuente = defaultdict(list)
    for i, it in enumerate(items):
        t = fecha_de(it["base_corta"])
        if t:
            por_fuente[it["fuente"]].append((t, i))
    for lista in por_fuente.values():
        lista.sort()
        inicio = lista[0][0]
        for (t0, i0), (t1, i1) in zip(lista, lista[1:]):
            if (t1 - t0).total_seconds() <= gap_rafaga_s and (t1 - inicio).total_seconds() <= duracion_max_rafaga_s:
                unir(i0, i1, "rafaga")
            else:
                inicio = t1

    rechazadas = 0
    for _, i, j in pares_cercanos(items, umbral_hash):
        if uf.find(i) == uf.find(j):
            continue
        if uf.tam[uf.find(i)] + uf.tam[uf.find(j)] > max_grupo_hash:
            rechazadas += 1
            continue
        unir(i, j, "casi-duplicado visual")

    for i, it in enumerate(items):
        it["grupo"] = uf.find(i)
    return {"razones": razones, "n_grupos": len({it["grupo"] for it in items}),
            "rechazadas_por_tope": rechazadas,
            "causas_por_grupo": {r: uf.causas[r] for r in {it["grupo"] for it in items}}}


# ------------------------------------------------------------------ split
def clave_estrato(it: dict) -> str:
    return f"{it['fuente']}|{'con_roto' if any(c[0] == 'Roto' for c in it['cajas']) else 'solo_sano'}"


def repartir(items: list[dict], proporciones: dict[str, float], benchmark: callable | None = None,
             semilla: int = 42) -> dict[int, str]:
    """Grupos -> split. Todo grupo que toque el benchmark va entero al benchmark."""
    grupos_bench = {it["grupo"] for it in items if benchmark and benchmark(it)}
    comp = defaultdict(Counter)
    vistos = set()
    for it in items:
        if it["grupo"] in grupos_bench or (it["grupo"], it["base"]) in vistos:
            continue
        vistos.add((it["grupo"], it["base"]))
        comp[it["grupo"]][clave_estrato(it)] += 1
    total = Counter()
    for c in comp.values():
        total.update(c)
    rng = random.Random(semilla)
    orden = list(comp)
    rng.shuffle(orden)
    orden.sort(key=lambda g: -sum(comp[g].values()))
    cuenta = {s: Counter() for s in proporciones}
    asignado = {g: "benchmark" for g in grupos_bench}
    for g in orden:
        s = max(proporciones, key=lambda s: sum(n * (proporciones[s] - cuenta[s][k] / max(1, total[k]))
                                                 for k, n in comp[g].items()))
        asignado[g] = s
        cuenta[s].update(comp[g])
    return asignado


def revisar_reparto(items: list[dict], asignado: dict[int, str], proporciones: dict[str, float],
                    minimo_relativo: float = 0.6) -> dict[str, float]:
    """Se niega a seguir si un split quedo muy por debajo de su proporcion."""
    por = defaultdict(set)
    for it in items:
        if asignado[it["grupo"]] != "benchmark":
            por[asignado[it["grupo"]]].add(it["base"])
    tot = sum(len(v) for v in por.values())
    real = {s: len(por.get(s, ())) / max(1, tot) for s in proporciones}
    malos = [s for s in proporciones if real[s] < minimo_relativo * proporciones[s]]
    if malos:
        raise RuntimeError("Reparto desbalanceado: " + ", ".join(
            f"{s}={100 * real[s]:.1f}% (objetivo {100 * proporciones[s]:.0f}%)" for s in proporciones))
    return real


def escribir_etiqueta(ruta: Path, cajas: list) -> None:
    ruta.write_text("".join(f"{CLASES.index(c[0])} {c[1]:.6f} {c[2]:.6f} {c[3]:.6f} {c[4]:.6f}\n" for c in cajas))


def materializar(items: list[dict], asignado: dict[int, str], destino: Path) -> dict[str, list[dict]]:
    """UNA imagen por original en todos los splits (prefiere la que venia de valid/test, sin
    aumentacion horneada): Ultralytics aumenta en linea, distinto en cada epoca."""
    if destino.exists():
        shutil.rmtree(destino)
    por_split = defaultdict(dict)
    for it in sorted(items, key=lambda x: (x["split_orig"] == "train", x["ruta"].name)):
        por_split[asignado[it["grupo"]]].setdefault(it["base"], it)
    final = {}
    for split, d in por_split.items():
        (destino / split / "images").mkdir(parents=True, exist_ok=True)
        (destino / split / "labels").mkdir(parents=True, exist_ok=True)
        final[split] = []
        for it in d.values():
            nombre = f"{it['fuente']}__{it['ruta'].stem}"[:180]
            dst = destino / split / "images" / (nombre + it["ruta"].suffix.lower())
            shutil.copy2(it["ruta"], dst)
            escribir_etiqueta(destino / split / "labels" / (nombre + ".txt"), it["cajas"])
            final[split].append({**it, "ruta_final": dst})
    return final


def escribir_data_yaml(destino: Path, splits: list[str]) -> Path:
    y = destino / "data.yaml"
    lineas = [f"path: {destino.as_posix()}", "train: train/images", "val: val/images"]
    for s in splits:
        if s not in ("train", "val"):
            lineas.append(f"{s}: {s}/images")
    lineas += ["nc: 2", "names: ['Roto', 'Sano']"]
    y.write_text("\n".join(lineas) + "\n")
    return y


def verificar_sin_fuga(final: dict[str, list[dict]], umbral_hash: int = 16, tolerancia: float = 0.02,
                       bloquear_visual: bool = True, log=print) -> bool:
    """DURO: ningun original ni grupo en dos splits (siempre bloquea).
    VISUAL: imagenes de val/test/benchmark casi identicas a algo de train. Con fuentes de estudio muy
    uniformes, huevos DISTINTOS en el mismo montaje pueden parecerse: con bloquear_visual=False se
    reporta como advertencia en vez de detener todo."""
    ok = True
    for campo in ("base", "grupo"):
        dueno = {}
        for s, lista in final.items():
            for it in lista:
                previo = dueno.setdefault(it[campo], s)
                if previo != s:
                    ok = False
                    log(f"  FUGA por {campo}: {it[campo]} en {previo} y {s}")
    tr = final.get("train", [])
    if tr:
        Vtr = np.stack([it["hv"] for it in tr])
        for s in ("val", "test", "benchmark"):
            ev = final.get(s, [])
            if not ev:
                continue
            cerca = sum(int(_popcount(it["hv"][0][None, None, :] ^ Vtr).min()) <= umbral_hash for it in ev)
            frac = cerca / len(ev)
            aviso = "" if frac <= tolerancia else ("  <- SUPERA la tolerancia" if bloquear_visual
                                                   else "  <- ADVERTENCIA: escenas parecidas entre splits; el test puede ser algo optimista")
            log(f"  {s:<9} parecidas a algo de train: {cerca}/{len(ev)} = {100 * frac:.1f}% (tolerancia {100 * tolerancia:.0f}%){aviso}")
            if frac > tolerancia and bloquear_visual:
                ok = False
    return ok


# --------------------------------------------------- negativos duros sinteticos
TINTAS = {"roja": (190, 30, 40), "azul": (35, 60, 170), "verde": (30, 120, 60),
          "morada": (110, 40, 140), "negra": (25, 25, 30)}
_FUENTES_TTF = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"]


def _fuente(tam: int):
    for f in _FUENTES_TTF:
        if Path(f).exists():
            return ImageFont.truetype(f, tam)
    try:
        return ImageFont.load_default(size=tam)
    except TypeError:
        return ImageFont.load_default()


def _texto_sello(rng: random.Random) -> str:
    return rng.choice([
        f"L{rng.randint(1000, 9999)}", f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}",
        f"0CO{rng.randint(1000, 99999)}", f"{rng.randint(0, 3)}ES{rng.randint(10000, 99999)}",
        "A", "AA", f"CAD {rng.randint(1, 28):02d}{rng.choice(['OCT', 'NOV', 'DIC', 'ENE'])}"])


def marcar_huevo(im: Image.Image, caja: tuple, rng: random.Random, estilo: str) -> Image.Image:
    """Dibuja una marca SUPERFICIAL sobre el huevo de `caja` (normalizada). El huevo sigue sano."""
    W, H = im.size
    _, cx, cy, w, h = caja[:5]
    ex, ey, rx, ry = cx * W, cy * H, w * W / 2 * 0.62, h * H / 2 * 0.62      # dentro de la elipse
    if rx < 6 or ry < 6:
        return im
    capa = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)
    alfa = rng.randint(140, 225)
    if estilo == "sello":
        tinta = TINTAS[rng.choice(["roja", "azul", "verde", "morada", "roja", "azul"])]
        fnt = _fuente(max(9, int(ry * rng.uniform(0.28, 0.42))))
        txt = _texto_sello(rng)
        caja_txt = Image.new("RGBA", (int(rx * 2.4), int(ry * 1.2)), (0, 0, 0, 0))
        dt = ImageDraw.Draw(caja_txt)
        dt.text((caja_txt.width / 2, caja_txt.height / 2), txt, fill=tinta + (alfa,), font=fnt, anchor="mm")
        if rng.random() < 0.35:
            dt.ellipse([2, 2, caja_txt.width - 3, caja_txt.height - 3], outline=tinta + (alfa,), width=2)
        caja_txt = caja_txt.rotate(rng.uniform(-30, 30), resample=Image.BICUBIC, expand=True)
        capa.alpha_composite(caja_txt, (int(ex - caja_txt.width / 2), int(ey - caja_txt.height / 2)))
    elif estilo in ("lapicero_color", "lapicero_oscuro"):
        tinta = TINTAS[rng.choice(["azul", "roja", "verde"])] if estilo == "lapicero_color" else TINTAS["negra"]
        for _ in range(rng.choice([1, 1, 2])):
            t0, largo = rng.uniform(0, 2 * math.pi), rng.uniform(0.6, 1.6)
            pts = []
            for k in range(16):
                t = t0 + largo * k / 15
                r = rng.uniform(0.35, 0.9)
                pts.append((ex + rx * r * math.cos(t) + rng.uniform(-2, 2), ey + ry * r * math.sin(t * 1.3) + rng.uniform(-2, 2)))
            d.line(pts, fill=tinta + (alfa,), width=max(2, int(min(rx, ry) * 0.05)), joint="curve")
    else:                                                   # suciedad
        for _ in range(rng.randint(3, 9)):
            px, py = ex + rng.uniform(-rx, rx) * 0.9, ey + rng.uniform(-ry, ry) * 0.9
            s = max(2, int(min(rx, ry) * rng.uniform(0.04, 0.14)))
            tono = (rng.randint(80, 125), rng.randint(60, 90), rng.randint(35, 60), rng.randint(110, 200))
            d.ellipse([px - s, py - s * rng.uniform(0.6, 1.4), px + s, py + s * rng.uniform(0.6, 1.4)], fill=tono)
    capa = capa.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    out = im.convert("RGBA")
    out.alpha_composite(capa)
    return out.convert("RGB")


def generar_negativos_duros(final_train: list[dict], destino: Path, fraccion: float = 0.35,
                            pesos: dict[str, float] | None = None, semilla: int = 7, log=print) -> Counter:
    """Copias MARCADAS de imagenes de train con huevos sanos. Solo en train. Etiqueta intacta."""
    pesos = pesos or {"sello": 0.40, "lapicero_color": 0.25, "lapicero_oscuro": 0.20, "suciedad": 0.15}
    estilos, w = zip(*pesos.items())
    rng = random.Random(semilla)
    hechos = Counter()
    for it in final_train:
        sanos = [c for c in it["cajas"] if c[0] == "Sano"]
        if not sanos or rng.random() > fraccion:
            continue
        im = Image.open(it["ruta_final"]).convert("RGB")
        usados = []
        for c in rng.sample(sanos, k=min(len(sanos), rng.choice([1, 1, 2]))):
            e = rng.choices(estilos, weights=w)[0]
            im = marcar_huevo(im, c, rng, e)
            usados.append(e)
        nombre = it["ruta_final"].stem + "__neg"
        im.save(destino / "train" / "images" / (nombre + ".jpg"), quality=92)
        escribir_etiqueta(destino / "train" / "labels" / (nombre + ".txt"), it["cajas"])
        hechos.update(usados)
        hechos["imagenes"] += 1
    log(f"  negativos duros: {hechos['imagenes']} imagenes nuevas | marcas: "
        + ", ".join(f"{k}={v}" for k, v in hechos.items() if k != "imagenes"))
    return hechos


# ------------------------------------------------------------ color de cascara
def rasgos_color(im: Image.Image, caja: tuple) -> tuple[float, float]:
    """(saturacion mediana, brillo mediano) del centro del huevo, en [0,1]."""
    W, H = im.size
    _, cx, cy, w, h = caja[:5]
    x1, y1 = int((cx - w * 0.28) * W), int((cy - h * 0.28) * H)
    x2, y2 = int((cx + w * 0.28) * W), int((cy + h * 0.28) * H)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0, 0.0
    hsv = np.asarray(im.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2))).convert("HSV"), dtype=np.float32) / 255.0
    return float(np.median(hsv[..., 1])), float(np.median(hsv[..., 2]))


def calibrar_color(muestras: list[tuple[float, float, str]], semilla: int = 0) -> dict:
    """Umbral de saturacion blanco/marron, ajustado en la mitad de las muestras con color
    real y medido en la otra mitad (sin trampa)."""
    rng = random.Random(semilla)
    m = list(muestras)
    rng.shuffle(m)
    mitad = len(m) // 2
    ajuste, prueba = m[:mitad], m[mitad:]

    def exactitud(lista, u):
        if not lista:
            return float("nan")
        ok_b = [s < u for s, _, c in lista if c == "blanco"]
        ok_m = [s >= u for s, _, c in lista if c == "marron"]
        partes = [np.mean(x) for x in (ok_b, ok_m) if x]
        return float(np.mean(partes))            # exactitud balanceada

    candidatos = np.linspace(0.05, 0.6, 111)
    u = float(max(candidatos, key=lambda t: exactitud(ajuste, t))) if ajuste else 0.2
    return {"umbral_saturacion": u, "exactitud_ajuste": exactitud(ajuste, u),
            "exactitud_prueba": exactitud(prueba, u), "n_ajuste": len(ajuste), "n_prueba": len(prueba)}


def ic95(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.96, k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - m), min(1.0, c + m))
