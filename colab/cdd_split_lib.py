# =============================================================================
#  Split limpio para datasets de CLASIFICACION exportados de Roboflow
#  (probado localmente con colab/test_cdd_split_lib.py antes de ir a Colab)
#
#  El problema que resuelve: un split aleatorio por imagen mete fuga cuando
#  hay (a) copias aumentadas del mismo original, (b) frames del mismo video,
#  (c) rafagas de fotos del mismo huevo o (d) duplicados subidos dos veces.
#  Si un frame cae en train y su vecino en test, el modelo "ya vio" ese huevo
#  y las metricas salen infladas: se ve genial y falla en campo.
#
#  Solucion: agrupar todo lo que es "el mismo huevo" y repartir GRUPOS, no
#  imagenes, estratificando por clase.
#
#  v2 — correccion tras el primer entrenamiento real:
#  El hash de 64 bits no separa "misma foto" de "huevo distinto con el mismo
#  fondo" (medido en 968 fotos reales: duplicados hasta 6 bits, el 5 % de los
#  huevos distintos a <=5 bits). Encadenado con union-find, eso formo UN grupo
#  con el 91.5 % del dataset y dejo val/test con 453 originales en vez de ~1.590.
#  Ahora: hash de 256 bits (duplicados p95=13, distintos p5=29), las uniones por
#  hash tienen TOPE de tamano, y revisar_reparto() se niega a seguir si un split
#  queda desbalanceado.
# =============================================================================
from __future__ import annotations

import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CARPETAS_SPLIT = {"train": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}


# ------------------------------------------------------------------ nombres
def nombre_base(nombre: str) -> str:
    """'20251002_174109_028_jpg.rf.ab12cd.jpg' -> '20251002_174109_028'.

    Roboflow renombra 'foto.jpg' a 'foto_jpg.rf.<hash>.jpg'. Todas las copias
    aumentadas de un mismo original comparten lo que va antes de '.rf.'.
    """
    base = nombre.split(".rf.")[0]
    base = re.sub(r"\.(jpe?g|png|bmp|webp)$", "", base, flags=re.I)
    base = re.sub(r"_(jpe?g|png|bmp|webp)$", "", base, flags=re.I)
    return base


RE_VIDEO = re.compile(r"^(\d{8}_\d{6})_(\d{1,5})$")                          # 20251002_174109_028
RE_FECHA = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(\d{2})_(\d{2})_(\d{2})(?:_(\d+))?$")  # 2023-04-09-20_19_15_972818
RE_FRAME = re.compile(r"^(.+?)[_\-\s]*frame[_\-\s]*\d+$", re.I)              # clip3_frame_0045


def clave_video(base: str) -> str | None:
    """Frames de un mismo video comparten prefijo: agruparlos es obligatorio."""
    if m := RE_VIDEO.match(base):
        return "video:" + m.group(1)
    if m := RE_FRAME.match(base):
        return "video:" + m.group(1).lower()
    return None


def fecha_de(base: str) -> datetime | None:
    if m := RE_FECHA.match(base):
        y, mo, d, h, mi, s, micro = m.groups()
        return datetime(int(y), int(mo), int(d), int(h), int(mi), int(s),
                        int((micro or "0")[:6].ljust(6, "0")))
    return None


# ------------------------------------------------------------ hash perceptual
def _dhash_de(im: Image.Image, lado: int) -> np.ndarray:
    a = np.asarray(im.resize((lado + 1, lado), Image.BILINEAR), dtype=np.int16)
    bits = np.packbits((a[:, 1:] > a[:, :-1]).flatten())          # lado*lado bits
    return bits.view(np.uint64).copy()                             # lado=16 -> 4 x uint64


def dhash_par(ruta: Path, lado: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """(dhash, dhash del espejo horizontal) con UNA sola decodificacion.

    lado=16 da 256 bits. Con 64 bits (lado=8) los duplicados verdaderos y los
    huevos distintos fotografiados sobre el mismo fondo se solapan: no existe
    un umbral que los separe (medido sobre fotos reales).
    """
    im = Image.open(ruta).convert("L")
    return _dhash_de(im, lado), _dhash_de(im.transpose(Image.FLIP_LEFT_RIGHT), lado)


def calcular_hashes(items: list[dict], lado: int = 16, log=print) -> None:
    log(f"  hash perceptual ({lado * lado} bits) de {len(items)} imagenes...")
    for it in items:
        it["h"], it["hf"] = dhash_par(it["ruta"], lado)


_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    """Bits en 1 por fila. x: (..., k) uint64 -> (...,)"""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).sum(axis=-1, dtype=np.int32)
    return _LUT[x.view(np.uint8)].sum(axis=-1, dtype=np.int32)


class UnionFind:
    """Union-find con tamanos (para el tope) y causas por grupo (para el diagnostico)."""

    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.tam = [1] * n
        self.causas: list[Counter] = [Counter() for _ in range(n)]

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def tam_de(self, x: int) -> int:
        return self.tam[self.find(x)]

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


# ------------------------------------------------------------------ inventario
def inventariar(raiz: Path) -> list[dict]:
    """Lee un export 'folder' de Roboflow: <raiz>/<split>/<clase>/<imagen>."""
    items = []
    for carpeta_split in sorted(p for p in raiz.iterdir() if p.is_dir()):
        split = CARPETAS_SPLIT.get(carpeta_split.name.lower())
        if split is None:
            continue
        for carpeta_clase in sorted(p for p in carpeta_split.iterdir() if p.is_dir()):
            for f in sorted(carpeta_clase.iterdir()):
                if f.suffix.lower() in EXT_IMG:
                    items.append({"ruta": f, "split_orig": split, "clase_orig": carpeta_clase.name,
                                  "base": nombre_base(f.name)})
    return items


def agrupar(items: list[dict], gap_rafaga_s: float = 20.0, duracion_max_rafaga_s: float = 120.0,
            umbral_hash: int = 16, max_grupo_hash: int = 30, usar_hash: bool = True, log=print) -> dict:
    """Asigna 'grupo' a cada item uniendo cuatro fuentes de 'es el mismo huevo'.

    Las tres primeras (nombre del original, video, rafaga) son evidencia fuerte y
    se aplican siempre. El hash visual es evidencia debil: se aplica de la pareja
    mas parecida a la menos, y NUNCA si la union produciria un grupo de mas de
    `max_grupo_hash` imagenes. Los duplicados reales vienen en parejas o grupos
    chicos; un grupo de miles solo puede salir de encadenar huevos distintos.

    Requiere calcular_hashes(items) antes si usar_hash=True.
    """
    n = len(items)
    uf = UnionFind(n)
    razones = Counter()

    def unir(i, j, causa):
        if uf.union(i, j, causa):
            razones[causa] += 1

    # 1) copias aumentadas del mismo original (mismo nombre antes de '.rf.')
    por_base = defaultdict(list)
    for i, it in enumerate(items):
        por_base[it["base"]].append(i)
    for idx in por_base.values():
        for j in idx[1:]:
            unir(idx[0], j, "copia aumentada")

    # 2) frames de un mismo video
    por_video = defaultdict(list)
    for i, it in enumerate(items):
        if k := clave_video(it["base"]):
            por_video[k].append(i)
    for idx in por_video.values():
        for j in idx[1:]:
            unir(idx[0], j, "frame del mismo video")

    # 3) rafagas: fotos con marca de tiempo a menos de gap_rafaga_s entre si.
    #    Con tope de duracion: sin el, alguien que fotografio cada 10 s durante
    #    una hora terminaria con un solo grupo gigante (encadenamiento).
    fechadas = sorted(((fecha_de(it["base"]), i) for i, it in enumerate(items) if fecha_de(it["base"])),
                      key=lambda t: t[0])
    inicio = fechadas[0][0] if fechadas else None
    for (t0, i0), (t1, i1) in zip(fechadas, fechadas[1:]):
        if (t1 - t0).total_seconds() <= gap_rafaga_s and (t1 - inicio).total_seconds() <= duracion_max_rafaga_s:
            unir(i0, i1, f"rafaga (<{gap_rafaga_s:g}s)")
        else:
            inicio = t1

    # 4) casi-duplicados visuales, de la pareja mas parecida a la menos, con tope
    rechazadas = 0
    if usar_hash and n > 1:
        h = np.stack([it["h"] for it in items])
        hf = np.stack([it["hf"] for it in items])
        pares = []
        for a in range(n - 1):
            d = np.minimum(_popcount(h[a] ^ h[a + 1:]), _popcount(h[a] ^ hf[a + 1:]))
            for b in np.nonzero(d <= umbral_hash)[0]:
                pares.append((int(d[b]), a, a + 1 + int(b)))
        pares.sort()
        for _, i, j in pares:
            if uf.find(i) == uf.find(j):
                continue
            if uf.tam_de(i) + uf.tam_de(j) > max_grupo_hash:
                rechazadas += 1
                continue
            unir(i, j, "casi-duplicado visual")

    for i, it in enumerate(items):
        it["grupo"] = uf.find(i)
    causas = {r: uf.causas[r] for r in {it["grupo"] for it in items}}
    return {"razones": razones, "n_grupos": len(causas), "rechazadas_por_tope": rechazadas,
            "causas_por_grupo": causas}


def grupos_mas_grandes(items: list[dict], info: dict, k: int = 5) -> list[dict]:
    """Para el diagnostico: los k grupos mas grandes, con su causa y ejemplos."""
    miembros = defaultdict(list)
    for it in items:
        miembros[it["grupo"]].append(it)
    top = sorted(miembros.items(), key=lambda kv: -len(kv[1]))[:k]
    return [{"tamano": len(m), "originales": len({it["base"] for it in m}),
             "causas": dict(info["causas_por_grupo"].get(g, {})),
             "ejemplos": list(dict.fromkeys(it["base"] for it in m))[:3]} for g, m in top]


def fuga_en_split_original(items: list[dict]) -> dict:
    """Cuantos grupos estan repartidos entre varios splits en el export original."""
    splits_por_grupo = defaultdict(set)
    for it in items:
        splits_por_grupo[it["grupo"]].add(it["split_orig"])
    cruzados = {g for g, s in splits_por_grupo.items() if len(s) > 1}
    imgs_eval_contaminadas = sum(1 for it in items
                                 if it["split_orig"] in ("val", "test") and it["grupo"] in cruzados)
    total_eval = sum(1 for it in items if it["split_orig"] in ("val", "test"))
    return {"grupos_cruzados": len(cruzados), "imgs_eval_contaminadas": imgs_eval_contaminadas,
            "total_eval": total_eval}


# ------------------------------------------------------------ split por grupos
def split_estratificado_por_grupos(items: list[dict], proporciones: dict[str, float],
                                   semilla: int = 42) -> dict[int, str]:
    """Reparte GRUPOS (no imagenes) respetando la proporcion de cada clase.

    Heuristica voraz: grupos grandes primero; cada grupo va al split que mas
    'deficit' relativo tiene para las clases que ese grupo contiene. Cuenta
    ORIGINALES, no archivos: las copias aumentadas no deben inflar un grupo.
    """
    comp = defaultdict(Counter)
    vistos = set()
    for it in items:
        if (it["grupo"], it["base"]) not in vistos:
            vistos.add((it["grupo"], it["base"]))
            comp[it["grupo"]][it["clase"]] += 1
    total = Counter()
    for c in comp.values():
        total.update(c)

    rng = random.Random(semilla)
    orden = list(comp)
    rng.shuffle(orden)
    orden.sort(key=lambda g: -sum(comp[g].values()))   # sort estable: los empates quedan barajados

    cuenta = {s: Counter() for s in proporciones}
    asignado = {}
    for g in orden:
        def deficit(s: str) -> float:
            return sum(n * (proporciones[s] - cuenta[s][c] / max(1, total[c])) for c, n in comp[g].items())
        s = max(proporciones, key=deficit)
        asignado[g] = s
        cuenta[s].update(comp[g])
    return asignado


def revisar_reparto(items: list[dict], asignado: dict[int, str], proporciones: dict[str, float],
                    info: dict, minimo_relativo: float = 0.6) -> dict[str, float]:
    """Se NIEGA a seguir si algun split quedo muy por debajo de su proporcion.

    Esto es lo que habria evitado el primer entrenamiento sobre un val de 227
    imagenes: un reparto desbalanceado no puede pasar en silencio.
    """
    por_split = defaultdict(set)
    for it in items:
        por_split[asignado[it["grupo"]]].add(it["base"])
    total = sum(len(v) for v in por_split.values())
    real = {s: len(por_split.get(s, ())) / max(1, total) for s in proporciones}
    malos = [s for s in proporciones if real[s] < minimo_relativo * proporciones[s]]
    if malos:
        top = grupos_mas_grandes(items, info)
        detalle = "\n".join(f"    {g['originales']:>6} originales  causas={g['causas']}  ej={g['ejemplos']}"
                            for g in top)
        raise RuntimeError(
            "El reparto quedo desbalanceado: "
            + ", ".join(f"{s}={100 * real[s]:.1f}% (objetivo {100 * proporciones[s]:.0f}%)" for s in proporciones)
            + "\n  Grupos mas grandes:\n" + detalle
            + "\n  Si domina 'casi-duplicado visual': baja umbral_hash o max_grupo_hash."
            + "\n  Si domina 'frame del mismo video': hay pocos videos muy largos; no se puede"
            + " separar sin fuga, reduce la proporcion de val/test o consigue mas videos."
        )
    return real


# --------------------------------------------------------------- materializar
def materializar(items: list[dict], asignado: dict[int, str], destino: Path,
                 sobremuestrear_hasta: float = 0.0, una_copia_en_train: bool = False,
                 semilla: int = 42, log=print) -> dict:
    """Copia a <destino>/<split>/<clase>/.

    - val/test: UNA imagen por original (las copias aumentadas no cuentan dos
      veces el mismo huevo), prefiriendo la que venia de valid/test (sin
      aumentacion).
    - train: todas las copias, o una por original si una_copia_en_train=True
      (con entrenamientos largos, las copias duplicadas hacen que cada "epoca"
      sean ~2 pasadas por cada huevo y aceleran la memorizacion).
    - sobremuestreo opcional de clases minoritarias, solo en train.
    """
    if destino.exists():
        shutil.rmtree(destino)
    por_split = defaultdict(list)
    for it in items:
        por_split[asignado[it["grupo"]]].append(it)

    final = {}
    for split, lista in por_split.items():
        if split in ("val", "test") or (split == "train" and una_copia_en_train):
            elegido = {}
            for it in sorted(lista, key=lambda x: (x["split_orig"] == "train", x["ruta"].name)):
                elegido.setdefault(it["base"], it)          # el primero gana: el no aumentado
            lista = list(elegido.values())
        final[split] = lista

    if sobremuestrear_hasta > 0 and "train" in final:
        rng = random.Random(semilla)
        cuenta = Counter(it["clase"] for it in final["train"])
        maximo = max(cuenta.values())
        extra = []
        for clase, n in sorted(cuenta.items()):
            meta = int(maximo * sobremuestrear_hasta)
            if n < meta:
                pool = [it for it in final["train"] if it["clase"] == clase]
                for k in range(meta - n):
                    it = dict(rng.choice(pool))
                    it["dup"] = k + 1
                    extra.append(it)
                log(f"  sobremuestreo train: {clase} {n} -> {meta}")
        final["train"] = final["train"] + extra

    usados = set()
    for split, lista in final.items():
        for it in lista:
            carpeta = destino / split / it["clase"]
            carpeta.mkdir(parents=True, exist_ok=True)
            nombre = it["ruta"].name
            if "dup" in it:
                nombre = f"{Path(nombre).stem}__dup{it['dup']}{Path(nombre).suffix}"
            clave = (split, it["clase"], nombre)
            if clave in usados:
                nombre = f"{it['split_orig']}__{nombre}"
                clave = (split, it["clase"], nombre)
            usados.add(clave)
            shutil.copy2(it["ruta"], carpeta / nombre)
    return final


def verificar_sin_fuga(final: dict, umbral_hash: int = 16, tolerancia_visual: float = 0.02,
                       log=print) -> bool:
    """Prueba independiente de fuga.

    - DURO: ningun original ni grupo puede estar en dos splits.
    - BLANDO: parejas visualmente casi identicas entre val/test y train. Algunas
      son esperables (las que el tope no dejo unir, casi siempre huevos distintos
      con el mismo fondo); solo es fallo si superan `tolerancia_visual` de val+test.
    """
    ok = True
    for campo in ("base", "grupo"):
        dueno = {}
        for split, lista in final.items():
            for it in lista:
                previo = dueno.setdefault(it[campo], split)
                if previo != split:
                    log(f"  FUGA por {campo}: {it[campo]} esta en {previo} y en {split}")
                    ok = False
    tr = final.get("train", [])
    ev = [it for s in ("val", "test") for it in final.get(s, [])]
    if tr and ev and "h" in tr[0]:
        htr = np.stack([it["h"] for it in tr])
        cerca = 0
        for it in ev:
            d = min(int(_popcount(it["h"] ^ htr).min()), int(_popcount(it["hf"] ^ htr).min()))
            cerca += d <= umbral_hash
        frac = cerca / len(ev)
        log(f"  parecidas a algo de train (<= {umbral_hash} bits): {cerca}/{len(ev)} = {100 * frac:.1f}%"
            f"  (tolerancia {100 * tolerancia_visual:.0f}%)")
        if frac > tolerancia_visual:
            log("  FUGA visual: demasiadas imagenes de val/test casi identicas a train")
            ok = False
    return ok


def ic95_proporcion(k: int, n: int) -> tuple[float, float]:
    """Intervalo de Wilson al 95% (funciona bien con proporciones pequenas)."""
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.96, k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    margen = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centro - margen), min(1.0, centro + margen))
