# =============================================================================
#  v3 — piezas de entrenamiento (probadas con un entrenamiento REAL en miniatura:
#  v3/prueba_mini_entrenamiento.py, ultralytics 8.4.157)
# =============================================================================
from __future__ import annotations

import csv
import math
import statistics
import time
from pathlib import Path


def aumentaciones_fotometricas() -> list:
    """Transformaciones de Albumentations para las fallas MEDIDAS en campo.

    Por defecto ultralytics aplica desenfoque con p=0.01 y brillo/gamma/JPEG con
    p=0. En campo el modelo 1 perdio el 50 % de los huevos con sobreexposicion y
    mando al empaque rotos desenfocados. Ninguna es geometrica: las cajas no se
    tocan (la geometria del huevo se cuida igual que en el modelo 1).
    """
    import albumentations as A
    return [
        A.MotionBlur(blur_limit=(3, 9), p=0.15),                              # celular movido
        A.Defocus(radius=(1, 4), alias_blur=(0.1, 0.3), p=0.10),              # foto desenfocada
        A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.35), contrast_limit=0.25, p=0.30),
        A.RandomGamma(gamma_limit=(70, 150), p=0.15),                         # sobre / sub exposicion
        A.ImageCompression(quality_range=(40, 95), p=0.20),                   # JPEG del celular
        A.GaussNoise(std_range=(0.02, 0.06), p=0.10),                         # poca luz = ruido
        A.CLAHE(p=0.05),
        A.ToGray(p=0.02),
    ]


def _lf_original(args) -> callable:
    """Replica el scheduler de ultralytics 8.4.157 (trainer._setup_scheduler)."""
    n, lrf = int(args.epochs), float(args.lrf)
    if args.cos_lr:
        return lambda x: max((1 - math.cos(x * math.pi / n)) / 2, 0) * (lrf - 1) + 1
    return lambda x: max(1 - x / n, 0) * (1.0 - lrf) + lrf


def crear_parada_ordenada(min_epocas: int, paciencia: int, mejora_minima: float, epocas_finales: int,
                          columna: str = "mAP50-95", log=print):
    """Regla de las sesiones anteriores, con un cierre ORDENADO en vez de un corte seco.

    - Nunca decide antes de `min_epocas`.
    - Despues, si pasan `paciencia` epocas sin una mejora de al menos `mejora_minima`
      en `columna` (mAP50-95: es el fitness de deteccion en 8.4.157), no corta: entra
      en una FASE FINAL de `epocas_finales` epocas sin mosaico y con la tasa de
      aprendizaje bajando a su minimo, y luego termina. Cortar en seco dejaria el LR
      alto y el mosaico abierto, que es justo donde se ganan los ultimos puntos.
    - Si nunca hay meseta, el entrenamiento llega al tope con el cierre normal.
    - Todo se recalcula desde results.csv: reanudar tras una desconexion da la misma
      decision. Engancharla con `registrar` (tambien al inicio): asi una reanudacion
      en plena fase final arranca ya sin mosaico y con el LR de la fase final.
    - El tope (`epochs` de train) se respeta: si la meseta llega en las ultimas
      `epocas_finales`, el cierre normal de ultralytics ya esta en marcha.
    - Cada linea trae el tiempo real por epoca y cuanto falta: ultralytics solo
      muestra el de la epoca en curso, y en Colab hay que planear las sesiones.

    Requiere entrenar con close_mosaic=epocas_finales y patience=0.
    """
    reloj = {"t": None, "dur": []}

    def restante(desde: int, hasta: int) -> str:
        if not reloj["dur"]:
            return ""
        s = statistics.median(reloj["dur"][-10:])
        return f" (~{max(hasta - desde, 0) * s / 3600:.1f} h)"

    def parada(trainer):
        ahora = time.time()
        if reloj["t"] is not None:
            reloj["dur"].append(ahora - reloj["t"])
        reloj["t"] = ahora
        ruta = Path(trainer.csv)
        if not ruta.exists():
            return
        with open(ruta, encoding="utf-8") as fh:
            filas = [{k.strip(): v for k, v in f.items()} for f in csv.DictReader(fh)]
        col = next((k for k in filas[0] if columna in k), None) if filas else None
        if col is None:
            return
        hist = [float(f[col]) for f in filas]
        ref, ref_ep, t = hist[0], 1, None
        for ep, v in enumerate(hist, 1):
            if v > ref + mejora_minima:
                ref, ref_ep = v, ep
            if t is None and ep >= min_epocas and ep - ref_ep >= paciencia:
                t = ep                                   # primera epoca en que se cumple la regla
        epoca, tope = len(hist), int(trainer.args.epochs)
        ritmo = f" | {statistics.median(reloj['dur'][-10:]) / 60:.1f} min/epoca" if reloj["dur"] else ""
        if t is None:
            extra = f" | minimo {min_epocas}" if epoca < min_epocas else ""
            fin_min = min(max(min_epocas, ref_ep + paciencia) + epocas_finales, tope)
            eta = (f": termina entre la {fin_min}{restante(epoca, fin_min)} y la {tope}{restante(epoca, tope)}"
                   if ritmo else "")
            log(f"  [parada] epoca {epoca}: {columna}={hist[-1]:.4f} | referencia {ref:.4f} (epoca {ref_ep})"
                f" | {epoca - ref_ep}/{paciencia} sin mejorar >= {mejora_minima:g}{extra}{ritmo}{eta}")
            return
        fin = min(t + epocas_finales, tope)
        if fin == tope and trainer.epochs == tope:
            log(f"  [parada] epoca {epoca}: meseta en la recta final; el cierre normal termina en la {tope}"
                f"{restante(epoca, tope)}.")
        elif trainer.epochs != fin:
            base = _lf_original(trainer.args)
            lrf, f0 = float(trainer.args.lrf), base(t - 1)

            def lf(x, t=t, f0=f0, base=base):
                if x < t:
                    return base(x)
                p = min(1.0, (x - t + 1) / epocas_finales)
                return lrf + (f0 - lrf) * 0.5 * (1 + math.cos(math.pi * p))

            trainer.lf = lf
            trainer.scheduler.lr_lambdas = [lf] * len(trainer.scheduler.lr_lambdas)
            trainer.epochs = fin
            log(f"  [parada] meseta desde la epoca {t} (ultima mejora significativa: {ref_ep}). "
                f"FASE FINAL de {epocas_finales} epocas: sin mosaico, LR -> minimo. "
                f"Termina en la {fin}{restante(epoca, fin)}.")
        else:
            log(f"  [parada] fase final: epoca {epoca} de {fin}{ritmo}{restante(epoca, fin)}")
        if epoca >= fin:
            trainer.stop = True

    return parada


def registrar(modelo, parada) -> None:
    """Engancha la regla al inicio del entrenamiento y al final de cada epoca.

    Al inicio importa al REANUDAR: ultralytics decide el mosaico y el LR al empezar
    cada epoca, asi que sin esto la primera epoca reanudada dentro de la fase final
    correria con mosaico y con el LR original.
    """
    modelo.add_callback("on_train_start", parada)
    modelo.add_callback("on_fit_epoch_end", parada)
