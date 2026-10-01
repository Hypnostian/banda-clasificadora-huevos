"""Entrenamiento REAL en miniatura con ultralytics 8.4.157 (CPU) para validar la v3.

No mide calidad (son 48 imagenes a 160 px): valida la MECANICA que en Colab
costaria horas descubrir rota.

    <venv>/python v3/prueba_mini_entrenamiento.py
"""
import csv
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import v3_entrenamiento as V  # noqa: E402

fallos = 0


def check(cond, msg, extra=""):
    global fallos
    print(f"  {'[OK]  ' if cond else '[FALLO]'} {msg}{(' -> ' + str(extra)) if extra != '' else ''}", flush=True)
    fallos += 0 if cond else 1


def mini_dataset(dest: Path) -> Path:
    for s, n in (("train", 48), ("val", 16)):
        (dest / s / "images").mkdir(parents=True)
        (dest / s / "labels").mkdir(parents=True)
        origen = RAIZ / "huevos" / ("train" if s == "train" else "valid")
        for img in sorted((origen / "images").glob("*.jpg"))[:n]:
            shutil.copy2(img, dest / s / "images" / img.name)
            shutil.copy2(origen / "labels" / (img.stem + ".txt"), dest / s / "labels" / (img.stem + ".txt"))
    (dest / "data.yaml").write_text(f"path: {dest.as_posix()}\ntrain: train/images\nval: val/images\nnc: 2\nnames: ['Roto', 'Sano']\n")
    return dest / "data.yaml"


ARGS = dict(imgsz=160, batch=8, epochs=12, patience=0, close_mosaic=2, cos_lr=True, lrf=0.01,
            optimizer="AdamW", lr0=0.001, warmup_epochs=0, cls_pw=0.5, cache="disk", workers=0,
            degrees=10.0, shear=0.0, perspective=0.0, mixup=0.0, flipud=0.5, fliplr=0.5,
            plots=False, verbose=False, seed=0, exist_ok=True)
REGLA = dict(min_epocas=3, paciencia=1, mejora_minima=1.0, epocas_finales=2)   # meseta forzada: t=3, fin=5
LR_MIN = 0.001 * 0.01                                                          # lr0 * lrf


def leer_resultados(run: Path):
    with open(run / "results.csv", encoding="utf-8") as fh:
        return [{k.strip(): v for k, v in f.items()} for f in csv.DictReader(fh)]


def desconectar(data: str, proyecto: Path, nombre: str, corte: int) -> Path:
    """Entrena en un proceso aparte que muere (os._exit) al terminar la epoca `corte`."""
    r = subprocess.run([sys.executable, __file__, "--hijo", data, str(proyecto), nombre, str(corte)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    check(r.returncode == 3, f"el proceso hijo se 'desconecto' tras la epoca {corte}", f"codigo {r.returncode}")
    ultimo = proyecto / nombre / "weights" / "last.pt"
    check(ultimo.exists(), "quedo last.pt para reanudar")
    return ultimo


if __name__ == "__main__" and "--hijo" in sys.argv:
    from ultralytics import YOLO
    data, proyecto, nombre, corte = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    m = YOLO("yolov8n.pt")
    V.registrar(m, V.crear_parada_ordenada(**REGLA, log=lambda *_: None))
    m.add_callback("on_fit_epoch_end", lambda tr: os._exit(3) if tr.epoch + 1 >= corte else None)
    m.train(data=data, project=proyecto, name=nombre, augmentations=V.aumentaciones_fotometricas(), **ARGS)
    sys.exit(0)

if __name__ == "__main__":
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    tmp = Path(tempfile.mkdtemp(prefix="v3_mini_"))
    os.chdir(tmp)                  # ultralytics descarga yolov8n.pt en la carpeta actual: no ensuciar el proyecto
    data = str(mini_dataset(tmp / "ds"))
    log_ul = io.StringIO()
    LOGGER.addHandler(logging.StreamHandler(log_ul))

    print("=" * 74); print("1) ENTRENAMIENTO NORMAL con meseta forzada (debe cerrar en orden y terminar en la 5)"); print("=" * 74)
    mensajes = []
    m = YOLO("yolov8n.pt")
    V.registrar(m, V.crear_parada_ordenada(**REGLA, log=mensajes.append))
    m.train(data=data, project=str(tmp / "runs"), name="normal", augmentations=V.aumentaciones_fotometricas(), **ARGS)
    run = Path(m.trainer.save_dir)
    filas = leer_resultados(run)
    texto = log_ul.getvalue()
    check(len(filas) == 5, "termina en la epoca t+finales = 5 (tope era 12)", f"{len(filas)} epocas")
    check(any("FASE FINAL" in x for x in mensajes), "el callback anuncia la fase final", next((x for x in mensajes if "FASE" in x), ""))
    check("Closing dataloader mosaic" in texto, "ultralytics CIERRA el mosaico al entrar en la fase final")
    lr = [float(f["lr/pg0"]) for f in filas]
    check(lr[4] < lr[3] < lr[2] and abs(lr[4] - LR_MIN) < 1e-6, "la tasa baja hasta lr0*lrf en la fase final",
          " -> ".join(f"{x:.2e}" for x in lr))
    check("MotionBlur" in texto and "Defocus" in texto and "ImageCompression" in texto,
          "ultralytics acepta las aumentaciones fotometricas", "aparecen en el log de albumentations")
    check(any("mAP50-95" in k for k in filas[0]), "results.csv trae la columna mAP50-95 que usa la regla")
    check("Class weights" in texto, "cls_pw activo en deteccion (pesos por clase)",
          next((ln.strip() for ln in texto.splitlines() if "Class weights" in ln), ""))
    eta = next((x for x in mensajes if "min/epoca" in x and "termina entre la 5" in x and "y la 12" in x), "")
    check(bool(eta), "cada linea trae min/epoca y cuanto falta (minimo y tope)", eta.strip())
    check(any("fase final: epoca 4 de 5" in x for x in mensajes), "y durante la fase final, cuanto le queda",
          next((x.strip() for x in mensajes if "fase final:" in x), ""))

    print("=" * 74); print("2) DESCONEXION en la epoca 2 y REANUDACION (debe terminar tambien en la 5)"); print("=" * 74)
    ultimo = desconectar(data, tmp / "runs", "reanudar", 2)
    mensajes2 = []
    m2 = YOLO(str(ultimo))
    V.registrar(m2, V.crear_parada_ordenada(**REGLA, log=mensajes2.append))
    m2.train(resume=True)
    filas2 = leer_resultados(tmp / "runs" / "reanudar")
    check(len(filas2) == 5, "tras reanudar termina en la MISMA epoca (5)", f"{len(filas2)} epocas")
    check(any("FASE FINAL" in x for x in mensajes2), "la regla se recalcula desde results.csv al reanudar")

    print("=" * 74); print("3) DESCONEXION en plena FASE FINAL (epoca 4) y REANUDACION"); print("=" * 74)
    # Control: engancharla solo al final de cada epoca (como estaba antes) deja la epoca reanudada
    # con mosaico y con el LR original. Si el control no muestra el defecto, la prueba no sirve.
    ultimo = desconectar(data, tmp / "runs", "control", 4)
    pos = len(log_ul.getvalue())
    mc = YOLO(str(ultimo))
    mc.add_callback("on_fit_epoch_end", V.crear_parada_ordenada(**REGLA, log=lambda *_: None))
    mc.train(resume=True)
    fc, tc = leer_resultados(tmp / "runs" / "control"), log_ul.getvalue()[pos:]
    lr_c = float(fc[4]["lr/pg0"])
    check("Closing dataloader mosaic" not in tc and lr_c > 10 * LR_MIN,          # coseno original en x=4: 7.5e-4
          "CONTROL sin el enganche al inicio: la epoca 5 corre con mosaico y LR alto (el defecto existe)",
          f"lr {lr_c:.2e}")

    ultimo = desconectar(data, tmp / "runs", "reanudar_final", 4)
    pos = len(log_ul.getvalue())
    mensajes3 = []
    m3 = YOLO(str(ultimo))
    V.registrar(m3, V.crear_parada_ordenada(**REGLA, log=mensajes3.append))
    m3.train(resume=True)
    f3, t3 = leer_resultados(tmp / "runs" / "reanudar_final"), log_ul.getvalue()[pos:]
    lr_5 = float(f3[4]["lr/pg0"])
    check(len(f3) == 5, "con registrar() termina en la 5", f"{len(f3)} epocas")
    check("Closing dataloader mosaic" in t3, "la epoca reanudada arranca SIN mosaico")
    check(abs(lr_5 - LR_MIN) < 1e-6, "y con el LR de la fase final (lr0*lrf)", f"{lr_5:.2e}")
    check(any("FASE FINAL" in x for x in mensajes3[:1]), "lo anuncia al arrancar, antes de entrenar",
          mensajes3[0].strip() if mensajes3 else "")

    print("=" * 74); print("4) TOPE: una meseta en la recta final no alarga el entrenamiento"); print("=" * 74)

    def entrenador_falso(n_filas: int, tope: int):
        d = Path(tempfile.mkdtemp(dir=tmp))
        (d / "results.csv").write_text("epoch,metrics/mAP50-95(B),lr/pg0\n"
                                       + "".join(f"{i},0.5,0.001\n" for i in range(1, n_filas + 1)))
        return types.SimpleNamespace(csv=d / "results.csv", epochs=tope, stop=False, lf=None,
                                     args=types.SimpleNamespace(epochs=tope, lrf=0.01, cos_lr=True),
                                     scheduler=types.SimpleNamespace(lr_lambdas=[None] * 3))

    cerca = dict(min_epocas=9, paciencia=1, mejora_minima=1.0, epocas_finales=4)   # t=9 y 9+4 > tope 10
    tr, msj = entrenador_falso(9, 10), []
    V.crear_parada_ordenada(**cerca, log=msj.append)(tr)
    check(tr.epochs == 10 and tr.scheduler.lr_lambdas == [None] * 3 and not tr.stop,
          "meseta en la 9 con tope 10: no pasa del tope ni toca el LR (el cierre normal ya va)", msj[-1].strip())
    tr = entrenador_falso(10, 10)
    V.crear_parada_ordenada(**cerca, log=lambda *_: None)(tr)
    check(tr.stop and tr.epochs == 10, "y se detiene en el tope")

    print("=" * 74); print("5) EXPORT ONNX y compatibilidad con el backend"); print("=" * 74)
    onnx_path = m.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False, nms=False)
    import onnxruntime as ort
    s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = s.get_outputs()[0].shape
    meta = s.get_modelmeta().custom_metadata_map
    check(list(out) == [1, 6, 8400], "salida [1, 6, 8400]: la misma forma que espera el backend", out)
    check(meta.get("names") == "{0: 'Roto', 1: 'Sano'}", "clases en el orden del backend", meta.get("names"))
    sys.path.insert(0, str(RAIZ / "backend"))
    from app.detector import DetectorHuevosONNX
    det = DetectorHuevosONNX(Path(onnx_path), 640, 0)
    img = next((RAIZ / "huevos" / "test" / "images").glob("*.jpg")).read_bytes()
    cajas, lat, _ = det.detectar(img, 0.001, 0.45)
    check(isinstance(cajas, list), "el DetectorHuevosONNX del backend lo carga y detecta sin cambios",
          f"{len(cajas)} cajas, {lat['total_ms']:.0f} ms")

    os.chdir(RAIZ)
    shutil.rmtree(tmp, ignore_errors=True)
    print("=" * 74)
    print("RESULTADO:", "TODO OK" if fallos == 0 else f"{fallos} FALLO(S)")
    print("=" * 74)
    sys.exit(1 if fallos else 0)
