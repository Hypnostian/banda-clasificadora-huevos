"""Configuracion central del clasificador de huevos.

Todo se puede sobrescribir por variables de entorno o por un archivo .env,
para no tener que tocar codigo durante la ventana de 4h de AWS Academy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ = Path(__file__).resolve().parent.parent


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ------------------------------------------------------------------ modelo
    modelo_path: Path = RAIZ / "modelos" / "best.onnx"
    # Nombre visible en /salud y en la app (vacio = nombre del archivo). Con dos
    # versiones desplegadas es la forma de saber cual esta respondiendo.
    modelo_nombre: str = ""
    modelo_imgsz: int = 640
    # 0 = que onnxruntime decida. En instancias de 2 vCPU dejar en 2.
    hilos_inferencia: int = 0

    # ------------------------------------------------- politica de decision
    # Umbral por debajo del cual consideramos que NO hay huevo en el frame.
    # Derivado de la tabla de costo de `scripts/evaluar_test.py`: el minimo
    # absoluto esta en 0.05 (costo 100) pero 0.15 cuesta casi lo mismo (106)
    # con la mitad de falsos positivos, y deja margen frente a objetos del
    # mundo real que el test sintetico de banda vacia no cubre (manos,
    # herramientas, etiquetas).
    conf_min_deteccion: float = 0.15
    # IoU para el NMS.
    iou_nms: float = 0.45

    # Politica ASIMETRICA de ruteo (ver docs/DECISIONES.md).
    # Un falso negativo (roto que se va al empaque) cuesta mucho mas que un
    # falso positivo (sano descartado), por eso el umbral para DESCARTAR es
    # bajo y el umbral para ACEPTAR es alto. La franja intermedia no se
    # decide sola: se manda a REVISION humana.
    tau_roto: float = 0.35     # p(Roto) >= tau_roto  -> DESCARTE
    # 0.85 y no 0.60: medido en scripts/diagnostico_campo.py. Un huevo SANO de
    # verdad nunca baja de p(Sano)=0.884; en cambio los rotos que se cuelan por
    # desenfoque caen entre 0.659 y 0.950. Cortar en 0.85 atrapa 10 de cada 25
    # de esos colados y manda a REVISION 0 de 22 sanos reales: es gratis.
    tau_sano: float = 0.85     # p(Sano) >= tau_sano  -> ACEPTADO
    # Banda de revision (None = apagada): tau_revision <= p(Roto) < tau_roto no se
    # bota ni se acepta, lo decide un humano. Medido con la v3 sobre el test
    # (docs/COMPARACION_V1_V3.md): con descarte desde 0.35 y banda desde 0.07 siguen
    # 0 rotos al empaque, igual que descartando desde 0.07, y los sanos botados bajan
    # de 4/43 a 1/43 (huevos marrones con brillo, fondos moteados). Si tau_revision
    # >= tau_roto no tiene efecto.
    tau_revision: float | None = None
    costo_falso_negativo: float = 12.0   # costo relativo de dejar pasar un roto
    costo_falso_positivo: float = 1.0    # costo relativo de botar un sano

    # --------------------------------------------------- compuerta de nitidez
    # Varianza del laplaciano del RECORTE del huevo, normalizado a 192x192.
    # Una foto desenfocada destruye la fisura y el modelo dice "Sano": con
    # blur gaussiano de radio 3 la fuga de rotos pasa de 0% a 38%. Con este
    # corte pasan el 100% de las fotos enfocadas y se rechaza el 93% de las
    # borrosas peligrosas (ver docs/ANALISIS_CAMPO.md).
    nitidez_minima: float = 4.0
    exigir_nitidez: bool = True

    # ------------------------------------------- consolidacion por huevo
    # En modo continuo la camara ve el MISMO huevo en varios frames. Tomar el
    # veredicto mas pesimista de todas las vistas baja la fuga de 36.7% a 0%
    # con solo 2 tomas: es la mejor mejora medida del sistema.
    consolidar_por_huevo: bool = True
    # Lo mismo DENTRO de un frame: una caja "Roto" y otra "Sano" sobre el mismo huevo
    # (IoU >= este valor) son un solo huevo y manda la mas sospechosa. 0 = apagado.
    fusion_iou_mismo_huevo: float = 0.6

    # ------------------------------------------------- simulacion de la banda
    # Velocidad nominal de la banda emulada (huevos/hora objetivo).
    banda_objetivo_huevos_hora: int = 9000
    # Ventana (segundos) para considerar que un bounding box es el MISMO huevo
    # que ya vimos en el frame anterior, y no contarlo dos veces.
    seguimiento_ttl_s: float = 1.5
    seguimiento_iou_min: float = 0.35

    # ------------------------------------------------------------- eventos
    # "kafka" usa Redpanda/Kafka real; "memoria" usa un bus in-process con la
    # misma interfaz (plan B si el broker no levanta en la demo).
    bus_tipo: Literal["kafka", "memoria"] = "memoria"
    kafka_bootstrap: str = "localhost:9092"
    topico_clasificados: str = "huevos.clasificados"
    topico_comandos: str = "banda.comandos"
    topico_curaduria: str = "huevos.curaduria"

    # Levantar los 3 consumers dentro del proceso de la API (recomendado para
    # la demo). Si se pone en False hay que correr los workers aparte.
    workers_embebidos: bool = True

    # ------------------------------------------------------------ actuador
    # "simulado" no necesita hardware; "serial" habla con el Arduino/ESP32.
    actuador_tipo: Literal["simulado", "serial"] = "simulado"
    serial_puerto: str = "COM3"
    serial_baudios: int = 115200
    serial_timeout_s: float = 0.5

    # ------------------------------------------------------------ almacenamiento
    dir_evidencias: Path = RAIZ / "evidencias"
    dir_curaduria: Path = RAIZ / "evidencias" / "curaduria"
    guardar_todas_las_evidencias: bool = False  # True = guarda cada frame (llena disco)

    # ------------------------------------------------------------------ api
    cors_origins: str = "*"
    max_bytes_imagen: int = 8 * 1024 * 1024


CONFIG = Config()

# Las rutas relativas del .env (p. ej. MODELO_PATH=modelos/best.onnx) se
# resuelven contra backend/, no contra el directorio desde el que se lanzo el
# proceso: asi `python workers/worker.py` y `uvicorn app.main:app` encuentran
# lo mismo sin importar desde donde se invoquen.
for _campo in ("modelo_path", "dir_evidencias", "dir_curaduria"):
    _valor = getattr(CONFIG, _campo)
    if not _valor.is_absolute():
        setattr(CONFIG, _campo, (RAIZ / _valor).resolve())

CONFIG.dir_evidencias.mkdir(parents=True, exist_ok=True)
CONFIG.dir_curaduria.mkdir(parents=True, exist_ok=True)

CLASES = ["Roto", "Sano"]
IDX_ROTO, IDX_SANO = 0, 1
