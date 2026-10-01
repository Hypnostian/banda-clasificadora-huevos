"""Los tres consumidores de la banda orientada a eventos.

    huevos.clasificados ---> ConsumidorActuador   -> banda.comandos -> servo
                        |
                        +--> ConsumidorMetricas   -> dashboard (SSE)
                        |
                        +--> ConsumidorCuraduria  -> dataset de reentrenamiento

Estan separados a proposito: el servo puede fallar y el dashboard sigue vivo;
el dashboard puede caerse y los huevos se siguen desviando. Ese desacople es
justamente lo que compra la arquitectura de eventos, y es lo que hay que
mostrarle al jurado (apagar un consumer en vivo y ver que el resto no se cae).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .actuator import Actuador
from .config import Config
from .metrics import AgregadorMetricas
from .schemas import ComandoBanda, Ruta

log = logging.getLogger("banda.consumidores")


class ConsumidorActuador:
    """Traduce la clasificacion en movimiento fisico y publica el comando emitido."""

    def __init__(self, cfg: Config, actuador: Actuador, publicar) -> None:
        self.cfg = cfg
        self.actuador = actuador
        self.publicar = publicar
        self.desviados = 0
        self.ultimo_comando: str | None = None

    async def __call__(self, evento: dict[str, Any]) -> None:
        ruta = Ruta(evento["ruta"])
        egg_id = evento["egg_id"]
        respuesta = await self.actuador.accionar(egg_id, ruta)
        self.ultimo_comando = respuesta
        if ruta is Ruta.DESCARTE:
            self.desviados += 1
        comando = ComandoBanda(
            egg_id=egg_id,
            accion={"DESCARTE": "DESVIAR", "ACEPTADO": "PASAR", "REVISION": "MARCAR"}[ruta.value],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        await self.publicar(self.cfg.topico_comandos, egg_id, comando.model_dump())


class ConsumidorMetricas:
    """Delgado a proposito: toda la logica esta en AgregadorMetricas."""

    def __init__(self, agregador: AgregadorMetricas) -> None:
        self.agregador = agregador

    async def __call__(self, evento: dict[str, Any]) -> None:
        await self.agregador.consumir(evento)


class ConsumidorCuraduria:
    """Cierra el ciclo de vida del modelo.

    Todo huevo que cayo en la franja de incertidumbre (ruta REVISION) se
    registra como candidato de etiquetado. Eso convierte la operacion de la
    banda en la fuente del proximo dataset: active learning barato y honesto,
    porque son exactamente los casos donde el modelo dijo "no se".
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.indice: Path = cfg.dir_curaduria / "candidatos.jsonl"
        self.indice.parent.mkdir(parents=True, exist_ok=True)
        self.registrados = 0

    async def __call__(self, evento: dict[str, Any]) -> None:
        if evento.get("ruta") != Ruta.REVISION.value:
            return
        fila = {
            "egg_id": evento["egg_id"],
            "frame_id": evento["frame_id"],
            "timestamp": evento["timestamp"],
            "clase_predicha": evento["clase"],
            "p_roto": evento["p_roto"],
            "p_sano": evento["p_sano"],
            "bbox": evento["bbox"],
            "motivo": evento.get("motivo", ""),
            "evidencia": f"{evento['frame_id']}.jpg",
            "etiqueta_humana": None,   # lo llena el operario al revisar
        }
        with self.indice.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(fila, ensure_ascii=False) + "\n")
        self.registrados += 1
        log.info("Candidato de reetiquetado %s (%d acumulados)", evento["egg_id"], self.registrados)
