"""Actuador de la banda: el servo que desvia el huevo roto.

Dos implementaciones intercambiables. La simulada es la que se usa por defecto
porque la demo NO puede depender de que el Arduino este conectado; la serial es
la que se enchufa cuando hay hardware. Ambas emiten el mismo log y la misma
confirmacion, asi que el dashboard se ve identico.

Protocolo serial (texto, terminado en \\n):
    ->  D:<egg_id>\\n     desviar (servo a 90 grados durante PULSO_MS)
    ->  P:<egg_id>\\n     pasar (no hace nada, solo confirma)
    ->  M:<egg_id>\\n     marcar para revision (enciende LED ambar)
    <-  OK:<egg_id>\\n    confirmacion del microcontrolador
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

from .config import Config
from .schemas import Ruta

log = logging.getLogger("banda.actuador")

_ACCION_POR_RUTA = {Ruta.DESCARTE: "D", Ruta.ACEPTADO: "P", Ruta.REVISION: "M"}


class Actuador(ABC):
    @abstractmethod
    async def accionar(self, egg_id: str, ruta: Ruta) -> str: ...

    @abstractmethod
    def cerrar(self) -> None: ...

    @property
    @abstractmethod
    def conectado(self) -> bool: ...

    @property
    @abstractmethod
    def descripcion(self) -> str: ...


class ActuadorSimulado(Actuador):
    """Emula el servo. Registra la orden y respeta el tiempo de actuacion real
    (~180 ms de ida y vuelta de un SG90) para que el throughput medido en la
    simulacion sea honesto."""

    TIEMPO_DESVIO_S = 0.18

    def __init__(self) -> None:
        self.ordenes: int = 0

    async def accionar(self, egg_id: str, ruta: Ruta) -> str:
        accion = _ACCION_POR_RUTA[ruta]
        self.ordenes += 1
        if ruta is Ruta.DESCARTE:
            await asyncio.sleep(self.TIEMPO_DESVIO_S)
        log.info("[SIM] %s:%s -> servo %s", accion, egg_id, ruta.value)
        return f"{accion}:{egg_id}"

    def cerrar(self) -> None:
        pass

    @property
    def conectado(self) -> bool:
        return True

    @property
    def descripcion(self) -> str:
        return "simulado (sin hardware)"


class ActuadorSerial(Actuador):
    """Habla con el Arduino/ESP32 por puerto serial.

    La escritura serial es bloqueante, asi que se manda a un hilo para no
    frenar el event loop de FastAPI.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._puerto = None
        self._lock = asyncio.Lock()
        try:
            import serial  # pyserial

            self._puerto = serial.Serial(
                cfg.serial_puerto, cfg.serial_baudios, timeout=cfg.serial_timeout_s
            )
            log.info("Serial abierto en %s @ %d", cfg.serial_puerto, cfg.serial_baudios)
        except Exception as exc:
            log.error("No pude abrir %s (%s). El actuador quedara inactivo.", cfg.serial_puerto, exc)

    async def accionar(self, egg_id: str, ruta: Ruta) -> str:
        accion = _ACCION_POR_RUTA[ruta]
        trama = f"{accion}:{egg_id}\n"
        if self._puerto is None:
            log.warning("Serial no disponible, se descarta la orden %s", trama.strip())
            return f"ERROR:{egg_id}"
        async with self._lock:
            return await asyncio.to_thread(self._escribir, trama, egg_id)

    def _escribir(self, trama: str, egg_id: str) -> str:
        try:
            self._puerto.reset_input_buffer()
            self._puerto.write(trama.encode())
            self._puerto.flush()
            respuesta = self._puerto.readline().decode(errors="ignore").strip()
            return respuesta or f"SIN_RESPUESTA:{egg_id}"
        except Exception as exc:
            log.exception("Fallo escribiendo al serial: %s", exc)
            return f"ERROR:{egg_id}"

    def cerrar(self) -> None:
        if self._puerto is not None:
            try:
                self._puerto.close()
            except Exception:
                pass

    @property
    def conectado(self) -> bool:
        return self._puerto is not None and getattr(self._puerto, "is_open", False)

    @property
    def descripcion(self) -> str:
        return f"serial {self.cfg.serial_puerto}@{self.cfg.serial_baudios}"


def construir_actuador(cfg: Config) -> Actuador:
    return ActuadorSerial(cfg) if cfg.actuador_tipo == "serial" else ActuadorSimulado()
