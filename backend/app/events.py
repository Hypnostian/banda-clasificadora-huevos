"""Bus de eventos de la banda.

Dos implementaciones con la MISMA interfaz:

  * BusKafka    -> Kafka/Redpanda real (lo que se defiende ante el jurado)
  * BusMemoria  -> bus in-process con colas asyncio (plan B de la demo)

El resto del sistema no sabe cual esta usando. Si el broker no levanta en la
ventana de AWS, se cambia BUS_TIPO=memoria en el .env y la demo sigue viva con
exactamente la misma semantica de productores y consumidores.
"""
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger("banda.eventos")

Manejador = Callable[[dict[str, Any]], Awaitable[None]]


class BusEventos(ABC):
    @abstractmethod
    async def iniciar(self) -> None: ...

    @abstractmethod
    async def cerrar(self) -> None: ...

    @abstractmethod
    async def publicar(self, topico: str, clave: str, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    async def suscribir(self, topico: str, grupo: str, manejador: Manejador) -> None: ...

    @property
    @abstractmethod
    def descripcion(self) -> str: ...


# --------------------------------------------------------------------------- #
class BusMemoria(BusEventos):
    """Colas asyncio. Fan-out real: cada suscriptor recibe todos los mensajes."""

    def __init__(self) -> None:
        self._colas: dict[str, list[asyncio.Queue]] = {}
        self._tareas: list[asyncio.Task] = []
        self._activo = False

    async def iniciar(self) -> None:
        self._activo = True
        log.info("Bus en memoria iniciado")

    async def cerrar(self) -> None:
        self._activo = False
        for t in self._tareas:
            t.cancel()
        await asyncio.gather(*self._tareas, return_exceptions=True)
        self._tareas.clear()

    async def publicar(self, topico: str, clave: str, payload: dict[str, Any]) -> None:
        for cola in self._colas.get(topico, []):
            try:
                cola.put_nowait(payload)
            except asyncio.QueueFull:
                log.warning("Cola llena en %s, se descarta el evento mas viejo", topico)
                _ = cola.get_nowait()
                cola.put_nowait(payload)

    async def suscribir(self, topico: str, grupo: str, manejador: Manejador) -> None:
        cola: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._colas.setdefault(topico, []).append(cola)

        async def bucle() -> None:
            while True:
                payload = await cola.get()
                try:
                    await manejador(payload)
                except Exception:
                    log.exception("Fallo el consumidor %s en %s", grupo, topico)

        self._tareas.append(asyncio.create_task(bucle(), name=f"consumer:{grupo}:{topico}"))
        log.info("Suscrito %s -> %s (memoria)", grupo, topico)

    @property
    def descripcion(self) -> str:
        return "memoria (in-process)"


# --------------------------------------------------------------------------- #
class BusKafka(BusEventos):
    """Kafka/Redpanda via aiokafka."""

    def __init__(self, bootstrap: str) -> None:
        self.bootstrap = bootstrap
        self._productor = None
        self._consumidores: list[Any] = []
        self._tareas: list[asyncio.Task] = []

    async def iniciar(self) -> None:
        from aiokafka import AIOKafkaProducer

        self._productor = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode(),
            key_serializer=lambda k: k.encode() if k else None,
            linger_ms=5,
            acks=1,
        )
        await self._productor.start()
        log.info("Productor Kafka conectado a %s", self.bootstrap)

    async def cerrar(self) -> None:
        for t in self._tareas:
            t.cancel()
        await asyncio.gather(*self._tareas, return_exceptions=True)
        for c in self._consumidores:
            await c.stop()
        if self._productor:
            await self._productor.stop()

    async def publicar(self, topico: str, clave: str, payload: dict[str, Any]) -> None:
        assert self._productor is not None, "llama a iniciar() primero"
        await self._productor.send_and_wait(topico, payload, key=clave)

    async def suscribir(self, topico: str, grupo: str, manejador: Manejador) -> None:
        from aiokafka import AIOKafkaConsumer

        consumidor = AIOKafkaConsumer(
            topico,
            bootstrap_servers=self.bootstrap,
            group_id=grupo,
            value_deserializer=lambda v: json.loads(v.decode()),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        await consumidor.start()
        self._consumidores.append(consumidor)

        async def bucle() -> None:
            async for msg in consumidor:
                try:
                    await manejador(msg.value)
                except Exception:
                    log.exception("Fallo el consumidor %s en %s", grupo, topico)

        self._tareas.append(asyncio.create_task(bucle(), name=f"consumer:{grupo}:{topico}"))
        log.info("Suscrito %s -> %s (kafka)", grupo, topico)

    @property
    def descripcion(self) -> str:
        return f"kafka @ {self.bootstrap}"


async def construir_bus(tipo: str, bootstrap: str) -> BusEventos:
    """Crea el bus pedido y cae a memoria si Kafka no responde.

    Esto es deliberado: en una demo con ventana de 4 horas, que el broker no
    levante no puede tumbar el sistema completo.
    """
    if tipo == "kafka":
        bus: BusEventos = BusKafka(bootstrap)
        try:
            await bus.iniciar()
            return bus
        except Exception as exc:
            log.error("Kafka no disponible (%s). Cayendo a bus en memoria.", exc)
    bus = BusMemoria()
    await bus.iniciar()
    return bus
