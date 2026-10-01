"""Consumidores standalone (modo microservicios reales).

Por defecto el backend corre los tres consumidores embebidos (WORKERS_EMBEBIDOS=true),
que es lo practico para la demo. Este archivo es para la otra mitad del
argumento: mostrar que son procesos independientes que escalan y fallan por
separado.

    # terminal 1
    BUS_TIPO=kafka WORKERS_EMBEBIDOS=false uvicorn app.main:app --host 0.0.0.0 --port 8000
    # terminal 2, 3, 4
    python workers/worker.py --rol actuador
    python workers/worker.py --rol metricas
    python workers/worker.py --rol curaduria

Apagar el worker de actuador en vivo y ver que el dashboard sigue recibiendo
eventos es la demostracion mas corta de por que la arquitectura es de eventos
y no una cadena de llamadas sincronas.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.actuator import construir_actuador          # noqa: E402
from app.config import CONFIG                         # noqa: E402
from app.consumers import (                           # noqa: E402
    ConsumidorActuador,
    ConsumidorCuraduria,
    ConsumidorMetricas,
)
from app.events import construir_bus                  # noqa: E402
from app.metrics import AgregadorMetricas             # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s")
log = logging.getLogger("banda.worker")


async def main() -> None:
    p = argparse.ArgumentParser(description="Consumidor standalone de la banda")
    p.add_argument("--rol", required=True, choices=["actuador", "metricas", "curaduria"])
    p.add_argument("--grupo", default=None, help="group_id de Kafka (por defecto grupo-<rol>)")
    args = p.parse_args()
    grupo = args.grupo or f"grupo-{args.rol}"

    if CONFIG.bus_tipo != "kafka":
        log.warning(
            "BUS_TIPO=%s. Un worker aparte solo recibe eventos si el bus es Kafka; "
            "con el bus en memoria los consumidores viven dentro del proceso de la API.",
            CONFIG.bus_tipo,
        )

    bus = await construir_bus(CONFIG.bus_tipo, CONFIG.kafka_bootstrap)
    actuador = None

    if args.rol == "actuador":
        actuador = construir_actuador(CONFIG)
        manejador = ConsumidorActuador(CONFIG, actuador, bus.publicar)
    elif args.rol == "metricas":
        agregador = AgregadorMetricas(CONFIG.banda_objetivo_huevos_hora)
        base = ConsumidorMetricas(agregador)

        async def manejador(evento):  # type: ignore[misc]
            await base(evento)
            m = agregador.instantanea()
            log.info(
                "total=%d rotura=%.1f%% revision=%.1f%% throughput=%d/h",
                m["total_huevos"], m["tasa_rotura_pct"], m["tasa_revision_pct"],
                m["throughput_huevos_hora"],
            )
    else:
        manejador = ConsumidorCuraduria(CONFIG)

    await bus.suscribir(CONFIG.topico_clasificados, grupo, manejador)
    log.info("Worker '%s' escuchando %s en %s", args.rol, CONFIG.topico_clasificados, bus.descripcion)

    parar = asyncio.Event()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(s, parar.set)
        except NotImplementedError:      # Windows
            pass
    try:
        await parar.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        log.info("Cerrando worker '%s'", args.rol)
        await bus.cerrar()
        if actuador:
            actuador.cerrar()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
