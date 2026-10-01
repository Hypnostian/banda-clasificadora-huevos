"""Agregador de metricas de produccion en tiempo real.

Es el consumidor del topico `huevos.clasificados` que alimenta el dashboard.
Reporta KPIs de PLANTA (tasa de rotura, throughput, tasa de revision), no
metricas de ML: eso es lo que le sirve a un jefe de linea.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any

from .schemas import Ruta


class AgregadorMetricas:
    VENTANA_THROUGHPUT_S = 60.0
    BUCKET_S = 10.0
    MAX_BUCKETS = 60          # 10 minutos de serie
    MAX_RECIENTES = 40

    def __init__(self, objetivo_huevos_hora: int) -> None:
        self.objetivo = objetivo_huevos_hora
        self.inicio = time.time()
        self.total = 0
        self.correcciones = 0
        self.por_ruta: dict[str, int] = {r.value: 0 for r in Ruta}
        self.por_clase: dict[str, int] = {"Roto": 0, "Sano": 0}
        self._marcas: deque[float] = deque()
        self._latencias: deque[float] = deque(maxlen=500)
        self._confianzas: deque[float] = deque(maxlen=500)
        self._serie: deque[dict[str, Any]] = deque(maxlen=self.MAX_BUCKETS)
        self.recientes: deque[dict[str, Any]] = deque(maxlen=self.MAX_RECIENTES)
        self._suscriptores: list[asyncio.Queue] = []

    # ------------------------------------------------------------- consumidor
    async def consumir(self, evento: dict[str, Any]) -> None:
        ahora = time.time()
        ruta = evento.get("ruta", Ruta.REVISION.value)
        clase = evento.get("clase", "Sano")
        ruta_anterior = evento.get("ruta_anterior")

        if evento.get("actualizacion") and ruta_anterior:
            # El huevo ya se habia contado; al verlo desde otro frame cambio de
            # ruta. Se mueve de casilla, NO se cuenta otra vez.
            self.por_ruta[ruta_anterior] = max(0, self.por_ruta.get(ruta_anterior, 0) - 1)
            self.por_ruta[ruta] = self.por_ruta.get(ruta, 0) + 1
            self.correcciones += 1
        else:
            self.total += 1
            self.por_ruta[ruta] = self.por_ruta.get(ruta, 0) + 1
            self.por_clase[clase] = self.por_clase.get(clase, 0) + 1
            self._marcas.append(ahora)
        while self._marcas and ahora - self._marcas[0] > self.VENTANA_THROUGHPUT_S:
            self._marcas.popleft()
        self._latencias.append(float(evento.get("latencia_inferencia_ms", 0.0)))
        self._confianzas.append(float(evento.get("confianza", 0.0)))
        if not evento.get("actualizacion"):
            # Una correccion mueve al huevo de casilla; no es un huevo mas en
            # la serie temporal.
            self._acumular_serie(ahora, ruta)
        self.recientes.appendleft(
            {
                "egg_id": evento.get("egg_id"),
                "clase": clase,
                "confianza": round(float(evento.get("confianza", 0)), 3),
                "ruta": ruta,
                "motivo": evento.get("motivo", ""),
                "timestamp": evento.get("timestamp"),
            }
        )
        await self._difundir()

    def _acumular_serie(self, ahora: float, ruta: str) -> None:
        cubo = int(ahora // self.BUCKET_S) * self.BUCKET_S
        if not self._serie or self._serie[-1]["t"] != cubo:
            self._serie.append({"t": cubo, "ACEPTADO": 0, "DESCARTE": 0, "REVISION": 0})
        self._serie[-1][ruta] = self._serie[-1].get(ruta, 0) + 1

    # ---------------------------------------------------------------- lectura
    def instantanea(self) -> dict[str, Any]:
        transcurrido = max(1e-6, time.time() - self.inicio)
        ventana = min(transcurrido, self.VENTANA_THROUGHPUT_S)
        huevos_hora = len(self._marcas) / ventana * 3600 if self._marcas else 0.0
        lat = sorted(self._latencias)
        procesados = max(1, self.total)
        return {
            "uptime_s": round(transcurrido, 1),
            "total_huevos": self.total,
            "correcciones_por_nueva_vista": self.correcciones,
            "por_ruta": dict(self.por_ruta),
            "por_clase": dict(self.por_clase),
            "tasa_rotura_pct": round(100 * self.por_ruta.get("DESCARTE", 0) / procesados, 2),
            "tasa_revision_pct": round(100 * self.por_ruta.get("REVISION", 0) / procesados, 2),
            "throughput_huevos_hora": round(huevos_hora),
            "objetivo_huevos_hora": self.objetivo,
            "cumplimiento_pct": round(100 * huevos_hora / self.objetivo, 1) if self.objetivo else 0,
            "latencia_p50_ms": round(lat[len(lat) // 2], 1) if lat else 0.0,
            "latencia_p95_ms": round(lat[int(len(lat) * 0.95)], 1) if lat else 0.0,
            "confianza_media": round(sum(self._confianzas) / len(self._confianzas), 3)
            if self._confianzas
            else 0.0,
            "serie": list(self._serie),
            "recientes": list(self.recientes),
        }

    # ------------------------------------------------------------------- SSE
    def suscribirse(self) -> asyncio.Queue:
        cola: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._suscriptores.append(cola)
        return cola

    def desuscribirse(self, cola: asyncio.Queue) -> None:
        if cola in self._suscriptores:
            self._suscriptores.remove(cola)

    async def _difundir(self) -> None:
        if not self._suscriptores:
            return
        datos = self.instantanea()
        for cola in list(self._suscriptores):
            if cola.full():
                try:
                    cola.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                cola.put_nowait(datos)
            except asyncio.QueueFull:
                pass

    def reiniciar(self) -> None:
        """Pone los contadores en cero sin botar a los clientes SSE conectados."""
        suscriptores = self._suscriptores
        self.__init__(self.objetivo)  # noqa: PLC2801
        self._suscriptores = suscriptores
