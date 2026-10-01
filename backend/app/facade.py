"""Patron Facade: un unico punto de entrada a todo el subsistema de inspeccion.

La API (y cualquier otro cliente: un script, un test, un worker batch) solo
conoce esto:

    facade = FacadeInspeccionHuevos(CONFIG)
    await facade.iniciar()
    resultado = await facade.inspeccionar(jpeg_bytes, estacion="banda-01")

Detras quedan escondidos: ONNX Runtime, el letterbox, el NMS en numpy, el
tracker entre frames, la politica de costo asimetrico, el productor de Kafka,
el guardado de evidencias y el actuador serial. Si manana se cambia YOLOv8m por
otro modelo, o Kafka por RabbitMQ, o el servo por un solenoide, la firma de
`inspeccionar` no se mueve.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .actuator import Actuador, construir_actuador
from .config import Config
from .consumers import ConsumidorActuador, ConsumidorCuraduria, ConsumidorMetricas
from .detector import DetectorHuevosONNX
from .events import BusEventos, construir_bus
from .metrics import AgregadorMetricas
from .routing import MotorDecision, SeguimientoBanda, fusionar_mismo_huevo
from .schemas import DeteccionHuevo, EventoHuevo, ResultadoInspeccion, Ruta, SaludSistema

log = logging.getLogger("banda.facade")


class FacadeInspeccionHuevos:
    """Orquesta captura -> inferencia -> decision -> evento -> actuacion."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.inicio = time.time()
        self.detector: DetectorHuevosONNX | None = None
        self.bus: BusEventos | None = None
        self.actuador: Actuador | None = None
        self.motor = MotorDecision(cfg)
        self.seguimiento = SeguimientoBanda(cfg)
        self.metricas = AgregadorMetricas(cfg.banda_objetivo_huevos_hora)
        self.consumidor_actuador: ConsumidorActuador | None = None
        self.consumidor_curaduria: ConsumidorCuraduria | None = None
        self._lock_inferencia = asyncio.Lock()

    # ------------------------------------------------------------ ciclo de vida
    async def iniciar(self) -> None:
        log.info("Cargando modelo %s ...", self.cfg.modelo_path)
        t0 = time.perf_counter()
        self.detector = DetectorHuevosONNX(
            self.cfg.modelo_path, self.cfg.modelo_imgsz, self.cfg.hilos_inferencia
        )
        log.info("Modelo listo en %.1fs (%s)", time.perf_counter() - t0, self.detector.proveedor)

        self.bus = await construir_bus(self.cfg.bus_tipo, self.cfg.kafka_bootstrap)
        self.actuador = construir_actuador(self.cfg)

        if self.cfg.workers_embebidos:
            await self._montar_consumidores()
        log.info("Facade lista. Bus=%s Actuador=%s", self.bus.descripcion, self.actuador.descripcion)

    async def _montar_consumidores(self) -> None:
        assert self.bus and self.actuador
        self.consumidor_actuador = ConsumidorActuador(self.cfg, self.actuador, self.bus.publicar)
        self.consumidor_curaduria = ConsumidorCuraduria(self.cfg)
        topico = self.cfg.topico_clasificados
        await self.bus.suscribir(topico, "grupo-actuador", self.consumidor_actuador)
        await self.bus.suscribir(topico, "grupo-metricas", ConsumidorMetricas(self.metricas))
        await self.bus.suscribir(topico, "grupo-curaduria", self.consumidor_curaduria)

    async def cerrar(self) -> None:
        if self.bus:
            await self.bus.cerrar()
        if self.actuador:
            self.actuador.cerrar()

    # --------------------------------------------------------------- operacion
    async def inspeccionar(
        self, imagen: bytes, estacion: str = "banda-01", guardar_evidencia: bool | None = None
    ) -> ResultadoInspeccion:
        """Procesa un frame de la banda y devuelve la ruta de cada huevo."""
        if self.detector is None or self.bus is None:
            raise RuntimeError("Facade no iniciada: llama a iniciar() primero")

        frame_id = f"frm_{uuid.uuid4().hex[:12]}"
        ts = datetime.now(timezone.utc).isoformat()

        # ONNX Runtime ya paraleliza internamente; serializar aqui evita que N
        # peticiones concurrentes se peleen los mismos cores y suban la p95.
        async with self._lock_inferencia:
            cajas, latencias, (W, H) = await asyncio.to_thread(
                self.detector.detectar, imagen, self.cfg.conf_min_deteccion, self.cfg.iou_nms
            )

        cajas = fusionar_mismo_huevo(cajas, self.cfg.fusion_iou_mismo_huevo)
        observaciones = self.seguimiento.asignar(cajas)
        detecciones: list[DeteccionHuevo] = []
        resumen = {r.value: 0 for r in Ruta}
        hay_revision = False
        # Se publica evento cuando el huevo es nuevo O cuando al verlo desde
        # otro frame cambio de ruta: asi una vista posterior puede corregir un
        # "ACEPTADO" prematuro.
        a_publicar: list[tuple[DeteccionHuevo, Ruta | None]] = []

        for caja, obs in zip(cajas, observaciones, strict=True):
            decision = self.motor.decidir(obs.p_roto, obs.p_sano, obs.nitidez)
            resumen[decision.ruta.value] += 1
            hay_revision = hay_revision or decision.ruta is Ruta.REVISION
            deteccion = DeteccionHuevo(
                egg_id=obs.egg_id,
                bbox=caja.bbox,
                clase="Roto" if obs.p_roto >= obs.p_sano else "Sano",
                confianza=round(max(obs.p_roto, obs.p_sano), 4),
                p_roto=round(obs.p_roto, 4),
                p_sano=round(obs.p_sano, 4),
                ruta=decision.ruta,
                motivo=decision.motivo,
                costo_esperado_aceptar=round(decision.costo_aceptar, 3),
                costo_esperado_descartar=round(decision.costo_descartar, 3),
                nuevo=obs.es_nuevo,
                nitidez=round(obs.nitidez, 1),
                calidad_insuficiente=decision.calidad_insuficiente,
                n_vistas=obs.n_vistas,
                p_roto_frame=round(caja.p_roto, 4),
            )
            detecciones.append(deteccion)

            cambio = obs.ruta_publicada is not None and obs.ruta_publicada != decision.ruta.value
            if obs.es_nuevo or cambio:
                anterior = Ruta(obs.ruta_publicada) if cambio else None
                a_publicar.append((deteccion, anterior))
                self.seguimiento.marcar_publicado(obs.egg_id, decision.ruta.value)

        guardar = self.cfg.guardar_todas_las_evidencias if guardar_evidencia is None else guardar_evidencia
        if guardar or hay_revision:
            await asyncio.to_thread(self._guardar_evidencia, frame_id, imagen, hay_revision)

        for det, ruta_anterior in a_publicar:
            evento = EventoHuevo(
                egg_id=det.egg_id,
                frame_id=frame_id,
                estacion=estacion,
                timestamp=ts,
                clase=det.clase,
                confianza=det.confianza,
                p_roto=det.p_roto,
                p_sano=det.p_sano,
                ruta=det.ruta,
                motivo=det.motivo,
                bbox=det.bbox,
                latencia_inferencia_ms=latencias["inferencia_ms"],
                modelo=Path(self.detector.modelo_path).name,
                nitidez=det.nitidez,
                n_vistas=det.n_vistas,
                ruta_anterior=ruta_anterior,
                actualizacion=ruta_anterior is not None,
            )
            await self.bus.publicar(self.cfg.topico_clasificados, det.egg_id, evento.model_dump(mode="json"))

        return ResultadoInspeccion(
            frame_id=frame_id,
            timestamp=ts,
            estacion=estacion,
            latencia_ms=latencias,
            ancho=W,
            alto=H,
            detecciones=detecciones,
            resumen=resumen,
            comando_actuador=self.consumidor_actuador.ultimo_comando
            if self.consumidor_actuador
            else None,
            calidad=self._diagnostico_calidad(detecciones),
        )

    def _diagnostico_calidad(self, detecciones: list[DeteccionHuevo]) -> dict[str, Any]:
        """Diagnostica LA FOTO, no el desenlace del ruteo.

        Son dos cosas distintas: una imagen puede estar movida y aun asi dejar
        ver la rotura (se descarta igual, y esta bien). Reportar "imagen
        nitida" en ese caso seria mentirle al operario, que no aprenderia a
        tomar mejores fotos.
        """
        if not detecciones:
            return {"sirve": True, "nitidez": None, "mensaje": "Sin huevo en el encuadre"}

        peor = min(detecciones, key=lambda d: d.nitidez)
        sirve = peor.nitidez >= self.cfg.nitidez_minima
        base = {
            "sirve": sirve,
            "nitidez": peor.nitidez,
            "minima": self.cfg.nitidez_minima,
            "bloqueo_decision": any(d.calidad_insuficiente for d in detecciones),
        }
        if sirve:
            base["mensaje"] = "Imagen nitida"
        elif base["bloqueo_decision"]:
            base["mensaje"] = "Foto borrosa: apoya el celular, acercate y repite la toma"
        else:
            # Borrosa, pero el modelo vio rotura de todos modos.
            base["mensaje"] = "Foto borrosa, pero la rotura se ve igual. Mejora la toma."
        return base

    def _guardar_evidencia(self, frame_id: str, imagen: bytes, es_revision: bool) -> None:
        destino = (self.cfg.dir_curaduria if es_revision else self.cfg.dir_evidencias) / f"{frame_id}.jpg"
        try:
            destino.write_bytes(imagen)
        except OSError as exc:
            log.warning("No pude guardar la evidencia %s: %s", destino, exc)

    # ------------------------------------------------------------------ estado
    def salud(self) -> SaludSistema:
        return SaludSistema(
            estado="ok" if self.detector else "cargando",
            modelo_cargado=self.detector is not None,
            modelo=self.cfg.modelo_nombre or Path(self.cfg.modelo_path).name,
            proveedor_onnx=self.detector.proveedor if self.detector else "-",
            bus=self.bus.descripcion if self.bus else "-",
            actuador=self.actuador.descripcion if self.actuador else "-",
            actuador_conectado=self.actuador.conectado if self.actuador else False,
            uptime_s=round(time.time() - self.inicio, 1),
        )

    def estado_operacion(self) -> dict[str, Any]:
        datos = self.metricas.instantanea()
        datos["desviados_por_servo"] = (
            self.consumidor_actuador.desviados if self.consumidor_actuador else 0
        )
        datos["candidatos_reentrenamiento"] = (
            self.consumidor_curaduria.registrados if self.consumidor_curaduria else 0
        )
        datos["politica"] = {
            "conf_min_deteccion": self.cfg.conf_min_deteccion,
            "tau_roto": self.cfg.tau_roto,
            "tau_sano": self.cfg.tau_sano,
            "costo_fn": self.cfg.costo_falso_negativo,
            "costo_fp": self.cfg.costo_falso_positivo,
        }
        return datos
