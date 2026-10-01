"""API FastAPI de la banda clasificadora de huevos.

Toda la complejidad esta detras de FacadeInspeccionHuevos: esta capa solo
valida entradas HTTP y serializa salidas.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .config import CONFIG
from .facade import FacadeInspeccionHuevos
from .schemas import ResultadoInspeccion, SaludSistema

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s"
)
log = logging.getLogger("banda.api")

facade = FacadeInspeccionHuevos(CONFIG)
AQUI = Path(__file__).resolve().parent


@asynccontextmanager
async def ciclo_vida(app: FastAPI):
    await facade.iniciar()
    yield
    await facade.cerrar()


app = FastAPI(
    title="Banda clasificadora de huevos - CNN End2End",
    description=(
        "Deteccion de huevos con dano sobre banda transportadora emulada. "
        "YOLOv8m -> ONNX Runtime -> politica de ruteo asimetrica -> eventos -> servo."
    ),
    version="1.0.0",
    lifespan=ciclo_vida,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CONFIG.cors_origins.split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ dashboard
@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(AQUI / "dashboard.html")


# ---------------------------------------------------------------------- salud
@app.get("/salud", response_model=SaludSistema, tags=["sistema"])
async def salud() -> SaludSistema:
    return facade.salud()


# ------------------------------------------------------------------ inspeccion
@app.post("/inspeccion", response_model=ResultadoInspeccion, tags=["banda"])
async def inspeccion(
    imagen: UploadFile = File(..., description="Frame JPEG/PNG capturado por la app camara"),
    estacion: str = Form("banda-01"),
    guardar_evidencia: bool = Form(False),
) -> ResultadoInspeccion:
    """Procesa un frame de la banda y devuelve la ruta que toma cada huevo."""
    contenido = await imagen.read()
    if not contenido:
        raise HTTPException(400, "Imagen vacia")
    if len(contenido) > CONFIG.max_bytes_imagen:
        raise HTTPException(
            413, f"Imagen de {len(contenido)//1024} KB; el maximo es {CONFIG.max_bytes_imagen//1024} KB"
        )
    try:
        return await facade.inspeccionar(contenido, estacion, guardar_evidencia)
    except Exception as exc:
        log.exception("Fallo procesando el frame")
        raise HTTPException(500, f"Error de inferencia: {exc}") from exc


# ------------------------------------------------------------------- metricas
@app.get("/metricas", tags=["banda"])
async def metricas() -> dict:
    return facade.estado_operacion()


@app.get("/stream", tags=["banda"])
async def stream(request: Request) -> StreamingResponse:
    """Server-Sent Events: empuja el estado de la banda al dashboard."""
    cola = facade.metricas.suscribirse()

    async def generador():
        try:
            yield f"data: {json.dumps(facade.estado_operacion())}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    await asyncio.wait_for(cola.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(facade.estado_operacion())}\n\n"
        finally:
            facade.metricas.desuscribirse(cola)

    return StreamingResponse(
        generador(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# -------------------------------------------------------------------- control
@app.post("/banda/reiniciar", tags=["banda"])
async def reiniciar() -> dict:
    """Pone el turno de produccion en cero (util entre corridas de la demo)."""
    facade.metricas.reiniciar()
    facade.seguimiento.reiniciar()
    return {"ok": True, "mensaje": "Contadores y seguimiento reiniciados"}


@app.get("/politica", tags=["banda"])
async def leer_politica() -> dict:
    return {
        "conf_min_deteccion": CONFIG.conf_min_deteccion,
        "tau_roto": CONFIG.tau_roto,
        "tau_sano": CONFIG.tau_sano,
        "tau_revision": CONFIG.tau_revision,
        "costo_falso_negativo": CONFIG.costo_falso_negativo,
        "costo_falso_positivo": CONFIG.costo_falso_positivo,
    }


@app.put("/politica", tags=["banda"])
async def ajustar_politica(
    tau_roto: float | None = None,
    tau_sano: float | None = None,
    conf_min_deteccion: float | None = None,
    tau_revision: float | None = None,
) -> dict:
    """Mueve los umbrales en caliente.

    Sirve para sustentar en vivo el trade-off: bajar tau_roto sube la captura
    de rotos y sube el descarte de sanos; el jurado lo ve en el dashboard sin
    reiniciar nada. tau_revision >= tau_roto apaga la banda de revision.
    """
    for nombre, valor in (
        ("tau_roto", tau_roto),
        ("tau_sano", tau_sano),
        ("conf_min_deteccion", conf_min_deteccion),
        ("tau_revision", tau_revision),
    ):
        if valor is None:
            continue
        if not 0.0 < valor < 1.0:
            raise HTTPException(400, f"{nombre} debe estar entre 0 y 1")
        setattr(CONFIG, nombre, valor)
    return await leer_politica()
