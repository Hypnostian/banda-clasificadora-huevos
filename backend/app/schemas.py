"""Contratos de datos: lo que entra por la API y lo que viaja por el bus de eventos."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Ruta(str, Enum):
    """Las tres salidas fisicas de la banda emulada."""

    ACEPTADO = "ACEPTADO"          # sigue derecho al empaque
    DESCARTE = "DESCARTE"          # el servo lo empuja a la canasta de rotos
    REVISION = "REVISION"          # se desvia a inspeccion humana (zona de incertidumbre)


class DeteccionHuevo(BaseModel):
    egg_id: str = Field(..., description="Identificador de trazabilidad del huevo")
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2] en pixeles del frame original")
    clase: Literal["Roto", "Sano"]
    confianza: float
    p_roto: float
    p_sano: float
    ruta: Ruta
    motivo: str
    costo_esperado_aceptar: float
    costo_esperado_descartar: float
    nuevo: bool = Field(True, description="False si es el mismo huevo ya contado en el frame anterior")
    nitidez: float = Field(0.0, description="Varianza del laplaciano del recorte; baja = foto movida")
    calidad_insuficiente: bool = Field(False, description="La foto no permite decidir")
    n_vistas: int = Field(1, description="Cuantos frames de este mismo huevo se han consolidado")
    p_roto_frame: float = Field(0.0, description="p(Roto) de ESTE frame, antes de consolidar")


class ResultadoInspeccion(BaseModel):
    frame_id: str
    timestamp: str
    estacion: str
    latencia_ms: dict[str, float]
    ancho: int
    alto: int
    detecciones: list[DeteccionHuevo]
    resumen: dict[str, int]
    comando_actuador: str | None = None
    calidad: dict[str, object] = Field(
        default_factory=dict,
        description="Diagnostico de la foto: nitidez, si sirve, y que hacer si no",
    )


class EventoHuevo(BaseModel):
    """Payload que se publica en el topico `huevos.clasificados`."""

    evento: Literal["huevo.clasificado"] = "huevo.clasificado"
    version: int = 1
    egg_id: str
    frame_id: str
    estacion: str
    timestamp: str
    clase: Literal["Roto", "Sano"]
    confianza: float
    p_roto: float
    p_sano: float
    ruta: Ruta
    motivo: str
    bbox: list[float]
    latencia_inferencia_ms: float
    modelo: str
    nitidez: float = 0.0
    n_vistas: int = 1
    ruta_anterior: Ruta | None = Field(
        None, description="Si el huevo cambio de ruta al verlo desde otro frame"
    )
    actualizacion: bool = Field(
        False, description="True si corrige un veredicto ya emitido para el mismo egg_id"
    )


class ComandoBanda(BaseModel):
    """Payload que se publica en el topico `banda.comandos`."""

    evento: Literal["banda.comando"] = "banda.comando"
    egg_id: str
    accion: Literal["DESVIAR", "PASAR", "MARCAR"]
    timestamp: str


class SaludSistema(BaseModel):
    estado: str
    modelo_cargado: bool
    modelo: str
    proveedor_onnx: str
    bus: str
    actuador: str
    actuador_conectado: bool
    uptime_s: float
