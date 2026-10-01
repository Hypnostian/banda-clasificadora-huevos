"""Politica de ruteo de la banda y seguimiento de huevos entre frames.

Aqui vive el valor agregado de ingenieria del proyecto: la banda NO tiene dos
salidas sino TRES, y el umbral no es 0.5 sino el que minimiza el COSTO esperado
bajo una matriz de costos asimetrica.

Justificacion medida sobre el split de test (145 imgs, 140 instancias Roto):

    conf   RotoOK  Roto->Sano  RotoMiss   ESCAPE%   FP sobre fondo
    0.15     133        0          7        5.00         22
    0.25     131        0          9        6.43         19
    0.45     129        0         11        7.86          9
    0.70     123        0         17       12.14          7

El error del modelo casi nunca es "confundir roto con sano" (0-1 casos): es
"no ver el huevo". Por eso bajar el umbral de deteccion compra recall barato,
y por eso la franja de incertidumbre se manda a un humano en vez de forzar una
decision binaria.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from .config import Config
from .detector import Caja
from .schemas import Ruta


@dataclass(slots=True)
class Decision:
    ruta: Ruta
    motivo: str
    costo_aceptar: float
    costo_descartar: float
    calidad_insuficiente: bool = False


class MotorDecision:
    """Traduce probabilidades del modelo en una ruta fisica de la banda.

    El orden de las reglas importa y es deliberado:

      1. Si la FOTO no sirve, no se decide nada sobre el huevo. Una imagen
         movida destruye la fisura y el modelo responde "Sano" con mucha
         confianza: es el peor error posible, porque es un falso negativo
         que ademas parece seguro. Antes de mirar las probabilidades hay que
         mirar si la evidencia es admisible.
      2. Duda hacia el lado caro (DESCARTE).
      3. Aceptar solo con evidencia fuerte.
      4. Todo lo demas, a un humano.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def decidir(self, p_roto: float, p_sano: float, nitidez: float | None = None) -> Decision:
        c_fn, c_fp = self.cfg.costo_falso_negativo, self.cfg.costo_falso_positivo

        # Costo esperado de cada accion, dado lo que cree el modelo.
        #   aceptar   -> si en realidad estaba roto, pagamos c_fn
        #   descartar -> si en realidad estaba sano, pagamos c_fp
        costo_aceptar = p_roto * c_fn
        costo_descartar = p_sano * c_fp

        # 1. Compuerta de calidad de imagen.
        if (
            self.cfg.exigir_nitidez
            and nitidez is not None
            and nitidez < self.cfg.nitidez_minima
            and p_roto < self.cfg.tau_roto
        ):
            return Decision(
                Ruta.REVISION,
                f"foto borrosa (nitidez={nitidez:.1f} < {self.cfg.nitidez_minima}): "
                "no se puede descartar una fisura",
                costo_aceptar,
                costo_descartar,
                calidad_insuficiente=True,
            )

        # 2. Sospecha de rotura: se descarta aunque la confianza no sea alta.
        if p_roto >= self.cfg.tau_roto:
            return Decision(
                Ruta.DESCARTE,
                f"p(Roto)={p_roto:.2f} >= tau_roto={self.cfg.tau_roto}",
                costo_aceptar,
                costo_descartar,
            )

        # 2b. Banda de revision (opcional): sospecha moderada de rotura. No se bota,
        #     porque puede ser un sano con brillo o una mancha, ni se acepta: con dos
        #     umbrales, p(Roto)=0.2 y p(Sano)=0.8 se aceptaria.
        if self.cfg.tau_revision is not None and p_roto >= self.cfg.tau_revision:
            return Decision(
                Ruta.REVISION,
                f"p(Roto)={p_roto:.2f} en la banda de revision "
                f"[{self.cfg.tau_revision}, {self.cfg.tau_roto})",
                costo_aceptar,
                costo_descartar,
            )

        # 3. Aceptar exige evidencia fuerte.
        if p_sano >= self.cfg.tau_sano:
            return Decision(
                Ruta.ACEPTADO,
                f"p(Sano)={p_sano:.2f} >= tau_sano={self.cfg.tau_sano} y p(Roto) bajo",
                costo_aceptar,
                costo_descartar,
            )

        # 4. Zona de incertidumbre.
        return Decision(
            Ruta.REVISION,
            f"zona de incertidumbre: p(Roto)={p_roto:.2f}, p(Sano)={p_sano:.2f} "
            f"< tau_sano={self.cfg.tau_sano}",
            costo_aceptar,
            costo_descartar,
        )


def fusionar_mismo_huevo(cajas: list[Caja], iou_min: float) -> list[Caja]:
    """Una sola caja por huevo dentro del MISMO frame, con la regla del peor caso.

    El NMS de YOLO es por clase: sobre un huevo dudoso pueden sobrevivir una caja "Roto" y
    otra "Sano". Sin esto salen como dos huevos con rutas contrarias (medido en fotos de
    campo: un roto salia a la vez ACEPTADO y DESCARTE) y el actuador recibiria PASAR y
    DESVIAR para el mismo huevo. Manda la vista mas sospechosa (mayor p_roto), igual que
    entre frames; la geometria es la de la caja mas confiable. Dos huevos distintos no
    llegan a IoU 0.6 (son solidos: a lo sumo se tocan). iou_min <= 0 lo apaga.
    """
    if iou_min <= 0 or len(cajas) < 2:
        return cajas
    quedan: list[Caja] = []
    for caja in sorted(cajas, key=lambda c: -c.confianza):
        bbox = (caja.x1, caja.y1, caja.x2, caja.y2)
        par = next((q for q in quedan if SeguimientoBanda._iou(bbox, (q.x1, q.y1, q.x2, q.y2)) >= iou_min), None)
        if par is None:
            quedan.append(caja)
            continue
        if caja.p_roto > par.p_roto:
            par.p_roto, par.p_sano = caja.p_roto, caja.p_sano
        par.nitidez = max(par.nitidez, caja.nitidez)
    return quedan


@dataclass(slots=True)
class _Pista:
    egg_id: str
    bbox: tuple[float, float, float, float]
    visto_en: float
    # Peor caso acumulado: la vista que mas sospecho de rotura.
    peor_p_roto: float = 0.0
    p_sano_asociado: float = 0.0
    mejor_nitidez: float = 0.0
    n_vistas: int = 1
    ruta_publicada: str | None = None


@dataclass(slots=True)
class Observacion:
    """Lo que la banda sabe de un huevo tras ver N frames de el."""

    egg_id: str
    es_nuevo: bool
    p_roto: float          # consolidado (peor caso entre todas las vistas)
    p_sano: float          # el de la vista que dio el peor p_roto
    nitidez: float         # la MEJOR nitidez vista (la foto mas util)
    n_vistas: int
    ruta_publicada: str | None


@dataclass
class SeguimientoBanda:
    """Evita contar el mismo huevo dos veces cuando la camara manda frames seguidos.

    Es un tracker deliberadamente simple (IoU contra el frame anterior + TTL).
    Sin esto, el throughput del dashboard seria una mentira: un huevo quieto
    frente a la camara durante 3 segundos se contaria 8 veces.
    """

    cfg: Config
    _pistas: list[_Pista] = field(default_factory=list)

    @staticmethod
    def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0:
            return 0.0
        aa = (a[2] - a[0]) * (a[3] - a[1])
        bb = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aa + bb - inter + 1e-9)

    def asignar(self, cajas: list[Caja]) -> list[Observacion]:
        """Empareja cada caja con el huevo que ya venia siguiendo y consolida.

        La consolidacion es "peor caso": si en alguna de las vistas el modelo
        sospecho rotura, esa sospecha manda. Un huevo roto visto de frente
        parece sano; visto de lado no. Quedarse con la vista mas favorable
        seria justo el error que no podemos permitirnos.
        """
        ahora = time.time()
        self._pistas = [p for p in self._pistas if ahora - p.visto_en <= self.cfg.seguimiento_ttl_s]
        disponibles = list(self._pistas)
        salida: list[Observacion] = []
        vivas: list[_Pista] = []

        for caja in cajas:
            bbox = (caja.x1, caja.y1, caja.x2, caja.y2)
            emparejada, mejor_iou = None, self.cfg.seguimiento_iou_min
            for pista in disponibles:
                iou = self._iou(bbox, pista.bbox)
                if iou >= mejor_iou:
                    emparejada, mejor_iou = pista, iou

            if emparejada is not None:
                disponibles.remove(emparejada)
                pista = emparejada
                pista.bbox, pista.visto_en = bbox, ahora
                pista.n_vistas += 1
                pista.mejor_nitidez = max(pista.mejor_nitidez, caja.nitidez)
                if self.cfg.consolidar_por_huevo:
                    if caja.p_roto > pista.peor_p_roto:
                        pista.peor_p_roto, pista.p_sano_asociado = caja.p_roto, caja.p_sano
                else:
                    pista.peor_p_roto, pista.p_sano_asociado = caja.p_roto, caja.p_sano
                es_nuevo = False
            else:
                pista = _Pista(
                    egg_id=f"egg_{uuid.uuid4().hex[:10]}",
                    bbox=bbox,
                    visto_en=ahora,
                    peor_p_roto=caja.p_roto,
                    p_sano_asociado=caja.p_sano,
                    mejor_nitidez=caja.nitidez,
                )
                es_nuevo = True

            vivas.append(pista)
            salida.append(
                Observacion(
                    egg_id=pista.egg_id,
                    es_nuevo=es_nuevo,
                    p_roto=pista.peor_p_roto,
                    p_sano=pista.p_sano_asociado,
                    nitidez=pista.mejor_nitidez,
                    n_vistas=pista.n_vistas,
                    ruta_publicada=pista.ruta_publicada,
                )
            )

        self._pistas = vivas
        return salida

    def marcar_publicado(self, egg_id: str, ruta: str) -> None:
        """Recuerda que ruta se emitio ya para este huevo, para no repetirla."""
        for pista in self._pistas:
            if pista.egg_id == egg_id:
                pista.ruta_publicada = ruta
                return

    def reiniciar(self) -> None:
        self._pistas.clear()
