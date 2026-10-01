"""Prueba de humo: valida el sistema completo sin levantar uvicorn ni la app.

Corre esto PRIMERO apenas termine el setup en AWS. Si pasa, el resto de la demo
funciona. Tarda menos de un minuto.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from app.config import CONFIG           # noqa: E402
from app.facade import FacadeInspeccionHuevos  # noqa: E402
from app.schemas import Ruta            # noqa: E402

OK, FALLO = "  [OK]  ", "  [FALLO]"
errores = 0


def check(condicion: bool, mensaje: str, detalle: str = "") -> None:
    global errores
    print(f"{OK if condicion else FALLO} {mensaje}{(' -> ' + detalle) if detalle else ''}")
    if not condicion:
        errores += 1


async def main() -> None:
    print("=" * 72)
    print("PRUEBA DE HUMO - banda clasificadora de huevos")
    print("=" * 72)

    check(CONFIG.modelo_path.exists(), f"modelo presente en {CONFIG.modelo_path}")
    if not CONFIG.modelo_path.exists():
        sys.exit("\nCopia best.onnx a backend/modelos/ antes de seguir.")

    facade = FacadeInspeccionHuevos(CONFIG)
    await facade.iniciar()
    check(facade.detector is not None, "modelo ONNX cargado", facade.detector.proveedor)
    check(facade.bus is not None, "bus de eventos activo", facade.bus.descripcion)
    check(facade.actuador is not None, "actuador listo", facade.actuador.descripcion)

    # --- inferencia sobre imagenes reales --------------------------------
    carpeta = RAIZ.parent / "huevos" / "test" / "images"
    if carpeta.exists():
        imagenes = sorted(carpeta.glob("*.jpg"))
        random.seed(7)
        muestra = random.sample(imagenes, min(12, len(imagenes)))
        rutas: dict[str, int] = {}
        detecciones_totales = 0
        for img in muestra:
            res = await facade.inspeccionar(img.read_bytes(), estacion="smoke")
            detecciones_totales += len(res.detecciones)
            for d in res.detecciones:
                rutas[d.ruta.value] = rutas.get(d.ruta.value, 0) + 1
        check(detecciones_totales > 0, f"detecto huevos en {len(muestra)} imagenes",
              f"{detecciones_totales} detecciones, rutas={rutas}")
        check(res.latencia_ms["total_ms"] < 5000,
              "latencia por frame razonable", f"{res.latencia_ms['total_ms']:.0f} ms")
    else:
        print(f"  [SKIP] no encuentro {carpeta}, se omite la prueba con imagenes reales")

    # --- imagen sintetica sin huevos --------------------------------------
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (128, 128, 128)).save(buf, "JPEG")
    res = await facade.inspeccionar(buf.getvalue(), estacion="smoke-vacio")
    check(isinstance(res.detecciones, list), "banda vacia no revienta",
          f"{len(res.detecciones)} detecciones sobre fondo gris")

    # --- politica de ruteo -------------------------------------------------
    NITIDA = 50.0
    casos = [
        (0.93, 0.02, NITIDA, Ruta.DESCARTE, "roto evidente"),
        (0.04, 0.95, NITIDA, Ruta.ACEPTADO, "sano evidente"),
        (0.31, 0.45, NITIDA, Ruta.REVISION, "zona de incertidumbre"),
        (0.36, 0.88, NITIDA, Ruta.DESCARTE, "duda hacia el lado seguro"),
        # El caso reportado en campo: huevo roto que sale "Sano 71%".
        (0.20, 0.71, NITIDA, Ruta.REVISION, "sano dudoso (71%) no se acepta"),
        # Foto movida: no se decide sobre el huevo, se pide repetir.
        (0.05, 0.93, 1.5, Ruta.REVISION, "foto borrosa no se acepta"),
        # Pero si aun borrosa se ve rotura, se descarta igual.
        (0.80, 0.10, 1.5, Ruta.DESCARTE, "borrosa pero con rotura visible"),
    ]
    for p_roto, p_sano, nitidez, esperada, nombre in casos:
        obtenida = facade.motor.decidir(p_roto, p_sano, nitidez).ruta
        check(obtenida is esperada, f"politica: {nombre}", f"{obtenida.value} (esperado {esperada.value})")

    # --- banda de revision (independiente del .env: se prueba con y sin ella) --
    from app.routing import MotorDecision
    sin_banda = MotorDecision(CONFIG.model_copy(update={"tau_roto": 0.35, "tau_sano": 0.75, "tau_revision": None}))
    con_banda = MotorDecision(CONFIG.model_copy(update={"tau_roto": 0.35, "tau_sano": 0.75, "tau_revision": 0.07}))
    for motor, p_roto, p_sano, esperada, nombre in (
        (sin_banda, 0.20, 0.80, Ruta.ACEPTADO, "sin banda, p(Roto)=0.20 con p(Sano)=0.80 se acepta"),
        (con_banda, 0.20, 0.80, Ruta.REVISION, "con banda, ese mismo huevo va a revision"),
        (con_banda, 0.05, 0.90, Ruta.ACEPTADO, "con banda, un sano claro se sigue aceptando"),
        (con_banda, 0.40, 0.50, Ruta.DESCARTE, "con banda, sobre tau_roto se sigue descartando"),
    ):
        obtenida = motor.decidir(p_roto, p_sano, NITIDA).ruta
        check(obtenida is esperada, f"banda: {nombre}", f"{obtenida.value} (esperado {esperada.value})")
    # --- dos cajas sobre el MISMO huevo: un solo huevo y manda la sospecha ------
    from app.detector import Caja
    from app.routing import fusionar_mismo_huevo
    sano_box = Caja(x1=100, y1=100, x2=200, y2=220, p_roto=0.02, p_sano=0.90, nitidez=20.0)
    roto_box = Caja(x1=104, y1=98, x2=203, y2=224, p_roto=0.46, p_sano=0.10, nitidez=18.0)
    vecino = Caja(x1=205, y1=100, x2=300, y2=220, p_roto=0.01, p_sano=0.95, nitidez=20.0)
    fusion = fusionar_mismo_huevo([sano_box, roto_box, vecino], 0.6)
    mismo = next((c for c in fusion if c.x1 < 150), None)            # el huevo de las dos cajas
    ruta_fusion = con_banda.decidir(mismo.p_roto, mismo.p_sano, mismo.nitidez).ruta if mismo else None
    check(len(fusion) == 2 and ruta_fusion is Ruta.DESCARTE,
          "caja Roto + caja Sano sobre el mismo huevo = 1 huevo, manda la sospecha; el vecino sigue aparte",
          f"{len(fusion)} huevos, ruta {ruta_fusion.value if ruta_fusion else '-'}")
    print(f"  (politica activa: tau_roto={CONFIG.tau_roto} tau_sano={CONFIG.tau_sano} "
          f"tau_revision={CONFIG.tau_revision} | modelo: {facade.salud().modelo})")

    # --- eventos y consumidores --------------------------------------------
    await asyncio.sleep(0.4)
    m = facade.estado_operacion()
    check(m["total_huevos"] > 0, "el consumidor de metricas recibio eventos",
          f"{m['total_huevos']} huevos, throughput {m['throughput_huevos_hora']}/h")
    check(facade.consumidor_actuador is not None and facade.consumidor_actuador.ultimo_comando is not None,
          "el consumidor actuador emitio comando al servo",
          str(facade.consumidor_actuador.ultimo_comando if facade.consumidor_actuador else None))

    # --- seguimiento: el mismo frame dos veces no cuenta doble -------------
    facade.metricas.reiniciar()
    facade.seguimiento.reiniciar()
    if carpeta.exists():
        img = muestra[0].read_bytes()
        r1 = await facade.inspeccionar(img, estacion="smoke-track")
        r2 = await facade.inspeccionar(img, estacion="smoke-track")
        nuevos_1 = sum(d.nuevo for d in r1.detecciones)
        nuevos_2 = sum(d.nuevo for d in r2.detecciones)
        check(nuevos_1 > 0 and nuevos_2 == 0,
              "el tracker no cuenta dos veces el mismo huevo",
              f"frame1={nuevos_1} nuevos, frame2={nuevos_2} nuevos")
        check(r2.detecciones and r2.detecciones[0].n_vistas == 2,
              "consolida las vistas del mismo huevo",
              f"n_vistas={r2.detecciones[0].n_vistas if r2.detecciones else 0}")

    # --- compuerta de nitidez sobre una imagen real desenfocada ------------
    if carpeta.exists():
        from PIL import ImageFilter

        rotas = [f for f in imagenes if f.name.startswith("damaged")]
        facade.seguimiento.reiniciar()
        original = Image.open(rotas[0]).convert("RGB")
        buf_n = io.BytesIO(); original.save(buf_n, "JPEG", quality=90)
        buf_b = io.BytesIO(); original.filter(ImageFilter.GaussianBlur(4)).save(buf_b, "JPEG", quality=90)

        r_nitida = await facade.inspeccionar(buf_n.getvalue(), estacion="smoke-nitida")
        facade.seguimiento.reiniciar()
        r_borrosa = await facade.inspeccionar(buf_b.getvalue(), estacion="smoke-borrosa")
        n_ok = r_nitida.detecciones[0].nitidez if r_nitida.detecciones else 0
        n_mal = r_borrosa.detecciones[0].nitidez if r_borrosa.detecciones else 0
        check(n_ok > n_mal, "la nitidez distingue foto enfocada de movida",
              f"nitida={n_ok} vs borrosa={n_mal}")
        check(r_nitida.calidad.get("sirve") is True, "la foto enfocada pasa la compuerta")
        if r_borrosa.detecciones and not r_borrosa.detecciones[0].ruta is Ruta.DESCARTE:
            check(r_borrosa.calidad.get("sirve") is False,
                  "la foto movida se marca como no utilizable",
                  str(r_borrosa.calidad.get("mensaje")))

    await facade.cerrar()
    print("=" * 72)
    print(f"RESULTADO: {'TODO OK' if errores == 0 else f'{errores} PRUEBA(S) FALLARON'}")
    print("=" * 72)
    sys.exit(1 if errores else 0)


if __name__ == "__main__":
    asyncio.run(main())
