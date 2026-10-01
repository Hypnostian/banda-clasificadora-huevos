"""Simulador de banda transportadora: reproduce una corrida de produccion.

Toma imagenes de una carpeta (por defecto el split de test) y las manda al
endpoint /inspeccion al ritmo de una banda real. Sirve para dos cosas:

  1. Demo sin telefono. Si la app Expo falla en la sustentacion, esto llena el
     dashboard igual. NUNCA dejes que la demo dependa de un solo camino.
  2. Medir el throughput real del backend en la instancia de AWS.

Uso:
    python scripts/simular_banda.py --api http://localhost:8000 --hpm 40
    python scripts/simular_banda.py --carpeta ../huevos/test/images --n 60
"""
from __future__ import annotations

import argparse
import asyncio
import random
import time
from pathlib import Path

import httpx

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


async def main() -> None:
    p = argparse.ArgumentParser(description="Simulador de banda transportadora")
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--carpeta", default=str(Path(__file__).resolve().parents[2] / "huevos" / "test" / "images"))
    p.add_argument("--hpm", type=int, default=30, help="huevos por minuto que entrega la banda")
    p.add_argument("--n", type=int, default=0, help="cuantos huevos pasar (0 = toda la carpeta)")
    p.add_argument("--estacion", default="banda-01")
    p.add_argument("--semilla", type=int, default=42)
    args = p.parse_args()

    carpeta = Path(args.carpeta)
    imagenes = sorted(f for f in carpeta.iterdir() if f.suffix.lower() in EXT)
    if not imagenes:
        raise SystemExit(f"No hay imagenes en {carpeta}")
    random.seed(args.semilla)
    random.shuffle(imagenes)
    if args.n:
        imagenes = (imagenes * (args.n // len(imagenes) + 1))[: args.n]

    intervalo = 60.0 / max(1, args.hpm)
    print(f"Banda: {len(imagenes)} huevos a {args.hpm}/min (1 cada {intervalo:.2f}s) -> {args.api}")
    print("-" * 78)

    conteo = {"ACEPTADO": 0, "DESCARTE": 0, "REVISION": 0}
    latencias: list[float] = []
    inicio = time.time()

    async with httpx.AsyncClient(timeout=60.0) as cli:
        try:
            salud = (await cli.get(f"{args.api}/salud")).json()
            print(f"Backend: {salud['modelo']} · {salud['proveedor_onnx']} · bus={salud['bus']} · actuador={salud['actuador']}\n")
        except Exception as exc:
            raise SystemExit(f"El backend no responde en {args.api}: {exc}")

        for i, ruta_img in enumerate(imagenes, 1):
            t_ciclo = time.perf_counter()
            try:
                r = await cli.post(
                    f"{args.api}/inspeccion",
                    files={"imagen": (ruta_img.name, ruta_img.read_bytes(), "image/jpeg")},
                    data={"estacion": args.estacion, "guardar_evidencia": "false"},
                )
                r.raise_for_status()
                res = r.json()
            except Exception as exc:
                print(f"[{i:>4}] ERROR {ruta_img.name}: {exc}")
                continue

            latencias.append(res["latencia_ms"]["total_ms"])
            for det in res["detecciones"]:
                conteo[det["ruta"]] = conteo.get(det["ruta"], 0) + 1
            etiquetas = ", ".join(
                f"{d['clase']}({d['confianza']:.2f})->{d['ruta']}" for d in res["detecciones"]
            ) or "sin huevo en el frame"
            print(f"[{i:>4}] {ruta_img.name[:34]:<34} {res['latencia_ms']['total_ms']:>6.0f} ms  {etiquetas}")

            resto = intervalo - (time.perf_counter() - t_ciclo)
            if resto > 0:
                await asyncio.sleep(resto)

    transcurrido = time.time() - inicio
    total = sum(conteo.values())
    latencias.sort()
    print("-" * 78)
    print(f"Corrida: {total} huevos en {transcurrido:.1f}s")
    for k, v in conteo.items():
        print(f"  {k:<9} {v:>5}  ({100*v/max(1,total):.1f}%)")
    if latencias:
        print(f"  latencia e2e  p50={latencias[len(latencias)//2]:.0f} ms  "
              f"p95={latencias[int(len(latencias)*0.95)]:.0f} ms")
        techo = 3600 / (latencias[len(latencias) // 2] / 1000)
        print(f"  techo teorico del backend: {techo:,.0f} huevos/h")


if __name__ == "__main__":
    asyncio.run(main())
