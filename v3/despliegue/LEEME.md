# Banda de huevos — v3 en el servidor

| carpeta | versión | puerto | papel |
|---|---|---|---|
| `~/huevos_v3` | **v3** (YOLOv8l, 8 fuentes) | **8000** | la que usa la app y el dashboard |
| `~/huevos` | modelo 1 (YOLOv8m) | 8001 | respaldo, sin tocar (solo local salvo que abras el puerto) |

La app **no cambia**: sigue apuntando a `http://34.236.46.110:8000`. En su barra superior y en
Ajustes aparece `v3 YOLOv8l`, así se sabe qué versión respondió.

## Política de la v3 (`~/huevos_v3/backend/.env`)

| p(Roto) | ruta |
|---|---|
| ≥ 0,35 | DESCARTE |
| 0,07 – 0,35 | REVISIÓN (banda nueva: ni se bota ni se acepta) |
| < 0,07 y p(Sano) ≥ 0,75 y foto nítida | ACEPTADO |
| lo demás | REVISIÓN |

Medido en el test (183 huevos): 0 rotos al empaque, 1 de 43 sanos botados, ~4 % a revisión.
Detalle en `docs/COMPARACION_V1_V3.md`.

## Comandos (al entrar por SSH se muestra el estado)

| comando | qué hace |
|---|---|
| `banda` | estado de las dos versiones |
| `banda-log` | log en vivo de la versión del 8000 |
| `banda-restart` | reinicia la v3 |
| `banda-smoke` | pruebas de humo de la v3 |
| `banda-demo` | simula 40 huevos contra la API |
| `banda-cambiar 1` | **vuelta atrás**: modelo 1 al 8000 (y `banda-cambiar v3` para volver) |

Reinstalar o actualizar desde la PC (raíz del proyecto, Git Bash): `bash v3/despliegue/subir_v3.sh`.
