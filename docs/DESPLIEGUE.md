# Despliegue en producción — EC2 `34.236.46.110`

Estado: **desplegado y corriendo**. El servicio arranca solo con la máquina.

> **Desde el 2026-09-25 la app usa la v3.** Corre en `~/huevos_v3`, en el puerto 8000, con el servicio
> `banda-huevos-v3`. El modelo 1 sigue intacto en `~/huevos`, de respaldo en el 8001. Vuelta atrás
> con un comando: `banda-cambiar 1`. Detalle en [`v3/despliegue/LEEME.md`](../v3/despliegue/LEEME.md).
> Lo que sigue describe la instalación original, que es la carpeta del modelo 1.

## El servidor

| | |
|---|---|
| IP elástica | `34.236.46.110` (fija, no cambia entre sesiones) |
| Instancia | `t2.large` · `i-0feb8c66787e3923e` |
| SO | Ubuntu 24.04.4 LTS · usuario `ubuntu` |
| CPU | 2 vCPU · Xeon E5-2686 v4 @2.30 GHz (Broadwell) · **AVX2, sin AVX-512** |
| RAM / disco | 7.9 GB · 30 GB (25 GB libres) |
| Python | 3.12.3 |
| Carpeta | `~/huevos/` (no toca `placa/`, `bike/` ni `proyecto/`) |
| Servicio | `banda-huevos.service` · `enabled` + `Restart=always` |
| Security Group | `launch-wizard-1` |

**Rendimiento medido en esta máquina** (no estimado):

| hilos | latencia media | huevos/h |
|---|---|---|
| 1 | 1160 ms | 3.103 |
| **2** | **637 ms** | **5.648** |
| 3 | 984 ms | 3.660 |
| 4 | 965 ms | 3.732 |

`HILOS_INFERENCIA=2` es el óptimo y es lo que quedó en el `.env`. Con 3-4 hilos
se sobresuscriben los 2 vCPU y empeora.

> El Xeon E5-2686 v4 es de 2016 y solo tiene AVX2. Por eso son 640 ms y no los
> ~400 ms que da un CPU moderno. Es el techo de esta instancia, no un problema
> de configuración.

**Métricas reproducidas en el servidor** — idénticas a las locales:
`mAP50 0.9769 · mAP50-95 0.9238 · P 0.9596 · R 0.9562`.

---

## Accesibilidad — verificada

El puerto **8000 está abierto** en el Security Group y comprobado desde
internet.

| Prueba | Resultado |
|---|---|
| `GET /salud` público | HTTP 200 · 200 ms |
| `GET /` (dashboard) | HTTP 200 · 14 KB · 291 ms |
| `GET /docs` (Swagger) | HTTP 200 |
| `GET /stream` (SSE) | stream vivo confirmado |
| `POST /inspeccion` huevo roto (42 KB) | `Roto 0.833 → DESCARTE` · **983 ms** |
| `POST /inspeccion` huevo sano (23 KB) | `Sano 0.958 → ACEPTADO` · **894 ms** |

Esas dos últimas son el camino real del celular: imagen redimensionada a 640 px
y JPEG q0.6, igual que hace `comprimir()` en la app. **~0.9 s de punta a punta**
(≈640 ms de inferencia + ≈250 ms de red).

### Arranque automático — verificado con un reinicio real

No basta con que `systemctl is-enabled` diga `enabled`. Reinicié la instancia
completa y la API volvió sola en **~85 s**, sin intervención manual, y la prueba
de humo pasó 14/14 después. Los contadores del turno viven en memoria, así que
un reinicio los deja en cero.

> ### Seguridad — léelo
> Con `0.0.0.0/0` **cualquiera que conozca la IP puede mandar imágenes al
> endpoint**: la API no tiene autenticación. Para una demo académica es
> aceptable, pero:
>
> - **Apaga el servicio cuando no lo uses**: `banda-stop`.
> - O restringe el *Source* del Security Group a tu IP — con la salvedad de que
>   entonces el celular en datos móviles no entrará.
>
> Si quieres autenticación de verdad para dejarlo encendido, se resuelve con una
> cabecera de API key en el backend y en la app; no está puesto porque añade un
> punto de fallo justo antes de la sustentación.

---

## Conectarte por SSH desde PowerShell

La llave necesita permisos restringidos o Windows la rechaza:

```powershell
icacls "D:\eggdetection\llaveproyectos.pem" /inheritance:r /grant:r "$($env:USERNAME):(F)"
```

```powershell
ssh -i "D:\eggdetection\llaveproyectos.pem" ubuntu@34.236.46.110
```

Al entrar verás el estado del sistema automáticamente.

### Comandos disponibles en el servidor

| Comando | Qué hace |
|---|---|
| `banda` | estado completo (servicio, API, modelo, turno, puerto) |
| `banda-log` | logs en vivo |
| `banda-restart` | reinicia y muestra el estado |
| `banda-stop` / `banda-start` | apagar / encender |
| `banda-demo` | simula 40 huevos contra la API |
| `banda-eval` | reproduce las métricas sobre el split test |
| `banda-smoke` | las 14 verificaciones del sistema |

---

## Operación

- **Logs**: el servicio escribe en `~/huevos/backend/logs/api.log`. Hay
  `logrotate` configurado (`/etc/logrotate.d/banda-huevos`): semanal, 4
  rotaciones, corte a 50 MB, comprimido. Sin eso el archivo crecería sin
  límite ahora que el endpoint es público y uvicorn registra cada petición.
- **RAM del servicio**: ~365 MB de los 7.9 GB. Sin presión de memoria.
- **Disco**: 21 % usado. La curaduría guarda un JPEG por cada huevo que cae en
  `REVISION` (~40 KB); a ese ritmo el disco no es una preocupación, pero si
  dejas el endpoint público y encendido mucho tiempo, revísalo con `df -h`.

## Túnel SSH (plan B desde el PC si algo pasa con el Security Group)

Funciona sin tocar el Security Group, pero **solo sirve desde el PC**, no desde
el celular:

```powershell
ssh -i "D:\eggdetection\llaveproyectos.pem" -N -L 8000:127.0.0.1:8000 ubuntu@34.236.46.110
```

Deja esa ventana abierta y abre <http://localhost:8000/>.

---

## Cambiar a Kafka real (opcional)

Quedó con `BUS_TIPO=memoria`, que no necesita Docker y arranca siempre. Para el
broker real:

```bash
sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker ubuntu && newgrp docker
cd ~/huevos/backend && docker compose up -d
sed -i 's/^BUS_TIPO=memoria/BUS_TIPO=kafka/' .env
sudo systemctl restart banda-huevos
```

Hay que abrir también el **8080** en el Security Group para la consola de
Redpanda. La interfaz del bus es idéntica: no cambia nada más del código.

---

## Actualizar el código desde tu PC

```powershell
cd D:\eggdetection
tar --exclude='__pycache__' --exclude='.venv' --exclude='evidencias' --exclude='*.onnx' --exclude='.env' -czf backend.tar.gz backend
scp -i llaveproyectos.pem backend.tar.gz ubuntu@34.236.46.110:~/huevos/
ssh -i llaveproyectos.pem ubuntu@34.236.46.110 "cd ~/huevos && tar xzf backend.tar.gz && rm backend.tar.gz && sudo systemctl restart banda-huevos && sleep 8 && ./estado.sh"
```

El `.env` del servidor está excluido del paquete a propósito: no se pisa.

---

## Diagnóstico rápido

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `curl` desde el PC no responde | el servicio está apagado, o alguien tocó la regla del SG | `banda` en el servidor |
| La app dice "No conecta" | lo mismo, o el servicio caído | `banda` en el servidor |
| El servicio no arranca | falta el modelo o el `.env` | `banda-log` y mirar el error |
| Respuestas de >2 s | créditos de CPU del t2 agotados | `top` → columna `st` (steal) |
| `Permission denied (publickey)` | ACL de la llave | el `icacls` de arriba |
| Sale `ec2-user@... denied` | usuario equivocado | el usuario es **`ubuntu`** |
