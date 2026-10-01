#!/usr/bin/env bash
# =============================================================================
#  Setup del backend de la banda clasificadora en AWS Academy (Learner Lab).
#
#  Pensado para la ventana de 4 horas: idempotente, sin compilar nada, sin GPU
#  y sin PyTorch. De cero a servicio arriba en ~3 minutos.
#
#      chmod +x scripts/setup_aws.sh
#      ./scripts/setup_aws.sh            # solo backend
#      ./scripts/setup_aws.sh --kafka    # backend + Redpanda en Docker
#
#  ANTES de correrlo, en la consola de AWS:
#    - Security Group: abrir TCP 8000 (API) y, si usas --kafka, 8080 (consola).
#    - Subir best.onnx a backend/modelos/  (scp o wget desde S3/Drive).
# =============================================================================
set -euo pipefail

AZUL='\033[0;34m'; VERDE='\033[0;32m'; AMBAR='\033[0;33m'; ROJO='\033[0;31m'; OFF='\033[0m'
paso()  { echo -e "\n${AZUL}==>${OFF} $*"; }
ok()    { echo -e "${VERDE}  OK${OFF}  $*"; }
aviso() { echo -e "${AMBAR}  !!${OFF}  $*"; }
morir() { echo -e "${ROJO}  XX${OFF}  $*" >&2; exit 1; }

CON_KAFKA=0
[[ "${1:-}" == "--kafka" ]] && CON_KAFKA=1

cd "$(dirname "$0")/.."
RAIZ="$(pwd)"
paso "Directorio del backend: $RAIZ"

# ---------------------------------------------------------------- 1. sistema
paso "1/6  Dependencias del sistema"
if   command -v dnf >/dev/null 2>&1; then GESTOR=dnf
elif command -v apt-get >/dev/null 2>&1; then GESTOR=apt
elif command -v yum >/dev/null 2>&1; then GESTOR=yum
else morir "No reconozco el gestor de paquetes (esperaba dnf, yum o apt)."; fi
ok "gestor detectado: $GESTOR"

if [[ $GESTOR == apt ]]; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3 python3-venv python3-pip curl >/dev/null
else
  sudo "$GESTOR" install -y -q python3.11 python3.11-pip curl >/dev/null 2>&1 \
    || sudo "$GESTOR" install -y -q python3 python3-pip curl >/dev/null
fi

# onnxruntime y el codigo usan sintaxis de Python 3.10+
PY=""
for cand in python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PY="$cand"; break
    fi
  fi
done
[[ -n "$PY" ]] || morir "Necesito Python 3.10 o superior. Instala python3.11 y vuelve a correr."
ok "interprete: $PY ($($PY -V 2>&1))"

# ------------------------------------------------------------- 2. entorno
paso "2/6  Entorno virtual"
if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
  ok "creado .venv"
else
  ok ".venv ya existia, se reutiliza"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip wheel

# --------------------------------------------------------- 3. dependencias
paso "3/6  Dependencias de Python (sin PyTorch: ~60 MB en vez de ~2.5 GB)"
python -m pip install --quiet -r requirements.txt
ok "instaladas"
python - <<'PY'
import onnxruntime, fastapi, numpy, PIL
print(f"  onnxruntime {onnxruntime.__version__} | fastapi {fastapi.__version__} | numpy {numpy.__version__}")
print(f"  providers: {onnxruntime.get_available_providers()}")
PY

# -------------------------------------------------------------- 4. modelo
paso "4/6  Modelo"
if [[ ! -f modelos/best.onnx ]]; then
  aviso "Falta modelos/best.onnx"
  echo "     Subelo desde tu maquina con:"
  echo "       scp -i <tu-llave>.pem best.onnx ec2-user@<IP>:$RAIZ/modelos/"
  echo "     o descargalo:  wget -O modelos/best.onnx '<URL>'"
  morir "Sin modelo no puedo seguir."
fi
TAM=$(du -h modelos/best.onnx | cut -f1)
ok "modelos/best.onnx presente ($TAM)"

# ------------------------------------------------------- 5. configuracion
paso "5/6  Configuracion"
if [[ ! -f .env ]]; then
  cp .env.example .env
  NUCLEOS=$(nproc)
  sed -i "s/^HILOS_INFERENCIA=.*/HILOS_INFERENCIA=${NUCLEOS}/" .env
  ok ".env creado (HILOS_INFERENCIA=${NUCLEOS}, detectados ${NUCLEOS} vCPU)"
else
  ok ".env ya existia, no se toca"
fi
mkdir -p evidencias/curaduria

if [[ $CON_KAFKA -eq 1 ]]; then
  paso "5b   Redpanda (broker compatible con Kafka)"
  if ! command -v docker >/dev/null 2>&1; then
    if [[ $GESTOR == apt ]]; then sudo apt-get install -y -qq docker.io >/dev/null
    else sudo "$GESTOR" install -y -q docker >/dev/null; fi
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER" || true
    aviso "Docker recien instalado: si 'docker ps' pide permisos, cierra sesion y vuelve a entrar (o usa sudo)."
  fi
  DC="docker compose"; docker compose version >/dev/null 2>&1 || DC="docker-compose"
  sudo $DC up -d
  sed -i 's/^BUS_TIPO=.*/BUS_TIPO=kafka/' .env
  ok "Redpanda levantado · consola en http://<IP-PUBLICA>:8080"
fi

# ------------------------------------------------------- 6. prueba de humo
paso "6/6  Prueba de humo"
if python scripts/smoke_test.py; then
  ok "el sistema responde correctamente"
else
  morir "La prueba de humo fallo. Revisa la salida de arriba antes de seguir."
fi

IP_PUB=$(curl -s --max-time 3 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "")
[[ -z "$IP_PUB" ]] && IP_PUB="<IP-PUBLICA>"

cat <<FIN

$(echo -e "${VERDE}=============================================================${OFF}")
 LISTO. Arranca el servicio con:

     ./scripts/run_all.sh

 Y desde tu maquina / telefono:

     API        http://${IP_PUB}:8000
     Dashboard  http://${IP_PUB}:8000/
     Swagger    http://${IP_PUB}:8000/docs

 En la app Expo pon esta URL base:  http://${IP_PUB}:8000

 RECORDATORIO: la IP publica cambia cada vez que reinicias el Learner Lab.
$(echo -e "${VERDE}=============================================================${OFF}")
FIN
