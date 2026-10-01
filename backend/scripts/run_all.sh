#!/usr/bin/env bash
# Arranque unico del backend de la banda.
#
#     ./scripts/run_all.sh              # API con consumidores embebidos
#     ./scripts/run_all.sh --workers    # API + los 3 consumidores como procesos aparte
#     PUERTO=8080 ./scripts/run_all.sh  # otro puerto
#
# Ctrl-C baja todo.
set -euo pipefail
cd "$(dirname "$0")/.."

PUERTO="${PUERTO:-8000}"
SEPARADOS=0
[[ "${1:-}" == "--workers" ]] && SEPARADOS=1

[[ -d .venv ]] || { echo "No hay .venv. Corre primero ./scripts/setup_aws.sh"; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate
[[ -f modelos/best.onnx ]] || { echo "Falta modelos/best.onnx"; exit 1; }

PIDS=()
limpiar() {
  echo -e "\nBajando servicios..."
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  exit 0
}
trap limpiar INT TERM

if [[ $SEPARADOS -eq 1 ]]; then
  grep -q '^BUS_TIPO=kafka' .env 2>/dev/null \
    || echo "AVISO: BUS_TIPO no es kafka; los workers separados no recibiran eventos."
  export WORKERS_EMBEBIDOS=false
  mkdir -p logs
  for rol in actuador metricas curaduria; do
    python workers/worker.py --rol "$rol" >"logs/worker-$rol.log" 2>&1 &
    PIDS+=($!)
    echo "  worker $rol  pid ${PIDS[-1]}  -> logs/worker-$rol.log"
  done
  sleep 1
fi

IP=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
echo "-------------------------------------------------------------"
echo "  Dashboard : http://${IP:-localhost}:${PUERTO}/"
echo "  Swagger   : http://${IP:-localhost}:${PUERTO}/docs"
echo "  URL para la app Expo: http://${IP:-localhost}:${PUERTO}"
echo "-------------------------------------------------------------"

# Un solo worker de uvicorn a proposito: onnxruntime ya usa todos los cores.
# Varios workers se pelean la CPU y suben la p95 sin subir el throughput.
exec uvicorn app.main:app --host 0.0.0.0 --port "$PUERTO" --workers 1 --log-level info
