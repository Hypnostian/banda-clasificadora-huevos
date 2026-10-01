#!/usr/bin/env bash
# Pone en el puerto 8000 (el que usa la app) la version pedida y deja la otra en 8001.
#   ~/huevos_v3/cambiar_modelo.sh v3     (lo normal)
#   ~/huevos_v3/cambiar_modelo.sh 1      (volver al modelo 1 si algo falla en la demo)
set -euo pipefail
case "${1:-}" in
  v3) P3=8000; P1=8001 ;;
  1)  P3=8001; P1=8000 ;;
  *)  echo "uso: $0 v3|1"; exit 1 ;;
esac
sudo sed -i -E "s/--port [0-9]+/--port $P3/" /etc/systemd/system/banda-huevos-v3.service
sudo sed -i -E "s/--port [0-9]+/--port $P1/" /etc/systemd/system/banda-huevos.service
sudo systemctl daemon-reload
sudo systemctl stop banda-huevos banda-huevos-v3
sudo systemctl start banda-huevos-v3 banda-huevos
for puerto in 8000 8001; do
  for _ in $(seq 1 60); do curl -s -m 2 "http://127.0.0.1:$puerto/salud" >/dev/null && break; sleep 2; done
done
"$(dirname "$0")/estado.sh"
