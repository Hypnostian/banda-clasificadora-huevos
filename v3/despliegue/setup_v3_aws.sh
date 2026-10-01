#!/usr/bin/env bash
# =============================================================================
#  Despliegue de la v3 en el servidor (AWS Academy, Ubuntu 24.04, usuario ubuntu).
#
#  Deja corriendo:
#    ~/huevos_v3  -> v3 (YOLOv8l) en el puerto 8000: el que usa la app
#    ~/huevos     -> modelo 1 (YOLOv8m) SIN TOCAR, en el puerto 8001 de respaldo
#
#  Lo lanza v3/despliegue/subir_v3.sh despues de copiar los archivos. Se puede
#  repetir sin romper nada.
# =============================================================================
set -euo pipefail
V3="$HOME/huevos_v3"; B3="$V3/backend"; B1="$HOME/huevos/backend"
paso() { echo -e "\n\033[0;32m==> $*\033[0m"; }

paso "1/6 Entorno de Python con las MISMAS versiones que el modelo 1 (probadas en este servidor)"
[ -x "$B3/.venv/bin/python" ] || python3 -m venv "$B3/.venv"
"$B1/.venv/bin/pip" freeze > /tmp/requisitos_modelo1.txt
"$B3/.venv/bin/pip" install -q --upgrade pip
"$B3/.venv/bin/pip" install -q -r /tmp/requisitos_modelo1.txt
"$B3/.venv/bin/python" -c "import onnxruntime, fastapi; print('   onnxruntime', onnxruntime.__version__, '| fastapi', fastapi.__version__)"
mkdir -p "$B3/logs" "$B3/evidencias/curaduria" "$B3/modelos"
[ -d "$V3/huevos" ] || cp -r "$HOME/huevos/huevos" "$V3/"     # split test para banda-smoke / banda-eval

paso "2/6 .env de la v3: el del modelo 1 con los umbrales de la v3 (docs/COMPARACION_V1_V3.md)"
python3 - "$B1/.env" "$B3/.env" <<'PY'
import sys
origen, destino = sys.argv[1], sys.argv[2]
v3 = {"MODELO_PATH": "modelos/best.onnx", "MODELO_NOMBRE": "v3 YOLOv8l", "CONF_MIN_DETECCION": "0.15",
      "TAU_ROTO": "0.35", "TAU_SANO": "0.75", "TAU_REVISION": "0.07"}
lineas, vistas = [], set()
for ln in open(origen, encoding="utf-8").read().splitlines():
    k = ln.split("=", 1)[0].strip()
    if k in v3:
        lineas.append(f"{k}={v3[k]}"); vistas.add(k)
    else:
        lineas.append(ln)
faltan = [f"{k}={v}" for k, v in v3.items() if k not in vistas]
if faltan:
    lineas += ["", "# --- v3: descarte si p(Roto)>=0.35, revision si 0.07<=p(Roto)<0.35"] + faltan
open(destino, "w", encoding="utf-8").write("\n".join(lineas) + "\n")
PY
grep -E "^(MODELO_|TAU_|CONF_)" "$B3/.env" | sed 's/^/   /'

paso "3/6 Modelo 1 -> puerto 8001 (su carpeta queda intacta)"
sudo sed -i -E 's/--port [0-9]+/--port 8001/; s/^Description=.*/Description=Banda de huevos - MODELO 1 (YOLOv8m), respaldo/' \
    /etc/systemd/system/banda-huevos.service

paso "4/6 v3 -> puerto 8000, arranque automatico y rotacion de logs"
sudo tee /etc/systemd/system/banda-huevos-v3.service >/dev/null <<UNIT
[Unit]
Description=Banda de huevos - v3 (YOLOv8l + ONNX + FastAPI)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=$B3
ExecStart=$B3/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
Restart=always
RestartSec=5
TimeoutStartSec=180
KillSignal=SIGINT
StandardOutput=append:$B3/logs/api.log
StandardError=append:$B3/logs/api.log

[Install]
WantedBy=multi-user.target
UNIT
sudo tee /etc/logrotate.d/banda-huevos-v3 >/dev/null <<ROT
$B3/logs/*.log {
    weekly
    rotate 4
    maxsize 50M
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su ubuntu ubuntu
}
ROT
sudo systemctl daemon-reload
sudo systemctl stop banda-huevos banda-huevos-v3 2>/dev/null || true
sudo systemctl enable --now banda-huevos-v3
sudo systemctl start banda-huevos

paso "5/6 Esperando a que respondan las dos APIs"
for puerto in 8000 8001; do
  for _ in $(seq 1 60); do curl -s -m 2 "http://127.0.0.1:$puerto/salud" >/dev/null && break; sleep 2; done
  echo "   :$puerto -> $(curl -s -m 3 "http://127.0.0.1:$puerto/salud" || echo SIN RESPUESTA)"
done

paso "6/6 Atajos de consola y estado al entrar por SSH"
chmod +x "$V3/estado.sh" "$V3/cambiar_modelo.sh"
[ -f "$HOME/.bashrc.antes_v3" ] || cp "$HOME/.bashrc" "$HOME/.bashrc.antes_v3"
python3 - "$HOME/.bashrc" <<'PY'
import sys
ruta = sys.argv[1]
lineas = open(ruta, encoding="utf-8").read().splitlines()
ini = next((i for i, l in enumerate(lineas) if l.startswith("# ===== BANDA HUEVOS")), None)
if ini is not None:
    # bloque nuevo: hasta su marca FIN; bloque viejo (sin marca): hasta la linea que muestra el estado
    fin = next((i for i in range(ini, len(lineas)) if lineas[i].startswith("# ===== FIN BANDA HUEVOS")), None)
    if fin is None:
        fin = next((i for i in range(ini, len(lineas)) if "BANDA_MOSTRADO" in lineas[i]), ini)
    del lineas[ini:fin + 1]
lineas += [
    "# ===== BANDA HUEVOS (v3 en :8000 para la app, modelo 1 de respaldo en :8001) =====",
    "alias banda='~/huevos_v3/estado.sh'",
    "alias banda-log='tail -n 40 -f ~/huevos_v3/backend/logs/api.log'",
    "alias banda-restart='sudo systemctl restart banda-huevos-v3 && sleep 10 && ~/huevos_v3/estado.sh'",
    "alias banda-stop='sudo systemctl stop banda-huevos-v3'",
    "alias banda-start='sudo systemctl start banda-huevos-v3'",
    "alias banda-demo='cd ~/huevos_v3/backend && .venv/bin/python scripts/simular_banda.py --api http://127.0.0.1:8000 --hpm 60 --n 40'",
    "alias banda-eval='cd ~/huevos_v3/backend && .venv/bin/python scripts/evaluar_test.py --split test'",
    "alias banda-smoke='cd ~/huevos_v3/backend && .venv/bin/python scripts/smoke_test.py'",
    "alias banda1-log='tail -n 40 -f ~/huevos/backend/logs/api.log'",
    "alias banda1-restart='sudo systemctl restart banda-huevos'",
    "alias banda-cambiar='~/huevos_v3/cambiar_modelo.sh'",
    '[[ $- == *i* ]] && [[ -z "$BANDA_MOSTRADO" ]] && export BANDA_MOSTRADO=1 && ~/huevos_v3/estado.sh',
    "# ===== FIN BANDA HUEVOS =====",
]
open(ruta, "w", encoding="utf-8").write("\n".join(lineas) + "\n")
PY
"$V3/estado.sh"
