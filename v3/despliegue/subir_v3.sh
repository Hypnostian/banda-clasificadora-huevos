#!/usr/bin/env bash
# =============================================================================
#  Sube la v3 al servidor y la deja corriendo en el puerto 8000.
#  Correr desde la raiz del proyecto (Git Bash):   bash v3/despliegue/subir_v3.sh
#  El modelo 1 queda intacto en ~/huevos, de respaldo en el puerto 8001.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
SERVIDOR=ubuntu@34.236.46.110
LLAVE=llaveproyectos.pem
OPC=(-i "$LLAVE" -o ConnectTimeout=15 -o BatchMode=yes)
MODELO=v3/entrenamientov3/weights/best.onnx

echo "==> 1/4 codigo del backend (sin .env, entorno, logs, evidencias ni el modelo 1)"
tar czf - --exclude='__pycache__' --exclude='.venv' --exclude='./.env' --exclude='./logs' \
    --exclude='./evidencias' --exclude='./modelos' -C backend . \
  | ssh "${OPC[@]}" "$SERVIDOR" 'mkdir -p ~/huevos_v3/backend/modelos && tar xzf - -C ~/huevos_v3/backend'

echo "==> 2/4 modelo v3 ($(du -h "$MODELO" | cut -f1)) con verificacion sha256"
LOCAL=$(sha256sum "$MODELO" | cut -c1-64)
REMOTO=$(ssh "${OPC[@]}" "$SERVIDOR" 'sha256sum ~/huevos_v3/backend/modelos/best.onnx 2>/dev/null | cut -c1-64' || true)
if [ "$LOCAL" != "$REMOTO" ]; then
  scp "${OPC[@]}" "$MODELO" "$SERVIDOR:~/huevos_v3/backend/modelos/best.onnx"
  REMOTO=$(ssh "${OPC[@]}" "$SERVIDOR" 'sha256sum ~/huevos_v3/backend/modelos/best.onnx | cut -c1-64')
fi
[ "$LOCAL" = "$REMOTO" ] && echo "    sha256 identico: ${LOCAL:0:12}" || { echo "    ERROR: el modelo llego distinto"; exit 1; }

echo "==> 3/4 metricas del entrenamiento, informe y scripts de operacion"
ssh "${OPC[@]}" "$SERVIDOR" 'mkdir -p ~/huevos_v3/modelo_v3 ~/huevos_v3/docs'
scp "${OPC[@]}" v3/entrenamientov3/{metricas_v3.json,args.yaml,results.csv,comparacion_v1_v3.json,prueba_contexto.json} \
    "$SERVIDOR:~/huevos_v3/modelo_v3/"
scp "${OPC[@]}" docs/COMPARACION_V1_V3.md "$SERVIDOR:~/huevos_v3/docs/"
scp "${OPC[@]}" v3/despliegue/LEEME.md "$SERVIDOR:~/huevos_v3/README.md"
scp "${OPC[@]}" v3/despliegue/{setup_v3_aws.sh,estado.sh,cambiar_modelo.sh} "$SERVIDOR:~/huevos_v3/"
ssh "${OPC[@]}" "$SERVIDOR" 'sed -i "s/\r$//" ~/huevos_v3/*.sh && chmod +x ~/huevos_v3/*.sh'

echo "==> 4/4 instalar y arrancar en el servidor"
ssh "${OPC[@]}" "$SERVIDOR" 'bash ~/huevos_v3/setup_v3_aws.sh'
