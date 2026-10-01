#!/usr/bin/env bash
# Estado de la banda: que version atiende a la app (:8000) y cual queda de respaldo.
V='\033[0;32m'; R='\033[0;31m'; A='\033[0;33m'; C='\033[0;36m'; O='\033[0m'
IP_FIJA=34.236.46.110
TOKEN=$(curl -s -X PUT --max-time 2 "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null)
IP=$(curl -s --max-time 2 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null)
[ -z "$IP" ] && IP=$IP_FIJA
puerto_de() { grep -oE -- '--port [0-9]+' "/etc/systemd/system/$1.service" 2>/dev/null | awk '{print $2}'; }

echo -e "\n${C}=============================================================${O}"
echo -e "${C}  BANDA CLASIFICADORA DE HUEVOS  ·  ONNX + FastAPI${O}"
echo -e "${C}=============================================================${O}"
for par in "banda-huevos-v3|v3 (YOLOv8l)     " "banda-huevos|modelo 1 (YOLOv8m)"; do
  svc=${par%%|*}; nombre=${par#*|}; p=$(puerto_de "$svc")
  if systemctl is-active --quiet "$svc"; then est="${V}ACTIVO${O}"; else est="${R}CAIDO ${O}"; fi
  if [ "$p" = "8000" ]; then rol="${V}<- la app usa este${O}"; else rol="respaldo (solo local)"; fi
  echo -e "  $nombre $est :$p  $rol"
  S=$(curl -s --max-time 4 "http://127.0.0.1:$p/salud" 2>/dev/null)
  [ -n "$S" ] && echo "$S" | python3 -c "import json,sys;d=json.load(sys.stdin);print(f\"      modelo {d['modelo']} · uptime {d['uptime_s']:.0f}s\")" 2>/dev/null
done
P=$(curl -s --max-time 4 http://127.0.0.1:8000/politica 2>/dev/null)
[ -n "$P" ] && echo "$P" | python3 -c "import json,sys;d=json.load(sys.stdin);print(f\"  politica   descarte p(Roto)>={d['tau_roto']} · revision desde {d.get('tau_revision')} · acepta p(Sano)>={d['tau_sano']}\")" 2>/dev/null
M=$(curl -s --max-time 4 http://127.0.0.1:8000/metricas 2>/dev/null)
[ -n "$M" ] && echo "$M" | python3 -c "import json,sys;d=json.load(sys.stdin);print(f\"  turno      {d['total_huevos']} huevos · rotura {d['tasa_rotura_pct']}% · revision {d['tasa_revision_pct']}%\")" 2>/dev/null
if timeout 4 curl -s -o /dev/null "http://${IP}:8000/salud" 2>/dev/null; then
  echo -e "  internet   ${V}puerto 8000 ABIERTO${O}  ->  http://${IP}:8000"
else
  echo -e "  internet   ${A}puerto 8000 CERRADO${O} en el Security Group"
fi
echo -e "${C}-------------------------------------------------------------${O}"
echo "  Dashboard  http://${IP}:8000/      Swagger  http://${IP}:8000/docs"
echo "  banda-log · banda-restart · banda-smoke · banda-demo · banda-eval   (version en :8000)"
echo "  banda-cambiar 1   pone el modelo 1 en :8000   ·   banda-cambiar v3   vuelve a la v3"
echo -e "${C}=============================================================${O}\n"
