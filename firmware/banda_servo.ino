/*
 * Actuador de la banda clasificadora de huevos.
 * Arduino UNO / Nano / ESP32 + servo SG90 + 3 LEDs.
 *
 * Protocolo serial (texto, una linea por orden):
 *    D:<egg_id>\n   DESVIAR  -> el servo empuja el huevo a la canasta de rotos
 *    P:<egg_id>\n   PASAR    -> sigue derecho al empaque (LED verde)
 *    M:<egg_id>\n   MARCAR   -> revision manual (LED ambar)
 * Responde:
 *    OK:<egg_id>\n
 *
 * Hardware minimo para que la demo sea fisica y no cueste casi nada:
 *    - 1 servo SG90 en el pin 9 (alimentarlo con 5V EXTERNOS, no del Arduino:
 *      el pico de corriente al mover reinicia la placa)
 *    - LED verde pin 4, ambar pin 5, rojo pin 6 (con resistencia de 220 ohm)
 *    - GND comun entre la fuente del servo y el Arduino
 *
 * Nota de ingenieria: el barrido del servo NO usa delay() bloqueante largo,
 * para que la placa pueda seguir leyendo el serial mientras el brazo vuelve.
 */

#include <Servo.h>

const uint8_t PIN_SERVO = 9;
const uint8_t PIN_VERDE = 4;
const uint8_t PIN_AMBAR = 5;
const uint8_t PIN_ROJO  = 6;

const int ANGULO_REPOSO  = 15;
const int ANGULO_DESVIO  = 95;
const unsigned long MS_DESVIO = 180;   // tiempo con el brazo afuera
const unsigned long MS_LED    = 400;   // cuanto queda encendido el indicador

Servo brazo;

bool desviando = false;
unsigned long tDesvio = 0;
unsigned long tLed = 0;
uint8_t ledActivo = 0;

char buffer[64];
uint8_t n = 0;

void setup() {
  Serial.begin(115200);
  brazo.attach(PIN_SERVO);
  brazo.write(ANGULO_REPOSO);
  pinMode(PIN_VERDE, OUTPUT);
  pinMode(PIN_AMBAR, OUTPUT);
  pinMode(PIN_ROJO, OUTPUT);
  // Parpadeo de arranque: confirma visualmente que la placa esta viva.
  for (uint8_t i = 0; i < 2; i++) {
    digitalWrite(PIN_VERDE, HIGH); digitalWrite(PIN_AMBAR, HIGH); digitalWrite(PIN_ROJO, HIGH);
    delay(120);
    digitalWrite(PIN_VERDE, LOW); digitalWrite(PIN_AMBAR, LOW); digitalWrite(PIN_ROJO, LOW);
    delay(120);
  }
  Serial.println("LISTO:banda-servo-1.0");
}

void encender(uint8_t pin) {
  digitalWrite(PIN_VERDE, LOW);
  digitalWrite(PIN_AMBAR, LOW);
  digitalWrite(PIN_ROJO, LOW);
  digitalWrite(pin, HIGH);
  ledActivo = pin;
  tLed = millis();
}

void procesar(char *linea) {
  char accion = linea[0];
  const char *id = (linea[1] == ':') ? linea + 2 : "sin-id";

  switch (accion) {
    case 'D':
      brazo.write(ANGULO_DESVIO);
      desviando = true;
      tDesvio = millis();
      encender(PIN_ROJO);
      break;
    case 'M':
      encender(PIN_AMBAR);
      break;
    case 'P':
    default:
      encender(PIN_VERDE);
      break;
  }
  Serial.print("OK:");
  Serial.println(id);
}

void loop() {
  // --- lectura no bloqueante del serial ---
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (n > 0) { buffer[n] = '\0'; procesar(buffer); n = 0; }
    } else if (n < sizeof(buffer) - 1) {
      buffer[n++] = c;
    }
  }

  // --- retorno del brazo ---
  if (desviando && millis() - tDesvio >= MS_DESVIO) {
    brazo.write(ANGULO_REPOSO);
    desviando = false;
  }

  // --- apagado del indicador ---
  if (ledActivo && millis() - tLed >= MS_LED) {
    digitalWrite(ledActivo, LOW);
    ledActivo = 0;
  }
}
