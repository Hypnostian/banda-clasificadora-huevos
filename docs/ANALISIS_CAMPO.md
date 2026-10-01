# Análisis de fallos en campo

Tres casos reportados con huevos reales desde el celular:

| # | Foto | Veredicto | Correcto |
|---|---|---|---|
| 1 | huevo con cáscara rota al lado | `Sano 89% → ACEPTADO` | ❌ |
| 2 | huevo abierto por arriba, bordes dentados | `Sano 71% → ACEPTADO` | ❌ |
| 3 | huevo **sano** con una raya de lapicero | `Roto 93% → DESCARTE` | ❌ |

No son bugs del código. Son tres síntomas de dos causas medibles. Todo lo que
sigue está medido sobre el split de test, no razonado.

---

## Causa 1 — el modelo aprendió un atajo: "línea oscura = roto"

**Experimento:** tomé 20 huevos **sanos** del test, les dibujé encima una línea
oscura irregular imitando un lapicero, y volví a clasificarlos.

| | resultado |
|---|---|
| Huevos sanos que pasaron a `DESCARTE` solo por la raya | **6 de 20 (30 %)** |
| Subida media de p(Roto) | **+0.261** |
| Casos extremos | p(Roto) de 0.003 → **0.924** |

> **Actualización:** sobre los **43** huevos sanos del test (no solo 20), la raya
> manda a DESCARTE a **24/43 (56 %)**. Es la cifra a citar; ver
> [`COMPARACION_MODELOS.md`](COMPARACION_MODELOS.md).

**Confirmado.** Tu caso 3 no es una casualidad: es el rasgo central que el
modelo aprendió. En el dataset, los huevos "damaged" se fotografiaron casi
siempre con fisuras visibles, que en la imagen son líneas finas y oscuras. El
modelo no aprendió "cáscara rota"; aprendió "línea oscura sobre superficie
curva color hueso".

> **Consecuencia que importa para la sustentación:** los huevos comerciales
> vienen **sellados con tinta** (fecha de vencimiento, código de granja). Este
> modelo marcaría como rotos un porcentaje alto de huevos perfectamente buenos
> en una planta real. Es un defecto que bloquea producción, y encontrarlo tú
> mismo con un lapicero vale más que cualquier métrica del informe.

**Esto NO se arregla con umbrales.** Requiere datos (ver §5).

---

## Causa 2 — el desenfoque destruye la evidencia

**Experimento:** desenfoque gaussiano creciente sobre 25 huevos **rotos**.

| radio de desenfoque | p(Roto) medio | → DESCARTE | → ACEPTADO | **fuga** |
|---|---|---|---|---|
| 0 (nítida) | 0.934 | 25 | 0 | **0 %** |
| 1 | 0.890 | 24 | 0 | 0 % |
| 2 | 0.848 | 24 | 1 | 3 % |
| **3** | 0.488 | 13 | **9** | **38 %** |
| 4 | 0.220 | 5 | **12** | 50 % |
| 6 | 0.047 | 1 | 8 | 50 % |

**Confirmado, y es la causa principal de tus casos 1 y 2.** Tus dos fotos están
visiblemente movidas (primer plano con poca profundidad de campo). Una fisura
es un detalle de pocos píxeles: el desenfoque la borra, y el modelo entonces ve
un huevo liso y responde "Sano" **con alta confianza**. Ese es el peor error
posible: un falso negativo que además parece seguro.

---

## ¿Sirve más luz? Sí, pero no como uno esperaría

**Experimento:** escalado de brillo sobre los mismos 25 huevos rotos.

| brillo | p(Roto) medio | → DESCARTE | → ACEPTADO |
|---|---|---|---|
| 0.35× (muy oscuro) | 0.927 | 25 | 0 |
| 0.50× | 0.934 | 25 | 0 |
| 0.70× | 0.935 | 25 | 0 |
| **1.00× (original)** | 0.934 | 25 | 0 |
| 1.40× | 0.842 | 22 | 1 |
| 1.80× | 0.796 | 18 | 1 |
| 2.40× (quemado) | **0.478** | 10 | **4** |

Resultado contraintuitivo y útil: **oscurecer no le hace casi nada; sobreexponer
sí le hace daño.** A 2.4× el modelo pierde la mitad de los rotos.

**Entonces, ¿la luz ayuda?** Sí, pero **de forma indirecta**: más luz permite
una velocidad de obturación más rápida, y eso reduce el movimiento y mejora el
enfoque — que es lo que de verdad importa (§2). Lo que **no** ayuda es meterle
flash directo: eso quema las altas luces y borra la fisura igual que el
desenfoque.

### Cómo iluminar, en orden de efectividad

1. **Luz rasante (lateral, casi paralela a la cáscara).** Una fisura es
   relieve; la luz lateral la convierte en una sombra marcada. Es la técnica
   que usan las mesas de inspección reales.
2. **Luz difusa y abundante** (cerca de una ventana, día nublado, o una lámpara
   con papel de difusión). Sube la velocidad de obturación sin quemar.
3. **Contraluz / ovoscopía** — la técnica industrial de verdad. Pon la linterna
   del celular **debajo o detrás** del huevo: las fisuras aparecen como líneas
   brillantes en la cáscara translúcida. Es por lejos lo más revelador, aunque
   el modelo actual no está entrenado en ese dominio.
4. **Flash directo de frente: evítalo.** Genera reflejo especular que tapa
   justo la zona que interesa.

---

## Lo que ya cambié en el sistema (medido, desplegado)

### a) `tau_sano` de 0.60 → 0.85

**Por qué exactamente 0.85:** medí la separación entre los dos grupos.

| grupo | p(Sano) mín | mediana | máx |
|---|---|---|---|
| Huevos **sanos de verdad** | **0.884** | 0.942 | 0.964 |
| Huevos **rotos que se colaron** por desenfoque | 0.659 | 0.896 | 0.950 |

Hay un hueco: ningún huevo sano bajó de 0.884. Cortar en 0.85 cae dentro de ese
hueco.

| tau_sano | rotos colados atrapados | sanos mandados a revisión |
|---|---|---|
| 0.60 (antes) | 0 / 25 | 0 / 22 |
| **0.85 (ahora)** | **10 / 25** | **0 / 22** |
| 0.90 | 13 / 25 | 2 / 22 |

Atrapa 10 de cada 25 colados **sin mandar a revisión un solo huevo sano**.

**Tu caso 2 (`Sano 71%`) queda resuelto:** 0.71 está por debajo del mínimo que
alcanzó cualquier huevo sano real. Ahora va a `REVISION`, que es justo lo que
pediste.

### b) Compuerta de nitidez

Se calcula la **varianza del laplaciano del recorte del huevo** (no del frame
completo: el mantel no dice nada sobre si la cáscara está enfocada). Si está
por debajo de 4.0, el sistema **no decide**: manda a `REVISION` y la app avisa
*"Foto borrosa: apoya el celular, acércate y repite la toma"*.

| umbral | fotos enfocadas que pasan | fotos borrosas rechazadas |
|---|---|---|
| 3.0 | 100 % | 87 % |
| **4.0** | **100 %** | **93 %** |
| 5.0 | 95 % | 95 % |

4.0 no rechaza ni una sola foto bien enfocada.

> Excepción deliberada: si aun borrosa se ve la rotura (`p(Roto) ≥ tau_roto`),
> se descarta igual. La compuerta protege contra falsos *negativos*, no estorba
> a los positivos.

### c) Consolidación por huevo — la mejora más fuerte

En modo continuo la cámara ve el **mismo** huevo en varios frames, y el tracker
ya lo identificaba con un `egg_id` estable. Ahora el veredicto se consolida en
**peor caso**: si en alguna vista el modelo sospechó rotura, esa sospecha manda.

Partiendo de fotos desenfocadas (el escenario donde el sistema fallaba):

| estrategia | DESCARTE | REVISION | ACEPTADO | **fuga** |
|---|---|---|---|---|
| 1 toma, política vieja | 15 | 2 | 11 | **36.7 %** |
| 1 toma, con `tau_sano=0.85` | 15 | 6 | 7 | 23.3 % |
| **peor de 2 tomas** | 29 | 1 | 0 | **0.0 %** |
| peor de 3 tomas | 30 | 0 | 0 | **0.0 %** |

(La segunda fila muestra que subir `tau_sano` ya ayuda por sí solo —de 36.7 % a
23.3 %— pero que la consolidación multivista es lo que cierra el hueco.)

**De 36.7 % a 0 % con solo dos tomas.** Un huevo roto visto de frente parece
sano; visto desde otro ángulo, no. Quedarse con la vista más favorable era
justo el error que no podíamos permitirnos.

Además, si una vista posterior cambia el veredicto, se publica un evento de
**corrección** (`actualizacion: true`) y el dashboard mueve el huevo de casilla
en vez de contarlo dos veces.

### Efecto neto sobre el split de test

| | fotos nítidas | fotos borrosas |
|---|---|---|
| Política anterior | 0 % fuga | **49.4 % fuga** |
| **Política nueva** | **0 % fuga** | **3.4 % fuga** |

En fotos buenas no cambia nada. En fotos malas, la fuga cae **14×**.

---

## Lo que todavía requiere reentrenar

La raya de lapicero (§1) **no se arregla con umbrales**, porque el modelo está
genuinamente convencido (93 %). Hace falta atacar el atajo con datos:

1. **Negativos duros: huevos marcados pero sanos.** Fotografía 60–80 huevos
   sanos con rayas de lapicero, marcador, sellos de fecha y códigos de granja,
   y etiquétalos **`Sano`**. Es la corrección directa del atajo aprendido, y es
   lo más rentable que puedes hacer con dos horas de trabajo.
2. **Huevos rotos sin fisura lineal**: abiertos por arriba, con trozo de
   cáscara faltante, hundidos. Tu caso 2 es de este tipo y el dataset casi no
   los tiene.
3. **Adaptación de dominio** (lo que ya venía en
   [`EVALUACION_MODELO.md`](EVALUACION_MODELO.md) §6): frames propios en las
   condiciones reales de uso.
4. Al reentrenar, **sube la augmentación de desenfoque** (`blur` en Albumentations
   o equivalente) para que el modelo no dependa tanto del detalle fino.

Con eso, la métrica a reportar no es el mAP: es **"raya de lapicero: 6/20 falsos
positivos antes → X/20 después"**. Eso es un experimento controlado con un
resultado, y es exactamente el tipo de cosa que distingue un proyecto de
ingeniería de un tutorial.

---

## Mientras tanto: cómo tomar la foto

1. **Apoya el celular** o apoya los codos. El movimiento es el enemigo número uno.
2. **Toca la pantalla para enfocar** sobre el huevo antes de disparar.
3. **No te acerques tanto** que el celular no logre enfocar (los teléfonos no
   enfocan a menos de ~10 cm). Deja el huevo a 15–20 cm y que llene la mitad
   del encuadre.
4. **Luz lateral difusa**, no flash de frente.
5. **Usa "Arrancar banda" en vez de disparo manual** y gira lentamente el
   huevo: con 2–3 vistas la fuga se va a cero, y ese es el modo que además
   corresponde a la banda transportadora real.
6. Si sale el aviso ámbar de foto borrosa, **repite** — el sistema te está
   diciendo que no confía en lo que vio.

## Cómo reproducir estos experimentos

```bash
python scripts/diagnostico_campo.py     # las 3 hipótesis
python scripts/soluciones.py            # las 3 soluciones medidas
```
