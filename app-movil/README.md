# App cámara de la banda

React Native + Expo (managed). El teléfono hace de **cámara y sensor de
presencia** de la banda: captura, comprime y manda el frame al backend, que
devuelve la ruta de cada huevo.

Mismo patrón del proyecto de detección de placas:
`foto → expo-image-manipulator → multipart/form-data → FastAPI`.

## Correr

```bash
npm install
npx expo start
```

Escanear el QR con **Expo Go**. No se necesita Android Studio
(ver `docs/DECISIONES.md`, ADR-02).

Si el teléfono no ve el QR o está en otra red:

```bash
npx expo start --tunnel
```

## Configurar

Tocar el **engranaje** arriba a la derecha:

- **URL del backend** — `http://<IP>:8000`. En AWS Academy **la IP pública
  cambia cada sesión**, por eso es un campo y no una constante. Se guarda con
  `AsyncStorage`. El botón *Probar y guardar* llama a `/salud` antes de cerrar:
  si no conecta, te lo dice ahí y no en medio de la demo.
- **Velocidad de la banda (ms)** — cada cuánto dispara el modo continuo.
  2000 ms = 30 huevos/min. No bajar de 500 ms: el backend tarda ~250-400 ms por
  frame y encolarías disparos.

## Modos

| Botón | Qué hace |
|---|---|
| **Disparo manual** | un frame por toque. Para probar huevo por huevo. |
| **Arrancar banda** | modo continuo: dispara cada N ms = la banda corriendo. |

El tick se re-arma **después** de cada respuesta (no con `setInterval`), así un
frame lento no acumula disparos encima del anterior.

## Lo que muestra

El veredicto sobre la cámara va siempre con **icono + etiqueta + color**
(`✓ ACEPTADO`, `⚠ REVISION`, `✗ DESCARTE`), nunca color solo: rojo y verde son
casi indistinguibles en deuteranopía.

Los contadores solo suman huevos marcados como `nuevo` por el backend; el
tracker del servidor ya descarta los frames repetidos del mismo huevo.

## Notas

- **No pongas `Content-Type` a mano** en la subida multipart. React Native tiene
  que generar el *boundary*; si lo fijas, FastAPI rechaza el request.
- Android bloquea HTTP en claro. `app.json` ya trae
  `expo-build-properties → usesCleartextTraffic: true` y el equivalente de iOS,
  porque el backend va por HTTP sin TLS.

## `Unsupported FormDataPart implementation` — resuelto

Si ves este error al subir una foto, la causa es la siguiente y ya está
corregida en [`src/api.js`](src/api.js).

Desde el **SDK 54**, Expo reemplaza el `fetch` global por su implementación
WinterCG (`expo/src/winter/fetch`). Ese `fetch` arma el multipart **en
JavaScript** y en `convertFormData.ts` solo acepta tres tipos de parte:

```
string   ·   Blob   ·   objeto con .bytes()
```

Cualquier otra cosa lanza `Unsupported FormDataPart implementation`. El propio
archivo de Expo lo dice: *"`uri` is not supported for React Native's FormData"*.

Es decir, el patrón clásico de React Native —el de todos los tutoriales, y el
que funcionaba hasta el SDK 53—

```js
formData.append('imagen', { uri, name, type });
await fetch(url, { method: 'POST', body: formData });   // ❌ revienta en SDK 54+
```

ya no sirve **con `fetch`**.

**La solución: `XMLHttpRequest`.** Expo reemplaza `fetch`, pero **no toca
`XMLHttpRequest`**. La red nativa de React Native sigue soportando la parte
`{ uri, name, type }`: `FormData.getParts()` la convierte en
`{ uri, headers, fieldName }` y el módulo nativo transmite el archivo
directamente desde el disco.

Ventaja adicional: el archivo **no se carga en memoria de JavaScript** (a
diferencia del camino `Blob`/`bytes()`), y se obtiene progreso de subida gratis
—que es lo que alimenta el indicador `subiendo 47%` sobre la cámara.

Otras salidas válidas, por si alguna vez hace falta:

| Alternativa | Cuándo |
|---|---|
| `new File(uri)` de `expo-file-system` + `fetch` | implementa `Blob`/`bytes()`; carga el archivo en memoria JS |
| `file.upload(url, { uploadType: MULTIPART })` | subida nativa con progreso, sin `FormData` |
