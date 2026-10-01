/**
 * Cliente del backend de la banda.
 *
 * ---------------------------------------------------------------------------
 * POR QUE LA SUBIDA USA XMLHttpRequest Y NO fetch
 * ---------------------------------------------------------------------------
 * Desde el SDK 54, Expo reemplaza el `fetch` global por su implementacion
 * WinterCG (`expo/src/winter/fetch`). Ese fetch arma el multipart en JS y en
 * `convertFormData.ts` solo acepta tres tipos de parte:
 *
 *      string  ·  Blob  ·  objeto con .bytes()
 *
 * y para cualquier otra cosa lanza:  "Unsupported FormDataPart implementation".
 * El propio archivo lo dice: "`uri` is not supported for React Native's
 * FormData". Es decir, el patron clasico de React Native
 *
 *      formData.append('imagen', { uri, name, type })
 *
 * — que es el que usan todos los tutoriales y el que funcionaba en SDK <= 53 —
 * revienta con el fetch de Expo.
 *
 * Expo NO reemplaza `XMLHttpRequest`. La red nativa de React Native sigue
 * soportando la parte `{ uri, name, type }`: `FormData.getParts()` la convierte
 * en `{uri, headers, fieldName}` y el modulo nativo transmite el archivo
 * directo desde el disco.
 *
 * Por eso la subida va por XHR. Ventaja extra: el archivo NO se carga en
 * memoria de JavaScript, y obtenemos progreso de subida gratis.
 *
 * Las peticiones sin archivo (GET /salud) siguen con fetch, que funciona bien.
 */

/** La v3 tarda ~1.3 s por frame en el servidor + red. */
const TIMEOUT_MS = 20000;

export const HOST_POR_DEFECTO = '34.236.46.110';

/** Versiones desplegadas en el servidor: una por puerto (ver v3/despliegue/LEEME.md). */
export const SERVIDORES = [
  { puerto: '8000', nombre: 'v3', detalle: 'YOLOv8l · la recomendada' },
  { puerto: '8001', nombre: 'Modelo 1', detalle: 'respaldo · solo si abriste el 8001 en AWS' },
];

export function normalizarUrl(url) {
  let u = (url || '').trim().replace(/\/+$/, '');
  if (u && !/^https?:\/\//i.test(u)) u = `http://${u}`;
  return u;
}

/** 'http://34.236.46.110:8000' -> { host: '34.236.46.110', puerto: '8000' } */
export function partirUrl(url) {
  const m = normalizarUrl(url).replace(/^https?:\/\//i, '').match(/^([^/:]+)(?::(\d+))?/);
  return { host: m ? m[1] : '', puerto: m && m[2] ? m[2] : '80' };
}

/** ('34.236.46.110', '8000') -> 'http://34.236.46.110:8000'. Tolera espacios, http:// o puerto pegados. */
export function armarUrl(host, puerto) {
  const h = (host || '').trim().replace(/^https?:\/\//i, '').replace(/[/:].*$/, '');
  const p = String(puerto || '').replace(/\D/g, '');
  return h ? `http://${h}${p ? `:${p}` : ''}` : '';
}

/**
 * GET /salud con limite de tiempo. Si el puerto esta cerrado en el Security Group la
 * conexion no falla, se queda colgada: por eso se aborta, y el rechazo tardio de fetch se
 * absorbe para que no salte despues como un error suelto.
 */
export async function verificarSalud(baseUrl, timeoutMs = 8000) {
  const url = `${normalizarUrl(baseUrl)}/salud`;
  const control = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const peticion = fetch(url, control ? { signal: control.signal } : undefined);
  peticion.catch(() => {});
  let reloj;
  const limite = new Promise((_, rechazar) => {
    reloj = setTimeout(() => {
      if (control) control.abort();
      rechazar(new Error(`no respondio en ${timeoutMs / 1000} s (¿puerto cerrado en AWS o servidor apagado?)`));
    }, timeoutMs);
  });
  try {
    const respuesta = await Promise.race([peticion, limite]);
    if (!respuesta.ok) throw new Error(`HTTP ${respuesta.status}`);
    return await respuesta.json();
  } finally {
    clearTimeout(reloj);
  }
}

/**
 * Manda un frame al backend por multipart nativo.
 *
 * @param {string}   baseUrl     http://34.236.46.110:8000
 * @param {string}   uriImagen   uri local del JPEG ya comprimido
 * @param {string}   estacion    identificador de la estacion de la banda
 * @param {function} onProgreso  opcional, recibe 0..1 mientras sube
 * @returns {Promise<object>}    ResultadoInspeccion
 */
export function inspeccionarFrame(baseUrl, uriImagen, estacion = 'banda-01', onProgreso) {
  return new Promise((resolver, rechazar) => {
    if (!uriImagen) {
      rechazar(new Error('No hay imagen que enviar'));
      return;
    }

    const formulario = new FormData();
    // Parte de archivo propietaria de React Native. La lee el modulo nativo,
    // no JavaScript. Ver la nota de arriba sobre por que esto exige XHR.
    formulario.append('imagen', {
      uri: uriImagen,
      name: `frame_${Date.now()}.jpg`,
      type: 'image/jpeg',
    });
    formulario.append('estacion', estacion);
    formulario.append('guardar_evidencia', 'false');

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${normalizarUrl(baseUrl)}/inspeccion`);
    xhr.timeout = TIMEOUT_MS;
    xhr.setRequestHeader('Accept', 'application/json');
    // OJO: no fijar Content-Type. React Native genera el boundary del
    // multipart por su cuenta; si lo fijamos a mano, FastAPI rechaza el body.

    if (onProgreso && xhr.upload) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgreso(e.loaded / e.total);
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolver(JSON.parse(xhr.responseText));
        } catch (err) {
          rechazar(new Error(`El servidor no devolvio JSON: ${String(xhr.responseText).slice(0, 120)}`));
        }
      } else {
        rechazar(new Error(`HTTP ${xhr.status} ${String(xhr.responseText).slice(0, 160)}`));
      }
    };

    xhr.onerror = () =>
      rechazar(
        new Error(
          `No se pudo alcanzar ${normalizarUrl(baseUrl)}. Revisa que el telefono tenga internet ` +
            `y que el puerto ${partirUrl(baseUrl).puerto} este abierto en el Security Group.`
        )
      );
    xhr.ontimeout = () => rechazar(new Error(`Tiempo de espera agotado (${TIMEOUT_MS / 1000}s)`));
    xhr.onabort = () => rechazar(new Error('Subida cancelada'));

    xhr.send(formulario);
  });
}
