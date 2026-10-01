/**
 * App camara de la banda transportadora.
 *
 * Dos modos:
 *   MANUAL     un disparo por toque (util para probar huevo por huevo)
 *   CONTINUO   dispara cada N ms = velocidad emulada de la banda
 *
 * El tick de la banda vive aqui a proposito: el telefono hace de sensor de
 * presencia + camara, igual que la fotocelda de una linea real.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator, Keyboard, KeyboardAvoidingView, Modal, Platform, Pressable, SafeAreaView,
  ScrollView, StyleSheet, Text, TextInput, View,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { StatusBar } from 'expo-status-bar';

import {
  HOST_POR_DEFECTO, SERVIDORES, armarUrl, inspeccionarFrame, normalizarUrl, partirUrl, verificarSalud,
} from './src/api';

const CLAVE_URL = '@banda/url';
const CLAVE_INTERVALO = '@banda/intervalo';

// IP elastica del servidor y puerto de la v3. Desde el engranaje se cambia de
// version con un toque (cada una corre en su puerto) o se escribe a mano.
const URL_POR_DEFECTO = armarUrl(HOST_POR_DEFECTO, SERVIDORES[0].puerto);

// Mismos roles de color que el dashboard. El color NUNCA va solo: siempre con
// icono y etiqueta, porque rojo y verde son casi identicos en deuteranopia.
const RUTA = {
  ACEPTADO: { color: '#0ca30c', icono: '✓', texto: 'ACEPTADO' },
  REVISION: { color: '#fab219', icono: '⚠', texto: 'REVISION' },
  DESCARTE: { color: '#d03b3b', icono: '✗', texto: 'DESCARTE' },
};

/** Reduce el frame antes de subirlo: el modelo trabaja a 640, mandar 4000 px
 *  de ancho solo gasta red. ~40-70 KB por frame en vez de ~2 MB.
 *
 *  API contextual de expo-image-manipulator (SDK 53+). La vieja
 *  `manipulateAsync` todavia existe en SDK 57 pero esta deprecada. */
async function comprimir(uri) {
  const contexto = ImageManipulator.manipulate(uri);
  contexto.resize({ width: 640 });
  const imagen = await contexto.renderAsync();
  const resultado = await imagen.saveAsync({ compress: 0.6, format: SaveFormat.JPEG });
  return resultado.uri;
}

export default function App() {
  const [permiso, pedirPermiso] = useCameraPermissions();
  const camara = useRef(null);
  const ocupado = useRef(false);
  const temporizador = useRef(null);

  const [url, setUrl] = useState('');                     // la guardada: a donde van las fotos
  const [host, setHost] = useState(HOST_POR_DEFECTO);     // formulario de ajustes
  const [puerto, setPuerto] = useState(SERVIDORES[0].puerto);
  const [intervalo, setIntervalo] = useState('2000');
  const [ajustesAbiertos, setAjustesAbiertos] = useState(false);
  const [salud, setSalud] = useState(null);
  const [probando, setProbando] = useState(false);
  const turno = useRef(0);

  const [continuo, setContinuo] = useState(false);
  const [enVuelo, setEnVuelo] = useState(false);
  const [fase, setFase] = useState('');
  const [ultimo, setUltimo] = useState(null);
  const [error, setError] = useState(null);
  const [conteo, setConteo] = useState({ ACEPTADO: 0, REVISION: 0, DESCARTE: 0, frames: 0 });

  // ------------------------------------------------------- configuracion
  // Cada prueba de conexion lleva su numero y solo cuenta la ULTIMA. Sin esto, la
  // prueba de un puerto cerrado (tarda 8 s en fallar) llegaba DESPUES de la del
  // puerto bueno y borraba la conexion: el "error" al pasar del 8001 al 8000.
  const probar = useCallback(async (destino, { cerrarSiConecta = false, abrirSiFalla = false } = {}) => {
    const mio = ++turno.current;
    setProbando(true);
    setError(null);
    try {
      const s = await verificarSalud(destino);
      if (mio !== turno.current) return;
      setSalud(s);
      if (cerrarSiConecta) setAjustesAbiertos(false);
    } catch (err) {
      if (mio !== turno.current) return;
      setSalud(null);
      setError(`No conecta con ${destino}: ${err.message}`);
      if (abrirSiFalla) setAjustesAbiertos(true);
    } finally {
      if (mio === turno.current) setProbando(false);
    }
  }, []);

  useEffect(() => {
    (async () => {
      const [u, i] = await Promise.all([
        AsyncStorage.getItem(CLAVE_URL),
        AsyncStorage.getItem(CLAVE_INTERVALO),
      ]);
      const inicial = normalizarUrl(u || URL_POR_DEFECTO);
      const partes = partirUrl(inicial);
      setUrl(inicial);
      setHost(partes.host);
      setPuerto(partes.puerto);
      if (i) setIntervalo(i);
      // Con la IP elastica ya configurada no hace falta abrir ajustes al arrancar:
      // se verifica contra /salud y solo se abren si no conecta.
      probar(inicial, { abrirSiFalla: true });
    })();
  }, [probar]);

  /** Guarda host + puerto como la URL activa y prueba la conexion. */
  const conectar = useCallback(async (h, p) => {
    Keyboard.dismiss();
    const destino = armarUrl(h, p);
    if (!destino) {
      setError('Escribe la IP del servidor');
      return;
    }
    const partes = partirUrl(destino);
    setHost(partes.host);
    setPuerto(partes.puerto);
    setUrl(destino);
    setSalud(null);
    await AsyncStorage.multiSet([[CLAVE_URL, destino], [CLAVE_INTERVALO, intervalo]]);
    await probar(destino, { cerrarSiConecta: true });
  }, [intervalo, probar]);

  // ------------------------------------------------------------ captura
  const capturar = useCallback(async () => {
    if (ocupado.current || !camara.current || !url) return;
    ocupado.current = true;
    setEnVuelo(true);
    try {
      setFase('capturando');
      const foto = await camara.current.takePictureAsync({ quality: 0.7, skipProcessing: true });
      setFase('comprimiendo');
      const comprimida = await comprimir(foto.uri);
      setFase('subiendo');
      const res = await inspeccionarFrame(url, comprimida, 'banda-01', (p) =>
        setFase(p >= 1 ? 'analizando' : `subiendo ${Math.round(p * 100)}%`)
      );
      setUltimo(res);
      setError(null);
      setConteo((c) => {
        const siguiente = { ...c, frames: c.frames + 1 };
        // Solo los huevos nuevos suman: el backend ya marca los repetidos.
        res.detecciones.filter((d) => d.nuevo).forEach((d) => {
          siguiente[d.ruta] = (siguiente[d.ruta] || 0) + 1;
        });
        return siguiente;
      });
    } catch (e) {
      setError(e.message);
    } finally {
      ocupado.current = false;
      setEnVuelo(false);
      setFase('');
    }
  }, [url]);

  // Tick de la banda. Se re-arma en cada ciclo en vez de usar setInterval para
  // que un frame lento no encole disparos encima del anterior.
  useEffect(() => {
    if (!continuo) {
      clearTimeout(temporizador.current);
      return;
    }
    let vivo = true;
    const ciclo = async () => {
      await capturar();
      if (vivo) temporizador.current = setTimeout(ciclo, Math.max(500, Number(intervalo) || 2000));
    };
    ciclo();
    return () => { vivo = false; clearTimeout(temporizador.current); };
  }, [continuo, capturar, intervalo]);

  // ------------------------------------------------------------- permisos
  if (!permiso) return <View style={e.centro}><ActivityIndicator color="#fff" /></View>;
  if (!permiso.granted) {
    return (
      <SafeAreaView style={e.centro}>
        <Text style={e.titulo}>Se necesita la camara</Text>
        <Text style={e.sutil}>La app hace de camara de la banda transportadora.</Text>
        <Pressable style={e.boton} onPress={pedirPermiso}>
          <Text style={e.botonTexto}>Dar permiso</Text>
        </Pressable>
      </SafeAreaView>
    );
  }

  const nuevos = (ultimo?.detecciones || []).filter((d) => d.nuevo);
  const principal = nuevos[0] || ultimo?.detecciones?.[0];
  const estilo = principal ? RUTA[principal.ruta] : null;
  const calidadMala = ultimo?.calidad && ultimo.calidad.sirve === false;
  const actual = partirUrl(url);

  return (
    <SafeAreaView style={e.raiz}>
      <StatusBar style="light" />

      <View style={e.barra}>
        <View style={{ flex: 1 }}>
          <Text style={e.marca}>Banda 01</Text>
          <Text style={e.sutil} numberOfLines={1}>
            {salud
              ? `${salud.modelo} · puerto ${actual.puerto}`
              : probando ? `conectando a ${url}…` : `sin conexion · ${url || 'sin configurar'}`}
          </Text>
        </View>
        <Pressable onPress={() => setAjustesAbiertos(true)} style={e.iconoBoton}>
          <Text style={{ fontSize: 19 }}>{'⚙'}</Text>
        </Pressable>
      </View>

      <View style={e.marcoCamara}>
        <CameraView ref={camara} style={{ flex: 1 }} facing="back" />
        {calidadMala && (
          <View style={e.avisoCalidad}>
            <Text style={e.avisoIcono}>{'⚠'}</Text>
            <Text style={e.avisoTexto}>{ultimo.calidad.mensaje}</Text>
          </View>
        )}
        {estilo && (
          <View style={[e.veredicto, { borderColor: estilo.color }]}>
            <Text style={[e.veredictoIcono, { color: estilo.color }]}>{estilo.icono}</Text>
            <View style={{ flex: 1 }}>
              <Text style={[e.veredictoTexto, { color: estilo.color }]}>{estilo.texto}</Text>
              <Text style={e.sutil}>
                {principal.clase} · {(principal.confianza * 100).toFixed(0)}% · {ultimo.latencia_ms.total_ms} ms
                {principal.n_vistas > 1 ? ` · ${principal.n_vistas} tomas` : ''}
              </Text>
              <Text style={e.motivo} numberOfLines={2}>{principal.motivo}</Text>
            </View>
          </View>
        )}
        {enVuelo && (
          <View style={e.velo}>
            <ActivityIndicator size="large" color="#fff" />
            {!!fase && <Text style={e.fase}>{fase}</Text>}
          </View>
        )}
      </View>

      <View style={e.contadores}>
        {['ACEPTADO', 'REVISION', 'DESCARTE'].map((k) => (
          <View key={k} style={e.contador}>
            <Text style={[e.contadorNum, { color: RUTA[k].color }]}>{conteo[k] || 0}</Text>
            <Text style={e.contadorEtq}>{RUTA[k].icono} {k}</Text>
          </View>
        ))}
        <View style={e.contador}>
          <Text style={e.contadorNum}>{conteo.frames}</Text>
          <Text style={e.contadorEtq}>frames</Text>
        </View>
      </View>

      {error && <Text style={e.error} numberOfLines={3}>{error}</Text>}

      <View style={e.acciones}>
        <Pressable
          style={[e.boton, e.botonSec, !url && e.desactivado]}
          disabled={!url || continuo}
          onPress={capturar}
        >
          <Text style={e.botonTexto}>Disparo manual</Text>
        </Pressable>
        <Pressable
          style={[e.boton, continuo ? e.botonParar : e.botonPrim, !url && e.desactivado]}
          disabled={!url}
          onPress={() => setContinuo((v) => !v)}
        >
          <Text style={e.botonTexto}>{continuo ? 'Detener banda' : 'Arrancar banda'}</Text>
        </Pressable>
      </View>

      {/* ----------------------------------------------------- ajustes --- */}
      <Modal visible={ajustesAbiertos} animationType="slide" transparent onRequestClose={() => setAjustesAbiertos(false)}>
        {/* KeyboardAvoidingView: el teclado ya no tapa el boton. keyboardShouldPersistTaps:
            el primer toque en un boton con el teclado abierto SI cuenta (antes solo lo cerraba). */}
        <KeyboardAvoidingView style={e.modalFondo} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
          <ScrollView style={e.modal} contentContainerStyle={{ padding: 20 }} keyboardShouldPersistTaps="handled">
            <Text style={e.titulo}>Configuracion</Text>

            <Text style={e.etiqueta}>Version del modelo</Text>
            <Text style={e.ayuda}>Cada version corre en un puerto del servidor. Un toque cambia y prueba la conexion.</Text>
            {SERVIDORES.map((s) => {
              const activo = actual.puerto === s.puerto;
              return (
                <Pressable key={s.puerto} style={[e.opcion, activo && e.opcionActiva]} onPress={() => conectar(host, s.puerto)}>
                  <View style={{ flex: 1 }}>
                    <Text style={e.opcionTitulo}>{s.nombre} · puerto {s.puerto}</Text>
                    <Text style={e.opcionDetalle}>{s.detalle}</Text>
                  </View>
                  {activo && (
                    <Text style={[e.opcionEstado, { color: salud ? '#0ca30c' : probando ? '#fab219' : '#e66767' }]}>
                      {salud ? '● conectado' : probando ? '… probando' : '○ sin conexion'}
                    </Text>
                  )}
                </Pressable>
              );
            })}

            <Text style={e.etiqueta}>O escribelo a mano</Text>
            <View style={e.fila}>
              <View style={{ flex: 1 }}>
                <Text style={e.ayuda}>IP del servidor</Text>
                <TextInput
                  style={e.input}
                  value={host}
                  onChangeText={setHost}
                  placeholder={HOST_POR_DEFECTO}
                  placeholderTextColor="#6b6b68"
                  autoCapitalize="none"
                  autoCorrect={false}
                  keyboardType="url"
                />
              </View>
              <View style={{ width: 96 }}>
                <Text style={e.ayuda}>Puerto</Text>
                <TextInput
                  style={e.input}
                  value={puerto}
                  onChangeText={(t) => setPuerto(t.replace(/\D/g, ''))}
                  placeholder="8000"
                  placeholderTextColor="#6b6b68"
                  keyboardType="number-pad"
                  maxLength={5}
                  selectTextOnFocus
                />
              </View>
            </View>
            <Text style={e.ayuda}>Se conectara a {armarUrl(host, puerto) || '—'}</Text>
            <Pressable
              style={[e.boton, e.botonPrim, { marginTop: 12 }, probando && e.desactivado]}
              disabled={probando}
              onPress={() => conectar(host, puerto)}
            >
              {probando ? <ActivityIndicator color="#fff" /> : <Text style={e.botonTexto}>Probar y guardar</Text>}
            </Pressable>
            {salud && !probando && (
              <Text style={e.ok}>
                Conectado · {salud.modelo} · puerto {actual.puerto}{'\n'}
                bus: {salud.bus} · actuador: {salud.actuador}
              </Text>
            )}
            {error && <Text style={e.errorModal}>{error}</Text>}

            <Text style={e.etiqueta}>Velocidad de la banda (ms entre huevos)</Text>
            <Text style={e.ayuda}>2000 ms = 30 huevos/min. No bajes de 500 ms. Se guarda sola.</Text>
            <TextInput
              style={e.input}
              value={intervalo}
              onChangeText={setIntervalo}
              onEndEditing={() => AsyncStorage.setItem(CLAVE_INTERVALO, intervalo)}
              keyboardType="number-pad"
              placeholder="2000"
              placeholderTextColor="#6b6b68"
            />
            <Pressable style={[e.boton, e.botonSec, { marginTop: 18 }]} onPress={() => setAjustesAbiertos(false)}>
              <Text style={e.botonTexto}>Cerrar</Text>
            </Pressable>
          </ScrollView>
        </KeyboardAvoidingView>
      </Modal>
    </SafeAreaView>
  );
}

const e = StyleSheet.create({
  raiz: { flex: 1, backgroundColor: '#0d0d0d' },
  centro: { flex: 1, backgroundColor: '#0d0d0d', alignItems: 'center', justifyContent: 'center', padding: 24 },
  barra: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 10, gap: 10 },
  marca: { color: '#fff', fontSize: 16, fontWeight: '600' },
  sutil: { color: '#898781', fontSize: 12 },
  iconoBoton: { width: 38, height: 38, borderRadius: 10, backgroundColor: '#1a1a19', alignItems: 'center', justifyContent: 'center' },
  marcoCamara: { flex: 1, margin: 16, borderRadius: 14, overflow: 'hidden', backgroundColor: '#000' },
  velo: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.35)', alignItems: 'center', justifyContent: 'center' },
  fase: { color: '#fff', fontSize: 13, marginTop: 10, letterSpacing: 0.3 },
  veredicto: {
    position: 'absolute', left: 12, right: 12, bottom: 12, flexDirection: 'row', alignItems: 'center', gap: 12,
    backgroundColor: 'rgba(13,13,13,0.9)', borderWidth: 2, borderRadius: 12, padding: 12,
  },
  veredictoIcono: { fontSize: 30, fontWeight: '700' },
  veredictoTexto: { fontSize: 19, fontWeight: '700', letterSpacing: 0.5 },
  motivo: { color: '#6f6e6a', fontSize: 10.5, marginTop: 3 },
  avisoCalidad: {
    position: 'absolute', left: 12, right: 12, top: 12, flexDirection: 'row', alignItems: 'center', gap: 9,
    backgroundColor: 'rgba(250,178,25,0.95)', borderRadius: 10, paddingVertical: 9, paddingHorizontal: 12,
  },
  avisoIcono: { fontSize: 17, color: '#0d0d0d' },
  avisoTexto: { flex: 1, color: '#0d0d0d', fontSize: 12.5, fontWeight: '600' },
  contadores: { flexDirection: 'row', gap: 8, paddingHorizontal: 16 },
  contador: { flex: 1, backgroundColor: '#1a1a19', borderRadius: 10, paddingVertical: 10, alignItems: 'center' },
  contadorNum: { color: '#fff', fontSize: 21, fontWeight: '700' },
  contadorEtq: { color: '#898781', fontSize: 10, marginTop: 2 },
  acciones: { flexDirection: 'row', gap: 10, padding: 16 },
  boton: { flex: 1, paddingVertical: 14, borderRadius: 11, alignItems: 'center' },
  botonPrim: { backgroundColor: '#2a78d6' },
  botonSec: { backgroundColor: '#2c2c2a' },
  botonParar: { backgroundColor: '#d03b3b' },
  botonTexto: { color: '#fff', fontWeight: '600', fontSize: 15 },
  desactivado: { opacity: 0.4 },
  error: { color: '#e66767', fontSize: 12.5, paddingHorizontal: 16, paddingTop: 8 },
  ok: { color: '#0ca30c', fontSize: 12.5, marginTop: 12, lineHeight: 18 },
  modalFondo: { flex: 1, backgroundColor: 'rgba(0,0,0,0.75)', justifyContent: 'flex-end' },
  modal: { maxHeight: '85%', backgroundColor: '#1a1a19', borderTopLeftRadius: 18, borderTopRightRadius: 18 },
  titulo: { color: '#fff', fontSize: 19, fontWeight: '700', marginBottom: 16 },
  etiqueta: { color: '#c3c2b7', fontSize: 13, fontWeight: '600', marginTop: 14 },
  ayuda: { color: '#898781', fontSize: 11.5, marginTop: 2, marginBottom: 6 },
  input: { backgroundColor: '#0d0d0d', color: '#fff', borderRadius: 10, padding: 13, fontSize: 15, borderWidth: 1, borderColor: '#2c2c2a' },
  errorModal: { color: '#e66767', fontSize: 12.5, marginTop: 10 },
  fila: { flexDirection: 'row', gap: 10 },
  opcion: {
    flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: '#0d0d0d', borderRadius: 11,
    paddingVertical: 12, paddingHorizontal: 14, marginTop: 8, borderWidth: 1.5, borderColor: '#2c2c2a',
  },
  opcionActiva: { borderColor: '#2a78d6' },
  opcionTitulo: { color: '#fff', fontSize: 15, fontWeight: '600' },
  opcionDetalle: { color: '#898781', fontSize: 11.5, marginTop: 2 },
  opcionEstado: { fontSize: 12, fontWeight: '600' },
});
