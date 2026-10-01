# Segundo modelo — clasificador YOLOv8-cls sobre CDD_eggs_SJ

| archivo | qué es |
|---|---|
| [`entrenar_cdd_eggs_yolov8cls.ipynb`](entrenar_cdd_eggs_yolov8cls.ipynb) | **El notebook para Colab.** Súbelo con *Archivo → Subir notebook*. |
| `cdd_split_lib.py` | Lógica del split sin fuga (va incrustada en el notebook). |
| `test_cdd_split_lib.py` | 25 pruebas: fugas plantadas, reproducibilidad y regresión del bug del grupo gigante. |
| `validar_notebook.py` | 65 verificaciones del notebook, incluido un ensayo en seco con modelo simulado. |
| `generar_notebook.py` | Genera el `.ipynb` incrustando la biblioteca probada. |
| `comparar_modelos.py` | Detector vs clasificador sobre los mismos 183 huevos. Resultado en [`docs/COMPARACION_MODELOS.md`](../docs/COMPARACION_MODELOS.md). |

## Qué responde el modelo

Aprende 3 clases (`Roto`, `Sano`, `Sucio`) y decide en forma jerárquica, con un solo número
—p(Roto)— y dos cortes calibrados en val:

| veredicto | cuándo |
|---|---|
| ✗ **DAÑADO** | p(Roto) ≥ `TAU_ROTO` |
| ⚠ **REVISIÓN** | `TAU_DUDA` ≤ p(Roto) < `TAU_ROTO` |
| ✓ **BUENO (SUCIO)** | p(Roto) < `TAU_DUDA` y p(Sucio) > p(Sano) |
| ✓ **BUENO** | p(Roto) < `TAU_DUDA` y p(Sano) ≥ p(Sucio) |

## Uso

1. Colab → **Archivo → Subir notebook** → `entrenar_cdd_eggs_yolov8cls.ipynb`.
2. **Entorno de ejecución → Cambiar tipo de entorno → T4 GPU.**
3. Panel izquierdo → **🔑 Secretos** → **+ Agregar nuevo secreto** → nombre `ROBOFLOW_API_KEY`,
   valor = tu **Private API Key** (no la *Publishable* `rf_...`) → activa **Acceso al notebook**.
4. **Entorno de ejecución → Ejecutar todas.** ~35–45 min.
   - Al inicio pide **permiso de Google Drive**: acéptalo (se pide al principio a propósito).
   - El entrenamiento se guarda en Drive época a época: sobrevive a una desconexión.
   - La **última celda** te pide subir fotos tuyas de huevos y muestra el veredicto de cada una.

Resultado en `MyDrive/eggdetection/`: el zip con `best.pt`, `best.onnx`, curvas, matriz de
confusión, `metricas.json` y el manifiesto del split.

## Si cambias la lógica

```bash
python colab/test_cdd_split_lib.py      # la biblioteca
python colab/generar_notebook.py        # regenerar el .ipynb
python colab/validar_notebook.py        # verificar el notebook completo
```
