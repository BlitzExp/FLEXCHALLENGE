# FLEXCHALLENGE

Clasificación del dispositivo de un ticket a partir del trabajo realizado.

## Algoritmo Q → AB

El archivo `clasificador_tickets.py` lee `data/Clasificacion de Tickets 2021.xlsm`, hoja **Data**:

- **Entrada Q:** Work Description (Close Details).
- **Objetivo AB:** Dispositivo.
- **Modelo:** TF-IDF de palabras (1–2) y fragmentos de caracteres (3–5), seguido de una máquina de soporte vectorial lineal (LinearSVC).

El modelo solo utiliza Q para predecir. AB proporciona las etiquetas de entrenamiento. No usa otras columnas como predictores. Q describe el trabajo de cierre, por lo que este algoritmo clasifica ese texto y no anticipa la falla al abrir el ticket.

## Instalación y uso

Desde la carpeta del proyecto, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-clasificacion.txt

# Entrenar, evaluar y guardar el modelo
.\.venv\Scripts\python.exe clasificador_tickets.py entrenar

# Obtener la categoría y hasta tres opciones para un texto nuevo
.\.venv\Scripts\python.exe clasificador_tickets.py predecir "se relleno tanque de aceite"

# Clasificar toda la columna Q, incluidas filas sin AB
.\.venv\Scripts\python.exe clasificador_tickets.py archivo

# Ejecutar las pruebas
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Ya se generó el modelo local en `modelos/clasificador_tickets.joblib`. Los modelos y las salidas se excluyen de Git porque son archivos generados. En otra copia del proyecto, ejecutar primero `entrenar`. Reentrenar también después de cambiar la versión de scikit-learn, el catálogo de categorías o los ejemplos.

Para utilizarlo desde Python o desde un notebook ubicado en la raíz del proyecto:

```python
from clasificador_tickets import cargar_modelo, clasificar_texto, predecir_textos

modelo = cargar_modelo()
categoria = clasificar_texto("se relleno tanque de aceite", modelo)
print(categoria)

# Procesamiento de una lista de textos
resultados = predecir_textos([
    "se reemplaza motor electrico",
    "se relleno tanque de aceite",
    "",
], modelo)
```

`clasificar_texto` devuelve una categoría existente en AB o `None` cuando falta texto o no hay vocabulario conocido. `predecir_textos` devuelve además alternativas, estado y cantidad de ejemplos de la categoría. Los puntajes de SVM ordenan las opciones: **no son probabilidades ni porcentajes de confianza**.

Se puede elegir otro archivo mediante `--archivo "ruta/al/libro.xlsm"`, otro modelo mediante `--modelo` y la salida mediante `--salida`. La hoja debe conservar los encabezados esperados en Q y AB. `--salida` es una carpeta para entrenar y un archivo CSV para clasificar. Los comandos pueden ejecutarse desde otra carpeta usando la ruta absoluta del script; sus rutas predeterminadas se resuelven respecto al proyecto.

## Preparación de datos

1. Se leen Q y AB por posición Excel, conservando el número de fila original y el identificador del ticket.
2. Los textos se normalizan en mayúsculas/minúsculas, acentos, puntuación y espacios. Las etiquetas de AB solo se normalizan en mayúsculas/minúsculas y espacios. Se devuelve la variante más frecuente del nombre original.
3. Para entrenar se requieren texto y categoría. No se inventan etiquetas para filas sin AB ni texto para las filas sin Q.
4. Los textos normalizados que tienen categorías contradictorias se excluyen del entrenamiento y se exportan para revisión. Las repeticiones del mismo texto y categoría cuentan una sola vez.
5. La separación 80/20 agrupa por texto normalizado **o** por ticket, incluyendo relaciones transitivas. Así, los mismos textos y tickets no aparecen en ambos conjuntos. TF-IDF se aprende únicamente en entrenamiento.
6. Se mide una única configuración, con semilla 42 y sin ajustar parámetros con la prueba. Luego se entrena el modelo de uso con todos los ejemplos limpios. La prueba es aleatoria por grupos, no temporal; no mide cambios futuros del proceso.

## Resultado de la ejecución

| Indicador | Resultado |
| --- | ---: |
| Filas leídas | 11,150 |
| Pares Q/AB presentes | 5,222 |
| Textos con categorías contradictorias | 34 |
| Filas excluidas por contradicciones | 187 |
| Repeticiones adicionales excluidas | 293 |
| Ejemplos utilizables | 4,742 |
| Categorías utilizables | 360 |
| Ejemplos de entrenamiento / prueba | 3793 / 949 |
| Exactitud en prueba | 44.26% |
| Categoría correcta entre las primeras tres | 64.81% |
| F1 macro sobre categorías presentes en prueba | 0.195 |
| F1 ponderado | 0.410 |
| Exactitud de elegir la categoría mayoritaria | 19.70% |

Hay 23 categorías de prueba ausentes del entrenamiento (25 filas); se incluyen como errores en la evaluación. No se excluyen las categorías raras para mejorar artificialmente la exactitud. Los resultados superan la base mayoritaria, pero requieren revisión humana: la primera opción se equivoca en más de la mitad de los ejemplos de prueba. No hay un umbral calibrado de aceptación automática. Los textos parcialmente desconocidos también pueden recibir una sugerencia incorrecta.

Variantes semánticas y errores de escritura en AB, por ejemplo singular/plural o palabras mal escritas, permanecen como categorías distintas. Su unificación requiere revisar el catálogo. La deduplicación agrupa textos idénticos tras normalizar, pero no garantiza separar paráfrasis o descripciones muy similares. Tampoco permite aprender categorías que no tienen ejemplos válidos.

## Archivos generados

- `salidas/predicciones.csv`: una fila por registro, con número de fila Excel, ticket, texto Q, etiqueta original AB, categoría sugerida, estado, número de ejemplos de esa categoría y alternativas. Se abre en Excel y permite cotejar el resultado con el original. No se escribe en AB.
- `salidas/metricas.json`: auditoría, métricas y precisión/recobrado/F1 por categoría.
- `salidas/evaluacion.csv`: predicciones únicamente de los ejemplos reservados de prueba, hechas antes del reajuste final.
- `salidas/conflictos.csv`: filas excluidas por etiquetas contradictorias.
- `salidas/categorias.csv`: categorías aprendidas y cantidades de textos únicos.
- `modelos/clasificador_tickets.joblib`: modelo, nombres, conteos, métricas y huella SHA-256 del archivo fuente. Cargar únicamente modelos propios o de confianza.

**El XLSM original se abre solo para lectura; conserva sus datos, fórmulas y macros.** Los resultados son CSV separados con codificación UTF-8 con BOM. Los campos de texto que puedan interpretarse como fórmulas llevan un apóstrofo inicial en el CSV.

Las predicciones de todas las filas se generan con el modelo final, que ya vio los ejemplos de entrenamiento. Por tanto, no deben utilizarse para medir exactitud. Para ello están `evaluacion.csv` y `metricas.json`. Al repetir los comandos se reemplazan los archivos de salida indicados.

## Análisis exploratorio existente

`EDA_mantenimiento.ipynb` contiene el análisis exploratorio previo de tickets y paros. No se modifica con este clasificador. Sus dependencias están en `requirements.txt`; `requirements-clasificacion.txt` permite instalar únicamente las necesarias para la clasificación.
