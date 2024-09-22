
# API de Seguridad Ciudadana

## Descripción

La API de Seguridad Ciudadana es un servicio web diseñado para proporcionar acceso a datos y modelos relacionados con la seguridad en la ciudad. Esta API permite a los desarrolladores integrar funcionalidades de análisis y predicción en sus aplicaciones, facilitando la toma de decisiones informadas basadas en datos históricos y modelos de machine learning.

## Funcionalidades

- **Acceso a Datos**: Proporciona endpoints para consultar datos sobre incidentes de seguridad en la ciudad.
- **Predicciones**: Utiliza modelos de regresión y clasificación para predecir tendencias de seguridad basadas en datos históricos.
- **Análisis de Componentes Principales (PCA)**: Implementa técnicas de reducción de dimensionalidad para mejorar la eficiencia del análisis de datos.
- **Codificación de Etiquetas**: Utiliza técnicas de codificación para preparar datos categóricos para su análisis.

## Endpoints

- `GET /entrenar_modelo`: Recupera una lista de incidentes de seguridad que te proporciona la API, luego realiza el procesamiento de los datos.
- `GET /predicciones`: Este es llamado en la funcion de entrenar modelo y envía datos para obtener predicciones sobre incidentes futuros.

### Flujo Completo: Entrenamiento y Predicción

1. **Entrenamiento del Modelo**: Primero, debes entrenar tu modelo con datos históricos. Esto se hace una vez y luego se guarda el modelo.

2. **Obtención de Datos y Procesamiento**: Luego, obtienes los datos que deseas predecir, procesas esos datos y finalmente los envías a la API para obtener la predicción.

### Ejemplo de Código

Aquí hay un ejemplo en Python que ilustra este flujo:

#### Paso 1: Entrenar el Modelo

```python
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Cargar datos históricos
data = pd.read_csv('datos_segurida.csv')

# Procesar datos (ejemplo)
X = data[['fecha', 'tipo_incidente', 'ubicacion', 'hora']]  # Características
y = data['resultado']  # Etiqueta

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar el modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Guardar el modelo
joblib.dump(modelo, 'modelo_clasificacion.pkl')
```

#### Paso 2: Enviar Datos para Predicción

```python
import requests
import json

# Datos a predecir
datos_a_predecir = {
    "fecha": "2024-09-22",
    "tipo_incidente": "robo",
    "ubicacion": "Centro",
    "hora": "14:30"
}

# Enviar la solicitud POST
url = 'http://localhost:5000/predicciones'
response = requests.post(url, json=datos_a_predecir)

# Imprimir la respuesta
if response.status_code == 200:
    print("Predicción:", response.json())
else:
    print("Error en la predicción:", response.status_code, response.text)
```

### Resumen del Flujo

1. **Entrenamiento del Modelo**: Se entrena un modelo con datos históricos y se guarda.
2. **Preparación de Datos**: Se preparan los datos que se quieren predecir.
3. **Envío de Solicitud**: Se envían los datos a la API mediante una solicitud POST.
4. **Recepción de Predicción**: Se recibe la respuesta con la predicción.

Este flujo asegura que los datos estén correctamente procesados y que el modelo esté listo para hacer predicciones basadas en nuevos datos.

## Requisitos

- Python 3.x
- Bibliotecas: `joblib`, `pandas`, `scikit-learn`, entre otras.

## Instalación

1. Clona el repositorio:
   ```bash
   git clone <URL del repositorio>
   ```
2. Navega al directorio del proyecto:
   ```bash
   cd <nombre del proyecto>
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Para iniciar la API, ejecuta el siguiente comando:
```bash
python app.py
```

La API estará disponible en `http://localhost:5000`.

## Contribuciones

Las contribuciones son bienvenidas. Si deseas colaborar, por favor abre un issue o envía un pull request.

