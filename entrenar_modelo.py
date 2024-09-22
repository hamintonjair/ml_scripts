import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# Ruta al archivo JSON
json_path = 'C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\datos_incidencias.json'

# Cargar los datos desde el archivo JSON
data = pd.read_json(json_path)

# Convertir columnas 'mes' y 'dia' a numéricas
data['mes'] = pd.to_numeric(data['mes'], errors='coerce')
data['dia'] = pd.to_numeric(data['dia'], errors='coerce')

# Extraer la hora y los minutos desde la columna original, asegurando que esté en el formato correcto 'HH:MM'
data['hora_original'] = data['hora']
data['hora'] = data['hora_original'].str.split(':').str[0].astype(int)    # Extraer la hora como entero
data['minutos'] = data['hora_original'].str.split(':').str[1].astype(int) # Extraer los minutos como entero

# Unificar hora y minutos en una sola columna con formato entero HHMM (por ejemplo, 14:30 se convierte en 1430)
data['hora_unificada'] = data['hora'] * 100 + data['minutos']

# Definir la función para clasificar la hora unificada
def clasificar_hora(hora_unificada):
    if 0 <= hora_unificada < 600:
        return 'Madrugada'
    elif 600 <= hora_unificada < 1200:
        return 'Mañana'
    elif 1200 <= hora_unificada < 1800:
        return 'Tarde'
    else:
        return 'Noche'

# Aplicar la clasificación a la columna 'hora_unificada'
data['intervalo_hora'] = data['hora_unificada'].apply(clasificar_hora)

# Crear columna de fin de semana
data['es_fin_de_semana'] = data['dia'].apply(lambda x: 1 if x in [6, 7] else 0)

# Selección de características para los modelos
X = data[['mes', 'intervalo_hora', 'es_fin_de_semana']]
y_cantidad = data['cantidad']  # Variable de salida para regresión
y_tipo = data['tipo_incidencia']  # Variable de salida para clasificación

# Convertir las categorías en variables dummies
X = pd.get_dummies(X, columns=['intervalo_hora'], drop_first=True)

# Convertir 'tipo_incidencia' a números
le = LabelEncoder()
y_tipo = le.fit_transform(y_tipo)

# Guardar las columnas para la predicción
columnas = X.columns

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_cantidad, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_tipo, test_size=0.2, random_state=42)

# Aplicar PCA para reducción de dimensionalidad (solo para regresión)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Entrenamiento de modelos
model_regresion = LinearRegression()
model_regresion.fit(X_train_pca, y_train)

model_clasificacion = RandomForestClassifier()
model_clasificacion.fit(X_train, y_train_clf)

# Guardar los modelos y PCA
joblib.dump(model_regresion, 'C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\modelo_regresion.pkl')
joblib.dump(model_clasificacion, 'C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\modelo_clasificacion.pkl')
joblib.dump(pca, 'C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\modelo_pca.pkl')
joblib.dump(le, 'C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\modelo_le.pkl')
joblib.dump(columnas, 'C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\columnas.pkl')
