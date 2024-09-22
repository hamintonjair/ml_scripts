import pandas as pd
import joblib
import json
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import webbrowser

warnings.filterwarnings("ignore", message="X has feature names, but LinearRegression was fitted without feature names")

# Ruta al archivo JSON donde se guardan las incidencias
json_path = 'C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\datos_incidencias.json'

# Cargar los datos desde el archivo JSON
try:
    with open(json_path, 'r') as file:
        data = json.load(file)
except json.JSONDecodeError as e:
    print("Error al decodificar el JSON de entrada:", e)
    sys.exit(1)
except FileNotFoundError as e:
    print("Archivo JSON no encontrado:", e)
    sys.exit(1)

# Convertir el JSON en un DataFrame de pandas
df = pd.DataFrame(data)

# Asegúrate de que los datos están en el formato correcto
df['mes'] = pd.to_numeric(df['mes'], errors='coerce')

# Extraer la hora y los minutos directamente desde la columna original
hora_original = df['hora']
df['hora'] = hora_original.str.split(':').str[0].astype(int)    # Extrae la hora como entero
df['minutos'] = hora_original.str.split(':').str[1].astype(int) # Extrae los minutos como entero

# Unificar hora y minutos en una sola columna con formato entero HHMM (por ejemplo, 14:30 se convierte en 1430)
df['hora_unificada'] = df['hora'] * 100 + df['minutos'] # Esto crea la columna con hora y minutos unificados

# Definir la función para clasificar la hora unificada
def clasificar_hora(hora_unificada):
    if 0 <= hora_unificada < 600:
        return '00:00-06:00'
    elif 600 <= hora_unificada < 1200:
        return '06:00-12:00'
    elif 1200 <= hora_unificada < 1800:
        return '12:00-18:00'
    else:
        return '18:00-24:00'

# Aplicar la función de clasificación a la columna unificada
df['intervalo_hora'] = df['hora_unificada'].apply(clasificar_hora)

# Crear la columna 'es_fin_de_semana' (1 si es sábado o domingo, 0 de lo contrario)
df['es_fin_de_semana'] = df['dia'].apply(lambda x: 1 if x in [6, 7] else 0)

# Preparación para el PDF y lo guardamos
pdf_path = 'C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\reporte_graficas.pdf'

# Crear el archivo PDF de cada gráfica
with PdfPages(pdf_path) as pdf:
      # Explicación general del aprendizaje aplicado
    plt.figure(figsize=(12, 6))
    plt.text(0.05, 0.5, 
             'Tipo de Aprendizaje Aplicado:\n\n'
             '1. Aprendizaje Supervisado: Se utilizan modelos de aprendizaje supervisado para hacer\n'
             ' predicciones sobre los incidentes, basados en datos históricos etiquetados.\n'
             '   - Modelo de Regresión: Se usa para predecir la cantidad de incidentes en \n'
             ' función de las características de los datos.\n'
             '   - Modelo de Clasificación: Se utiliza para predecir el tipo de incidencia que se \n'
             ' espera en función de las características de los datos.\n\n'
             '2. Análisis de Componentes Principales (PCA): Se aplica PCA para reducir la\n'
             ' dimensionalidad de los datos y mejorar la eficiencia de los modelos.\n'
             '   - PCA transforma los datos en componentes principales que capturan la mayor\n'
             ' varianza posible con menos variables.\n\n'
             'Cada gráfico en este informe proporciona una visión de diferentes aspectos \n'
             ' del análisis de datos de incidentes.\n', 
             ha='left', va='center', fontsize=14)
    
    plt.axis('off')
    pdf.savefig()
    plt.close()
    	
    # Primer gráfico
    plt.figure(figsize=(12, 6))
    plt.text(0.05, 0.5, 
			'Explicación de las Predicciones:\n\n'
			'Las predicciones generadas por los modelos de aprendizaje supervisado son fundamentales\n'
			'para la toma de decisiones en el análisis de incidentes. La elección de estos modelos se\n'
			'basó en su capacidad para manejar grandes volúmenes de datos y extraer patrones significativos.\n\n'
			'   - Modelo de Regresión: Este modelo permite estimar la cantidad esperada de incidentes\n'
			'     en función de variables predictivas, facilitando la planificación y gestión de recursos.\n\n'
			'   - Modelo de Clasificación: Ayuda a categorizar los incidentes en tipos específicos,\n'
			'     lo que es crucial para priorizar las acciones y estrategias de respuesta según el tipo de\n'
			'     incidencia más probable.\n\n'
			'El uso de PCA contribuye a simplificar el análisis al reducir la complejidad del modelo sin\n'
			'perder la capacidad de capturar las características más relevantes de los datos.\n', 
			ha='left', va='center', fontsize=14)
    plt.axis('off')
    pdf.savefig()
    plt.close()

    # Segundo gráfico
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='dia')
    plt.title('Distribución de Incidentes por Día de la Semana')
    plt.xlabel('Día de la Semana')
    plt.ylabel('Número de Incidentes')
    plt.subplots_adjust(bottom=0.30)  # Ajustar el margen inferior
    plt.text(0.5, -0.3, 'Este gráfico muestra la distribución de incidentes a lo largo de los días de la semana.', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='mes')
    plt.title('Distribución de Incidentes por Mes')
    plt.xlabel('Mes')
    plt.ylabel('Número de Incidentes')
    plt.subplots_adjust(bottom=0.30)  # Ajustar el margen inferior
    plt.text(0.5, -0.3, 'Este gráfico muestra el número de incidentes registrados en cada mes del año.', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    pdf.savefig()
    plt.close()

    # Tercer gráfico
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='intervalo_hora')
    plt.title('Distribución de Incidentes por Intervalo de Hora')
    plt.xlabel('Intervalo de Hora')
    plt.ylabel('Número de Incidentes')
    plt.subplots_adjust(bottom=0.30)  # Ajustar el margen inferior
    plt.text(0.5, -0.3, 'Este gráfico muestra la cantidad de incidentes distribuidos en diferentes intervalos horarios del día.', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    pdf.savefig()
    plt.close()

    # Cuarto gráfico
    plt.figure(figsize=(12, 7))
    sns.countplot(data=df, x='barrio', order=df['barrio'].value_counts().index)
    plt.title('Distribución de Incidentes por Barrio')
    plt.xlabel('Barrios')
    plt.ylabel('Número de Incidentes')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.45)  # Ajustar el margen inferior
    plt.text(0.5, -0.7, 'Este gráfico muestra la cantidad de incidentes por barrio, ordenados de mayor a menor.', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    pdf.savefig()
    plt.close()

    # Quinto gráfico
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='tipo_incidencia')
    plt.title('Distribución de Incidentes por Tipo de Incidencia')
    plt.xlabel('Tipo de Incidencias')
    plt.ylabel('Número de Incidentes')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.47)  # Ajustar el margen inferior
    plt.text(0.5, -0.8, 'Este gráfico muestra la distribución de incidentes según el tipo de incidencia.', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    pdf.savefig()
    plt.close()

    # Sexto gráfico
    plt.figure(figsize=(12, 8))
    heatmap_data = df.groupby(['dia', 'intervalo_hora']).size().unstack()
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True)
    plt.title('Número de Incidentes por Día y Intervalo de Hora')
    plt.xlabel('Intervalo de Hora')
    plt.ylabel('Día de la Semana')
    plt.subplots_adjust(bottom=0.4)  # Ajustar el margen inferior
    plt.text(0.5, -0.3, 'Este gráfico muestra la cantidad de incidentes distribuidos por día de la semana y por intervalo horario.', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    pdf.savefig()
    plt.close()

    # Séptimo gráfico
    plt.figure(figsize=(10, 6))
    correlation_matrix = df[['mes', 'hora', 'es_fin_de_semana']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación')
    plt.subplots_adjust(bottom=0.4)  # Ajustar el margen inferior
    plt.text(0.5, -0.3, 'Este gráfico muestra la matriz de correlación entre las variables mes, hora y si es fin de semana.', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    pdf.savefig()
    plt.close()

# Abrir el archivo PDF en el navegador predeterminado
webbrowser.open_new(pdf_path)


# Cargar modelos y PCA
try:
    model_regresion = joblib.load('C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\modelo_regresion.pkl')
    model_clasificacion = joblib.load('C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\modelo_clasificacion.pkl')
    pca = joblib.load('C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\modelo_pca.pkl')
    le = joblib.load('C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\modelo_le.pkl')
    columnas = joblib.load('C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\columnas.pkl')
except FileNotFoundError as e:
    print("Modelo no encontrado:", e)
    sys.exit(1)

# Preparación de los datos de entrada para la predicción
X_nueva = df[['mes', 'hora', 'intervalo_hora', 'es_fin_de_semana']]
X_nueva = pd.get_dummies(X_nueva, columns=['intervalo_hora'], drop_first=True)

# Asegúrate de que las columnas coincidan con las del entrenamiento
X_nueva = X_nueva.reindex(columns=columnas, fill_value=0)

# Aplicar PCA a los datos
X_nueva_pca = pca.transform(X_nueva)

# Predicciones
df['cantidad_pred'] = model_regresion.predict(X_nueva_pca)
df['tipo_incidencia_pred'] = le.inverse_transform(model_clasificacion.predict(X_nueva))


# ****************************************************************
# Guardar los resultados en un archivo CSV
resultados_csv_path = 'C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\resultados_predicciones.csv'
df.to_csv(resultados_csv_path, index=False)
print(f"Predicciones guardadas en {resultados_csv_path}")

# ************************************************

# Ruta al archivo CSV con los resultados de las predicciones
resultados_csv_path = 'C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\resultados_predicciones.csv'

# Leer el archivo CSV
df_resultados = pd.read_csv(resultados_csv_path)

# Analizar patrones por barrio
# Agrupar por barrio y calcular las estadísticas
patrones_barrios = df_resultados.groupby('barrio').agg({
    'cantidad': 'sum',                  # Total de incidencias por barrio
    'cantidad_pred': 'mean',     # Promedio de la predicción de regresión por barrio
}).reset_index()

# Calcular la moda de la predicción de clasificación para cada barrio
patrones_barrios['prediccion_clasificacion_mode'] = df_resultados.groupby('barrio')['tipo_incidencia_pred'].agg(pd.Series.mode).reset_index(drop=True)

# Analizar patrones por tipo de incidencia
# Agrupar por tipo de incidencia y calcular las estadísticas
patrones_tipos = df_resultados.groupby('tipo_incidencia').agg({
    'cantidad': 'sum',                  # Total de incidencias por tipo
    'cantidad_pred': 'mean',     # Promedio de la predicción de regresión por tipo
}).reset_index()

# Calcular la moda de la predicción de clasificación para cada tipo de incidencia
patrones_tipos['prediccion_clasificacion_mode'] = df_resultados.groupby('tipo_incidencia')['tipo_incidencia_pred'].agg(pd.Series.mode).reset_index(drop=True)

# Mostrar los patrones encontrados
print("Patrones por barrio:")
print(patrones_barrios)

print("\nPatrones por tipo de incidencia:")
print(patrones_tipos)

# Guardar los resultados en archivos CSV para su posterior análisis
patrones_barrios_csv_path = 'C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\patrones_barrios.csv'
patrones_tipos_csv_path = 'C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\patrones_tipos.csv'

patrones_barrios.to_csv(patrones_barrios_csv_path, index=False)
patrones_tipos.to_csv(patrones_tipos_csv_path, index=False)

print(f"\nPatrones por barrio guardados en {patrones_barrios_csv_path}")
print(f"Patrones por tipo de incidencia guardados en {patrones_tipos_csv_path}")

# *****************************************************************************************
# Mostrar resultados
print(df[['cantidad_pred', 'tipo_incidencia_pred']].head())

# Preparación para el PDF de las nuevas visualizaciones
pdf_predictions_path = 'C:\\xampp\\htdocs\\Seguridad_Ciudadana\\ml_scripts\\reporte_predicciones.pdf'

# Crear el archivo PDF de cada gráfica
with PdfPages(pdf_predictions_path) as pdf:
    
    # Tercer gráfico de predicciones
    plt.figure(figsize=(10, 6))
    df_baras = df.groupby('es_fin_de_semana').sum().reset_index()
    sns.barplot(data=df_baras, x='es_fin_de_semana', y='cantidad_pred', hue='es_fin_de_semana', palette='viridis', dodge=False, legend=False)
    plt.title('Cantidad Predicha en Fin de Semana vs. Días Laborales')
    plt.xlabel('Es Fin de Semana')
    plt.ylabel('Cantidad Predicha')
    plt.xticks(ticks=[0, 1], labels=['Día Laboral', 'Fin de Semana'])
    plt.subplots_adjust(bottom=0.35)  # Ajustar el margen inferior
    plt.text(0.5, -0.3, 'Este gráfico muestra la cantidad total de incidencias predicha para días laborales y fines de semana.', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.grid(True)
    pdf.savefig()
    plt.close()


    plt.figure(figsize=(12, 8))
    df_baras = df.groupby('tipo_incidencia_pred').sum().reset_index()
    sns.barplot(data=df_baras, x='tipo_incidencia_pred', y='cantidad_pred', hue='tipo_incidencia_pred', palette='viridis', legend=False)
    plt.title('Cantidad Total Predicha por Tipo de Incidencia')
    plt.xlabel('Tipo de Incidencia')
    plt.ylabel('Cantidad Predicha')
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.45)  # Ajusta el margen inferior
    plt.text(0.5, -0.5, 'Este gráfico muestra la cantidad total de incidentes predicha para cada tipo de incidencia.', 
			ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.5, -0.6, 'Cada barra representa un tipo de incidencia con su correspondiente cantidad predicha,', 
			ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.5, -0.7, 'facilitando la comparación entre diferentes tipos.', 
			ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.grid(True)
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(12, 8))
    df_torta = df['tipo_incidencia_pred'].value_counts()
    plt.pie(df_torta, labels=df_torta.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', len(df_torta)))
    plt.title('Distribución de Incidencias Predicha por Tipo')
    plt.subplots_adjust(bottom=0.25)  # Ajusta el margen inferior
    plt.text(0.5, -0.1, 'Este gráfico circular ilustra la proporción de incidentes predicha por tipo.', 
         ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.5, -0.2, 'Cada segmento representa el porcentaje del total de incidentes predichos para un tipo específico,', 
         ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.5, -0.3, 'permitiendo una visualización clara de la distribución relativa.', 
         ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='tipo_incidencia_pred', y='cantidad_pred', hue='tipo_incidencia_pred', palette='viridis', legend=False)
    plt.title('Distribución de Cantidades Predichas por Tipo de Incidencia')
    plt.xlabel('Tipo de Incidencia')
    plt.ylabel('Cantidad Predicha')
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.47, top=0.9)  # Ajusta el margen inferior y superior
    plt.text(0.5, -0.6, 'Este gráfico de caja muestra la distribución de las cantidades predichas de incidentes para cada tipo de incidencia.', 
         ha='center', va='top',transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.5, -0.7, 'Incluye los valores medianos, cuartiles y posibles valores atípicos,', 
         ha='center', va='top',transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.5, -0.8, 'proporcionando una visión detallada de la dispersión y tendencia central de los datos predichos.', 
         ha='center', va='top',transform=plt.gca().transAxes, fontsize=12)
    plt.grid(True)
    pdf.savefig()
    plt.close()
        

# Abrir el archivo PDF en el navegador predeterminado
webbrowser.open_new(pdf_predictions_path)
