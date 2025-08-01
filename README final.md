 # Predicción de Evasión de Clientes (Churn Prediction)
 
Este proyecto busca predecir si un cliente cancelará su contrato (churn) utilizando técnicas de ciencia de datos y machine learning.

 ## Objetivo
Predecir la variable objetivo Churn (1 = cliente canceló, 0 = cliente activo) a partir de características del cliente y su comportamiento.

## 1. Preprocesamiento de Datos
 Eliminación de Columnas Inútiles
Se eliminó la columna customerID, ya que es un identificador único y no aporta valor predictivo.

python
Copiar
Editar
df = df.drop(columns=['customerID'])
 Codificación de Variables Categóricas
Se aplicó One-Hot Encoding a todas las variables categóricas usando pd.get_dummies:

python
Copiar
Editar
df_encoded = pd.get_dummies(df, drop_first=False)
Esto generó nuevas columnas binarias para cada categoría, como gender_Male, Contract_One year, etc.

 Conversión de Variable Objetivo (Churn)
Se limpiaron y mapearon los valores de la columna Churn:

python
Copiar
Editar
df['Churn'] = df['Churn'].astype(str).str.strip()
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
df = df.dropna(subset=['Churn'])
df['Churn'] = df['Churn'].astype(int)
 Verificación del Balance de Clases
python
Copiar
Editar
print(df['Churn'].value_counts(normalize=True))
Resultado:

0 (No canceló): 73.46%

1 (Canceló): 26.54%

Hay un moderado desbalance de clases.

## 2. Balanceo de Clases con SMOTE
Se utilizó SMOTE (Synthetic Minority Oversampling Technique) para equilibrar la clase minoritaria.

python
Copiar
Editar
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

### Separar X e y
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

### División entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

### Imputación de NaN con la media
imputer = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

### Aplicación de SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
Verificación del balance:

python
Copiar
Editar
from collections import Counter
print(Counter(y_train_resampled))  # Resultado esperado: {0: N, 1: N}
## 3. Estandarización de los Datos
Se estandarizaron las variables con StandardScaler (media = 0, std = 1), necesario para modelos basados en distancia.

python
Copiar
Editar
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)
## 4. Análisis de Correlación
Se generó y visualizó la matriz de correlación para identificar relaciones entre variables, y especialmente con la variable Churn.

python
Copiar
Editar
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = df_encoded.corr(numeric_only=True)
corr_matrix = corr_matrix.loc[
    corr_matrix['Churn'].abs().sort_values(ascending=False).index,
    corr_matrix['Churn'].abs().sort_values(ascending=False).index
]

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(16, 12))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.1f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.7}

## Imputación de valores faltantes
Se utilizó SimpleImputer con estrategia 'mean' para rellenar los valores numéricos faltantes en X_train y X_test.

La imputación se realizó antes del balanceo de clases y se reutilizó el mismo imputador en los datos de prueba para evitar data leakage.

## Balanceo de clases con SMOTE
Se aplicó SMOTE (Synthetic Minority Over-sampling Technique) para aumentar las instancias de la clase minoritaria en y_train.

Resultado: las clases quedaron balanceadas (misma cantidad de ejemplos de cancelación y no cancelación).

## Estandarización de variables
Se usó StandardScaler para escalar X_train_resampled y X_test, dejando los datos con media 0 y desviación estándar 1.

Esto es esencial para modelos sensibles a la escala como Regresión Logística.

## Análisis Exploratorio Dirigido
Se analizaron relaciones clave entre variables predictoras y la cancelación de clientes (Churn):

Tipo de contrato vs Churn: los contratos mensuales tienen mayor tasa de cancelación.

Gasto total vs Churn: clientes que cancelan tienden a tener menor gasto acumulado.

Se usaron gráficos como boxplots, countplots y barplots para visualizar estas relaciones.

## Análisis de Correlación
Se generó una matriz de correlación mejorada, usando seaborn.heatmap, para identificar relaciones entre variables.

Se ordenaron las variables según su correlación con Churn y se aplicó una máscara triangular para legibilidad.

Variables como Contract, Tenure y MonthlyCharges mostraron alta relación con la cancelación.

## División de los Datos
Se dividieron los datos en entrenamiento (70%) y prueba (30%) usando train_test_split con estratificación para mantener la proporción de clases.

## Modelado
Se entrenaron dos modelos con estrategias distintas:

### Modelo 1: Regresión Logística
Utilizó datos imputados, balanceados y normalizados.

Ideal para tareas donde la relación entre variables es lineal.

Sensible a la escala, por lo que se normalizaron las variables.

### Modelo 2: Random Forest
Utilizó datos imputados y balanceados, sin necesidad de normalización.

Robusto ante outliers y variables en distintas escalas.

Capaz de capturar relaciones no lineales y manejar colinealidad.

## Evaluación de Modelos
Se usaron las siguientes métricas de evaluación en el conjunto de prueba:

Accuracy (exactitud)

Precision

Recall

F1-score

Matriz de confusión

## Resultados:
Métrica	Regresión Logística	Random Forest
Accuracy	0.746	0.778
Precision	0.514	0.600
Recall	0.800	0.492
F1-score	0.626	0.541

## Comparación Visual
Se generó un gráfico de barras para comparar visualmente las métricas de ambos modelos. Se observa que:

Regresión Logística tiene mejor recall y F1-score para la clase positiva (clientes que cancelan).

Random Forest obtiene mayor accuracy y precision, pero su recall para la clase minoritaria es más bajo.

## Análisis Crítico
El modelo de Regresión Logística fue más efectivo para identificar clientes que cancelan (mejor recall).

## Interpretación de Modelos
### Modelo 1: Regresión Logística
Variables clave e impacto:

tenure (antigüedad): a mayor antigüedad, menor probabilidad de churn (impacto negativo).

Charges.Total y InternetService_Fiber optic: asociados positivamente al churn.

Contract_One year y Contract_Two year: contratos a largo plazo disminuyen cancelaciones.

PaymentMethod_Electronic check: método de pago electrónico asociado a mayor churn.

Desempeño:

Accuracy: 74.6%

Recall (churn): 80.0% (alta detección de clientes que cancelan)

Precision (churn): 51.4%

### Modelo 2: Árbol de Decisión
Variables más importantes:

PaymentMethod_Electronic check es la variable más influyente.

tenure, Charges.Total, Charges.Monthly e InternetService_Fiber optic también son relevantes.

Factores demográficos como género y edad tienen impacto moderado.

Desempeño:

Accuracy: 77.8%

Recall (churn): 49.2%

Precision (churn): 60.0%

## Conclusiones
La regresión logística es mejor para identificar clientes que probablemente cancelen (mayor recall), mientras que el árbol de decisión ofrece predicciones más precisas con menos falsos positivos.

El método de pago electrónico y la antigüedad del cliente son factores claves en la cancelación.

Contratos a largo plazo (1 y 2 años) reducen el churn, lo que sugiere que incentivar estos contratos puede mejorar la retención.

Usuarios con fibra óptica presentan mayor riesgo de cancelación, por lo que es importante investigar y mejorar su experiencia.

## Recomendaciones
Fomentar contratos a largo plazo mediante promociones y ofertas atractivas.

Optimizar la experiencia y soporte para usuarios con fibra óptica.

Revisar el proceso y confianza en el método de pago electrónico.

Implementar campañas de retención para clientes nuevos o con baja antigüedad.

Monitorear clientes con altos gastos totales para evitar cancelaciones.

## Próximos pasos
Evaluar modelos adicionales como Random Forest o XGBoost para mejorar desempeño.

Explorar técnicas de ensamblaje o stacking para combinar fortalezas de varios modelos.

Integrar análisis de sentimientos o feedback del cliente para enriquecer la predicción.

Implementar un sistema de alerta para el equipo de retención basado en el modelo seleccionado.


    
)
plt.title('Matriz de Correlación (Churn)', fontsiz
