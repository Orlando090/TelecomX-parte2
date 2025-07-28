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
)
plt.title('Matriz de Correlación (Churn)', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
