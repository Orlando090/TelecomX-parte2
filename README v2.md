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

Random Forest mostró mejor desempeño general (accuracy), pero fue menos sensible a la clase minoritaria.

Ningún modelo mostró señales claras de overfitting, ya que las métricas en test son consistentes con los resultados esperados.

Posible mejora futura: ajustar hiperparámetros, usar pipeline, probar XGBoost o usar técnicas de selección de características.
