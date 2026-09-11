# Problem sets

Dos talleres extraclase, en parejas, con un peso de 15 % cada uno. El Problem Set 1 cubre las sesiones 1 a 5 y el Problem Set 2 las sesiones 6 a 9. Siguen la misma lógica que los problem sets del curso de Inferencia Causal: un hilo conductor, componentes con una parte conceptual corta y una parte aplicada en `R`, y una rúbrica explícita.

## Estructura común

Cada enunciado tiene cinco bloques:

1. **Información general:** fecha de entrega, peso, componentes y sus pesos, archivos de datos, política de reproducibilidad y de uso de IA.
2. **Hilo conductor:** el problema económico que atraviesa todo el taller. El PS1 trabaja sobre los precios de vivienda en Bogotá y el panel de mesas electorales; el PS2 sobre el panel de mesas electorales (Gelvez, Cardiles, Martínez-González & Muñoz, 2026) y, como opción, imágenes satelitales.
3. **Componentes:** cada uno con *contexto*, *parte conceptual* (dos o tres preguntas cortas de razonamiento) y *parte aplicada* (pasos numerados sobre los datos, con las tablas y figuras que deben aparecer en el reporte).
4. **Rúbrica** de 100 puntos: componentes (90), reproducibilidad (6) y calidad del reporte (4).
5. **Anexo de formato:** la tabla de resultados obligatoria (métrica fuera de muestra con intervalo, modelo de referencia, mejor modelo) y las figuras mínimas (predicho vs. observado, error por deciles o subgrupos).

Entregable: un único `.pdf` y un repositorio o `.zip` con un script maestro `00_run.R` que reproduce todas las tablas y figuras en un entorno limpio (rutas relativas, `sessionInfo()` o `renv`). Se permite el uso de IA con declaración explícita de qué se usó y para qué.

## Problem Set 1: Fundamentos, evaluación y regularización (sesiones 1–5)

| Componente | Sesión | Peso | Qué se hace |
|---|---|---|---|
| C1. Predecir, no explicar | 1–2 | 20 % | Por qué el error de entrenamiento es optimista; partición entrenamiento/prueba con semilla; modelos de referencia (media, OLS); $R^2$ dentro y fuera de muestra; predicho vs. observado; error por deciles del precio. |
| C2. Más allá de la línea recta | 2 | 20 % | Polinomios, *splines* y $k$-NN frente a OLS; tabla de RMSE, MAE y $R^2$ fuera de muestra; dónde falla cada modelo (extrapolación, colas). |
| C3. Clasificación con costos | 3 | 20 % | ¿Ganó el eventual ganador en la mesa? Logit vs. $k$-NN; umbral elegido por costos; matriz de confusión; ROC/AUC vs. precisión-*recall*; calibración y desbalance. |
| C4. Validación honesta | 4 | 20 % | $k$-fold vs. validación cruzada espacial; *leakage* al preprocesar fuera del *pipeline*; intervalos *bootstrap* para el RMSE; el conjunto de prueba se usa una sola vez. |
| C5. Regularización | 5 | 20 % | Muchas covariables (barrios, palabras de la descripción): OLS vs. Ridge vs. Lasso vs. Elastic Net; $\lambda$ por validación cruzada; caminos de coeficientes; "el Lasso selecciona predictores, no causas". |
| Bono (opcional) | 1–5 | +5 | Competencia: archivo `predicciones.csv` sobre un conjunto de prueba oculto; tabla de posiciones publicada por el docente. |

## Problem Set 2: Árboles, interpretación y redes (sesiones 6–9)

| Componente | Sesión | Peso | Qué se hace |
|---|---|---|---|
| C1. Árboles, bosques y *boosting* | 6 | 30 % | Árbol podado, *random forest* y XGBoost para la participación del ganador y el ganador local; afinación por validación cruzada; OOB vs. CV vs. prueba; tabla comparativa con los modelos del PS1 sobre la misma prueba. |
| C2. Abrir la caja | 7 | 30 % | Importancia por impureza vs. permutación con remuestreo; PDP, ICE y ALE; SHAP (*beeswarm*, dependencia, una mesa); ablación por bloques de predictores; error por subgrupos; qué no dice el modelo. |
| C3. Redes neuronales | 8 | 25 % | Red densa (`keras`) vs. XGBoost sobre los mismos datos tabulares: escalado, *early stopping*, *dropout*; curva de aprendizaje. Opcional (+5): rasgos de una CNN preentrenada sobre imágenes satelitales por mesa más Ridge, con validación cruzada por municipio. |
| C4. Sin $Y$ y la frontera causal | 9 | 15 % | Índice socioeconómico con PCA frente al estrato; segmentación con $k$-means; por qué un predictor seleccionado no es una causa; doble selección en dos párrafos. |

## Calendario propuesto

| Taller | Se publica | Se entrega |
|---|---|---|
| PS1 | Sesión 3 | Antes de la sesión 6 |
| PS2 | Sesión 7 | Antes de la sesión 9 |

Las fechas exactas se fijan con el calendario académico del período.

## Estado

Estructura de trabajo (septiembre de 2026). Los enunciados finales se publican en PDF en `pset-1/` y `pset-2/` junto con los archivos de datos que correspondan.
