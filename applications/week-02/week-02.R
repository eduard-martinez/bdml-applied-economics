##============================================================================##
# week-02.R  —  La regresión lineal como máquina de predecir
#------------------------------------------------------------------------------#
# El caso del curso: ¿cuánto del voto de un puesto de votación se puede
# predecir con el censo de su entorno? Segunda vuelta de 2022, Petro vs.
# Hernández; Cali, Palmira, Yumbo y Jamundí. El guion de la práctica:
#   1. los datos y la partición entrenamiento/prueba (semilla 2026)
#   2. la referencia (la media) y la recta con una variable, con su RMSE
#   3. los residuos de la recta: contra el ajustado, por municipio, en el mapa
#   4. varios modelos con más variables, todos contra la misma prueba
#   5. el mejor modelo y sus predicciones
# La base se lee directo del repositorio del curso en GitHub: no hay que
# descargar nada.
##============================================================================##

## configuracion inicial
rm(list = ls())
if (!require(pacman)) install.packages("pacman")
pacman::p_load(rio, dplyr)

## las tres metricas del curso
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
mae <- function(y, yhat) mean(abs(y - yhat))
r2_prueba <- function(y, yhat, y_train) 1 - sum((y - yhat)^2) / sum((y - mean(y_train))^2)

## promedio de los k puestos mas cercanos en el mapa (se usa en la seccion 4)
knn_reg <- function(coord_train, y_train, coord_nueva, k) {
           prediccion <- rep(NA, nrow(coord_nueva))
           for (i in 1:nrow(coord_nueva)) {
             distancia <- sqrt((coord_train[, 1] - coord_nueva[i, 1])^2 + (coord_train[, 2] - coord_nueva[i, 2])^2)
             vecinos <- order(distancia)[1:k]
             prediccion[i] <- mean(y_train[vecinos])
           }
           return(prediccion)
}

##============================================================================##
##=== 1. Los datos y la particion (semilla 2026)                           ===##
##============================================================================##

## un puesto por fila, con sus votos y su entorno censal, desde el repositorio
valle <- import("https://raw.githubusercontent.com/eduard-martinez/bdml-applied-economics/main/applications/week-02/input/puestos_valle_2022.rds", trust = T)

## el caso de hoy: Cali y su area metropolitana
metro <- valle %>%
         filter(municipio %in% c("CALI", "PALMIRA", "YUMBO", "JAMUNDI"))
nrow(metro)

## la particion: 75% para aprender y 25% bajo llave hasta la seccion 4
set.seed(2026)
prueba_id <- sample(nrow(metro), size = round(0.25 * nrow(metro)))
train <- metro[-prueba_id, ]
test <- metro[prueba_id, ]
c(entrenamiento = nrow(train), prueba = nrow(test))

##============================================================================##
##=== 2. La referencia y la recta con una variable                         ===##
##============================================================================##

## la referencia que no usa ninguna x: predecir la media del entrenamiento
media_train <- mean(train$voto_petro)
media_train

## la recta: el voto contra la educacion superior del entorno
m1 <- lm(voto_petro ~ educ_superior, data = train)
summary(m1)

## la nube y la recta de minimos cuadrados
plot(train$educ_superior, train$voto_petro, pch = 16, col = adjustcolor("black", 0.5),
     xlab = "Educación superior en el entorno (%)", ylab = "Voto por Petro (%)",
     main = "257 puestos de entrenamiento y la recta")
abline(m1, col = "blue", lwd = 2)

## el rmse dentro y en la prueba, contra la referencia
data.frame(modelo = c("media del entrenamiento", "recta (educación superior)"),
           rmse_dentro = c(rmse(train$voto_petro, rep(media_train, nrow(train))),
                           rmse(train$voto_petro, predict(m1, train))),
           rmse_prueba = c(rmse(test$voto_petro, rep(media_train, nrow(test))),
                           rmse(test$voto_petro, predict(m1, test))))

##============================================================================##
##=== 3. Los residuos de la recta: ¿que le falta?                          ===##
##============================================================================##

train$ajustado <- predict(m1, train)
train$residuo <- train$voto_petro - train$ajustado

## contra el ajustado: la suavizada queda plana, la forma lineal basta
plot(train$ajustado, train$residuo, pch = 16, col = adjustcolor("black", 0.5),
     xlab = "Voto ajustado (%)", ylab = "Residuo (puntos)",
     main = "Residuos contra ajustados: sin patrón")
abline(h = 0, lty = 2)
lines(lowess(train$ajustado, train$residuo), col = "red", lwd = 2)

## por municipio: la recta le queda corta a Jamundi y a Yumbo, y le sobra en Palmira
train %>%
  group_by(municipio) %>%
  summarise(puestos = n(), residuo_medio = round(mean(residuo), 1), .groups = "drop") %>%
  arrange(residuo_medio)

## en el mapa: manchas de un mismo signo = a la recta le falta la ubicacion
plot(train$lon, train$lat, pch = 16, col = ifelse(train$residuo > 0, "firebrick", "steelblue"),
     cex = 0.5 + abs(train$residuo) / 10, xlab = "Longitud", ylab = "Latitud",
     main = "Residuos: votó más (rojo) o menos (azul) de lo que dice la educación")

##============================================================================##
##=== 4. Varios modelos, la misma prueba                                   ===##
##============================================================================##

## mas flexibilidad sobre la misma variable: la cubica y el grado 12
m3 <- lm(voto_petro ~ poly(educ_superior, 3), data = train)
m12 <- lm(voto_petro ~ poly(educ_superior, 12), data = train)

## mas variables: nueve columnas del censo (estrato, edad, mujeres, afro,
## servicios y tamano del puesto)
m9 <- lm(voto_petro ~ educ_superior + estrato_bajo + estrato_alto + edad_20_29 + edad_60_74 +
                      mujeres + afro + servicios + log(registrados), data = train)
round(coef(summary(m9)), 3)

## sin formula: el promedio de los k puestos mas cercanos en el mapa
coord_train <- as.matrix(train[, c("lon", "lat")])
coord_test <- as.matrix(test[, c("lon", "lat")])

## las predicciones de cada modelo, una sola vez, sobre la misma prueba
pred_test <- list(rep(media_train, nrow(test)), predict(m1, test), predict(m3, test),
                  predict(m12, test), predict(m9, test),
                  knn_reg(coord_train, train$voto_petro, coord_test, k = 1),
                  knn_reg(coord_train, train$voto_petro, coord_test, k = 5),
                  knn_reg(coord_train, train$voto_petro, coord_test, k = 25))
pred_train <- list(rep(media_train, nrow(train)), predict(m1, train), predict(m3, train),
                   predict(m12, train), predict(m9, train),
                   knn_reg(coord_train, train$voto_petro, coord_train, k = 1),
                   knn_reg(coord_train, train$voto_petro, coord_train, k = 5),
                   knn_reg(coord_train, train$voto_petro, coord_train, k = 25))

## la tabla canonica: k = 1 es el mejor dentro y de los peores fuera; ocho
## variables mas apenas mueven la aguja frente a la recta
tabla <- data.frame(modelo = c("media del entrenamiento", "recta (educación superior)", "cúbica",
                               "grado 12", "nueve variables", "k-NN mapa, k = 1",
                               "k-NN mapa, k = 5", "k-NN mapa, k = 25"),
                    rmse_dentro = sapply(pred_train, function(p) rmse(train$voto_petro, p)),
                    rmse_prueba = sapply(pred_test, function(p) rmse(test$voto_petro, p)),
                    mae_prueba = sapply(pred_test, function(p) mae(test$voto_petro, p)),
                    r2_prueba = sapply(pred_test, function(p) r2_prueba(test$voto_petro, p, train$voto_petro)))
tabla %>%
  mutate(across(where(is.numeric), ~round(.x, 2))) %>%
  arrange(rmse_prueba)

##============================================================================##
##=== 5. El mejor modelo y sus predicciones                                ===##
##============================================================================##

## gana por poco el de nueve variables (6,76 contra 6,83 de la recta); con el
## cerramos la semana
mejor <- m9
test$prediccion <- predict(mejor, test)
test$error <- test$voto_petro - test$prediccion

## rmse final, una sola vez, en la prueba
rmse(test$voto_petro, test$prediccion)

## predicho contra observado: la diagonal es la prediccion perfecta; las
## predicciones se comprimen hacia la media
plot(test$voto_petro, test$prediccion, pch = 16, col = "blue", xlim = c(10, 100), ylim = c(10, 100),
     xlab = "Voto observado (%)", ylab = "Voto predicho (%)",
     main = "El mejor modelo en los 86 puestos de prueba")
abline(a = 0, b = 1, lty = 2)

## los puestos que el modelo clava...
test %>%
  mutate(voto_petro = round(voto_petro, 1), prediccion = round(prediccion, 1), error = round(error, 1)) %>%
  select(nombre, municipio, voto_petro, prediccion, error) %>%
  arrange(abs(error)) %>%
  head(5)

## ...y los que se le escapan: casi todos fuera de Cali, donde la ubicacion
## dice cosas que el censo no (el tema de la sesion 3)
test %>%
  mutate(voto_petro = round(voto_petro, 1), prediccion = round(prediccion, 1), error = round(error, 1)) %>%
  select(nombre, municipio, voto_petro, prediccion, error) %>%
  arrange(desc(abs(error))) %>%
  head(5)
