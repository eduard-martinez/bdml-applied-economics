##============================================================================##
# week-01.R  —  Un mismo conjunto de datos, tres modelos (y un dato nuevo)
#------------------------------------------------------------------------------#
# Replica en R la diapositiva "Un mismo conjunto de datos, tres modelos" con
# 50 avisos simulados de apartamentos y muestra que le pasa a cada modelo
# cuando llega un aviso que no estaba en la muestra.
#   - input/vivienda.rds  50 avisos: area (m2) y precio (millones de pesos)
#   - input/nuevos.rds    10 avisos que llegan despues
# Fijar el directorio de trabajo en applications/week-01.
##============================================================================##

## configuracion inicial
rm(list = ls())
if (!require(pacman)) install.packages("pacman")
pacman::p_load(rio, dplyr)

## raiz del error cuadratico medio (se usa varias veces)
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

##============================================================================##
##=== 1. Los datos: 50 avisos  (grano: aviso)                              ===##
##============================================================================##

vivienda <- import("input/vivienda.rds", trust = T)
head(vivienda)

## los puntos, sin ningun modelo todavia
plot(vivienda$area, vivienda$precio, pch = 16,
     xlab = "area (m2)", ylab = "precio (millones)", main = "50 avisos")

##============================================================================##
##=== 2. Un mismo conjunto de datos, tres modelos                          ===##
##============================================================================##

## tres polinomios del area: grado 1, 3 y 12
m1  <- lm(precio ~ poly(area, 1),  data = vivienda)
m3  <- lm(precio ~ poly(area, 3),  data = vivienda)
m12 <- lm(precio ~ poly(area, 12), data = vivienda)

## una malla de areas para dibujar cada curva
malla <- data.frame(area = seq(30, 200, by = 1))
malla$grado_1  <- predict(m1, malla)
malla$grado_3  <- predict(m3, malla)
malla$grado_12 <- predict(m12, malla)

## la diapositiva, con datos
par(mfrow = c(1, 3))
plot(vivienda$area, vivienda$precio, pch = 16, ylim = c(0, 1300),
     main = "grado 1", xlab = "area", ylab = "precio")
lines(malla$area, malla$grado_1, col = "blue", lwd = 2)
plot(vivienda$area, vivienda$precio, pch = 16, ylim = c(0, 1300),
     main = "grado 3", xlab = "area", ylab = "precio")
lines(malla$area, malla$grado_3, col = "blue", lwd = 2)
plot(vivienda$area, vivienda$precio, pch = 16, ylim = c(0, 1300),
     main = "grado 12", xlab = "area", ylab = "precio")
lines(malla$area, malla$grado_12, col = "blue", lwd = 2)
par(mfrow = c(1, 1))

## error dentro de la muestra: cuanto mas flexible, mejor "ajusta"
ajuste <- data.frame(modelo      = c("grado 1", "grado 3", "grado 12"),
                     rmse_dentro = c(rmse(vivienda$precio, fitted(m1)),
                                     rmse(vivienda$precio, fitted(m3)),
                                     rmse(vivienda$precio, fitted(m12))))
ajuste

##============================================================================##
##=== 3. Llega un dato nuevo                                               ===##
##============================================================================##

## un aviso que ningun modelo vio
nuevos <- import("input/nuevos.rds", trust = T)
nuevo  <- nuevos %>% slice(1)
nuevo

## ¿que precio le pone cada modelo?
prediccion <- data.frame(modelo    = c("grado 1", "grado 3", "grado 12"),
                         predicho  = c(predict(m1, nuevo), predict(m3, nuevo), predict(m12, nuevo)),
                         observado = nuevo$precio)
prediccion

## lo agregamos a la muestra y reajustamos los tres modelos
vivienda_mas <- bind_rows(vivienda, nuevo)
m1_mas  <- lm(precio ~ poly(area, 1),  data = vivienda_mas)
m3_mas  <- lm(precio ~ poly(area, 3),  data = vivienda_mas)
m12_mas <- lm(precio ~ poly(area, 12), data = vivienda_mas)
malla$grado_1_mas  <- predict(m1_mas, malla)
malla$grado_3_mas  <- predict(m3_mas, malla)
malla$grado_12_mas <- predict(m12_mas, malla)

## antes (punteada) y despues (azul) de agregar el aviso nuevo (rojo)
par(mfrow = c(1, 3))
plot(vivienda$area, vivienda$precio, pch = 16, ylim = c(0, 1300),
     main = "grado 1", xlab = "area", ylab = "precio")
lines(malla$area, malla$grado_1, lty = 2, lwd = 2)
lines(malla$area, malla$grado_1_mas, col = "blue", lwd = 2)
points(nuevo$area, nuevo$precio, col = "red", pch = 16, cex = 2)
plot(vivienda$area, vivienda$precio, pch = 16, ylim = c(0, 1300),
     main = "grado 3", xlab = "area", ylab = "precio")
lines(malla$area, malla$grado_3, lty = 2, lwd = 2)
lines(malla$area, malla$grado_3_mas, col = "blue", lwd = 2)
points(nuevo$area, nuevo$precio, col = "red", pch = 16, cex = 2)
plot(vivienda$area, vivienda$precio, pch = 16, ylim = c(0, 1300),
     main = "grado 12", xlab = "area", ylab = "precio")
lines(malla$area, malla$grado_12, lty = 2, lwd = 2)
lines(malla$area, malla$grado_12_mas, col = "blue", lwd = 2)
points(nuevo$area, nuevo$precio, col = "red", pch = 16, cex = 2)
par(mfrow = c(1, 1))

## cuanto se movio cada curva con un solo dato
cambio <- data.frame(modelo        = c("grado 1", "grado 3", "grado 12"),
                     cambio_maximo = c(max(abs(malla$grado_1_mas - malla$grado_1)),
                                       max(abs(malla$grado_3_mas - malla$grado_3)),
                                       max(abs(malla$grado_12_mas - malla$grado_12))))
cambio

##============================================================================##
##=== 4. Llegan diez avisos nuevos                                         ===##
##============================================================================##

## ¿cual predice mejor los avisos que no vio?
error <- data.frame(modelo      = c("grado 1", "grado 3", "grado 12"),
                    rmse_dentro = ajuste$rmse_dentro,
                    rmse_nuevos = c(rmse(nuevos$precio, predict(m1, nuevos)),
                                    rmse(nuevos$precio, predict(m3, nuevos)),
                                    rmse(nuevos$precio, predict(m12, nuevos))))
error

## y si el aviso que llega es otro: reajustamos con cada uno de los diez, uno a la vez
par(mfrow = c(1, 3))
plot(vivienda$area, vivienda$precio, pch = 16, col = "grey60", ylim = c(0, 1300),
     main = "grado 1", xlab = "area", ylab = "precio")
for (i in 1:10) {
  vivienda_i <- bind_rows(vivienda, nuevos %>% slice(i))
  m_i        <- lm(precio ~ poly(area, 1), data = vivienda_i)
  lines(malla$area, predict(m_i, malla), col = adjustcolor("blue", 0.5))
}
points(nuevos$area, nuevos$precio, col = "red", pch = 16)
plot(vivienda$area, vivienda$precio, pch = 16, col = "grey60", ylim = c(0, 1300),
     main = "grado 3", xlab = "area", ylab = "precio")
for (i in 1:10) {
  vivienda_i <- bind_rows(vivienda, nuevos %>% slice(i))
  m_i        <- lm(precio ~ poly(area, 3), data = vivienda_i)
  lines(malla$area, predict(m_i, malla), col = adjustcolor("blue", 0.5))
}
points(nuevos$area, nuevos$precio, col = "red", pch = 16)
plot(vivienda$area, vivienda$precio, pch = 16, col = "grey60", ylim = c(0, 1300),
     main = "grado 12", xlab = "area", ylab = "precio")
for (i in 1:10) {
  vivienda_i <- bind_rows(vivienda, nuevos %>% slice(i))
  m_i        <- lm(precio ~ poly(area, 12), data = vivienda_i)
  lines(malla$area, predict(m_i, malla), col = adjustcolor("blue", 0.5))
}
points(nuevos$area, nuevos$precio, col = "red", pch = 16)
par(mfrow = c(1, 1))
