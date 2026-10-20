## Big Data y Machine Learning para Economia Aplicada
## week-01: un mismo conjunto de datos, tres modelos (y un dato nuevo)

##==: 0. initial setup :==##

## clean environment
rm(list = ls())

## load/install packages
if (!require(pacman)) install.packages("pacman")
pacman::p_load(rio, dplyr)

## la unica funcion del script: la raiz del error cuadratico medio
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

##==: 1. los datos: 50 avisos :==##

## 1.1. load data: 50 avisos con area (m2) y precio (millones)
vivienda <- import("input/vivienda.rds", trust = T)
head(vivienda)

## 1.2. los puntos, sin ningun modelo todavia: ¿que forma tiene la relacion?
plot(vivienda$area, vivienda$precio, pch = 16,
     xlab = "area (m2)", ylab = "precio (millones)", main = "50 avisos")

##==: 2. un mismo conjunto de datos, tres modelos :==##

## 2.1. tres polinomios del area sobre los mismos 50 avisos: grados 1, 3 y 12
m1  <- lm(precio ~ poly(area, 1),  data = vivienda)
m3  <- lm(precio ~ poly(area, 3),  data = vivienda)
m12 <- lm(precio ~ poly(area, 12), data = vivienda)

## 2.2. una malla de areas (30 a 200) para dibujar cada curva
malla <- data.frame(area = seq(30, 200, by = 1))
malla$grado_1  <- predict(m1, malla)
malla$grado_3  <- predict(m3, malla)
malla$grado_12 <- predict(m12, malla)

## 2.3. la diapositiva, con datos: sesgo en la recta, varianza en el grado 12
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

## 2.4. el rmse dentro de la muestra: baja siempre con el grado (regla de oro #1)
ajuste <- data.frame(modelo      = c("grado 1", "grado 3", "grado 12"),
                     rmse_dentro = c(rmse(vivienda$precio, fitted(m1)),
                                     rmse(vivienda$precio, fitted(m3)),
                                     rmse(vivienda$precio, fitted(m12))))
ajuste

##==: 3. llega un dato nuevo :==##

## 3.1. los avisos que llegan despues: nos quedamos con el primero
nuevos <- import("input/nuevos.rds", trust = T)
nuevo  <- nuevos %>% slice(1)
nuevo

## 3.2. ¿que precio le pone cada modelo? predict() sin tocar los modelos
prediccion <- data.frame(modelo    = c("grado 1", "grado 3", "grado 12"),
                         predicho  = c(predict(m1, nuevo), predict(m3, nuevo), predict(m12, nuevo)),
                         observado = nuevo$precio)
prediccion

## 3.3. agregarlo a la muestra (51 avisos) y reestimar los tres desde cero
vivienda_mas <- bind_rows(vivienda, nuevo)
m1_mas  <- lm(precio ~ poly(area, 1),  data = vivienda_mas)
m3_mas  <- lm(precio ~ poly(area, 3),  data = vivienda_mas)
m12_mas <- lm(precio ~ poly(area, 12), data = vivienda_mas)
malla$grado_1_mas  <- predict(m1_mas, malla)
malla$grado_3_mas  <- predict(m3_mas, malla)
malla$grado_12_mas <- predict(m12_mas, malla)

## 3.4. antes (punteada) y despues (azul) de agregar el aviso nuevo (rojo)
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

## 3.5. el cambio maximo de cada curva: cuanto depende cada modelo de un solo dato
cambio <- data.frame(modelo        = c("grado 1", "grado 3", "grado 12"),
                     cambio_maximo = c(max(abs(malla$grado_1_mas - malla$grado_1)),
                                       max(abs(malla$grado_3_mas - malla$grado_3)),
                                       max(abs(malla$grado_12_mas - malla$grado_12))))
cambio

##==: 4. llegan diez avisos nuevos :==##

## 4.1. rmse dentro vs rmse en los diez nuevos: el orden se invierte
error <- data.frame(modelo      = c("grado 1", "grado 3", "grado 12"),
                    rmse_dentro = ajuste$rmse_dentro,
                    rmse_nuevos = c(rmse(nuevos$precio, predict(m1, nuevos)),
                                    rmse(nuevos$precio, predict(m3, nuevos)),
                                    rmse(nuevos$precio, predict(m12, nuevos))))
error

## 4.2. reajustar con cada aviso, uno a la vez: las diez curvas de cada modelo (la varianza en vivo)
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
