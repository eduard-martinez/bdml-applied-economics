##============================================================================##
# week-01.R  —  Un mismo conjunto de datos, tres modelos (y un dato nuevo)
#------------------------------------------------------------------------------#
# Replica en R la diapositiva "Un mismo conjunto de datos, tres modelos" con
# 50 avisos simulados de apartamentos y muestra que le pasa a cada modelo
# cuando llega un aviso que no estaba en la muestra.
# Lee:
#   - input/vivienda.rds  (grano: aviso; 50 avisos con area en m2 y precio en
#                          millones de pesos: la muestra con la que ajustamos)
#   - input/nuevos.rds    (grano: aviso; 10 avisos que llegan despues y que
#                          ningun modelo vio)
# No produce archivos: solo tablas en consola y graficos.
# Fijar el directorio de trabajo en applications/week-01 (en RStudio: Session >
# Set Working Directory); las rutas son relativas a esa carpeta.
##============================================================================##

## configuracion inicial: pacman instala (si falta) y carga los paquetes;
## rio para leer los datos, dplyr para slice() y bind_rows()
rm(list = ls())
if (!require(pacman)) install.packages("pacman")
pacman::p_load(rio, dplyr)

## raiz del error cuadratico medio (en millones, como el precio). unica funcion
## del script: se usa en las secciones 2 y 4
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

##============================================================================##
##=== 1. Los datos: 50 avisos  (grano: aviso)                              ===##
##============================================================================##

## la muestra: 50 avisos con area (m2) y precio (millones). trust = T porque
## rio pide confirmar que el .rds viene de una fuente confiable
vivienda <- import("input/vivienda.rds", trust = T)
head(vivienda)

## los puntos, sin ningun modelo todavia: ¿que forma tiene la relacion?
plot(vivienda$area, vivienda$precio, pch = 16,
     xlab = "area (m2)", ylab = "precio (millones)", main = "50 avisos")

##============================================================================##
##=== 2. Un mismo conjunto de datos, tres modelos                          ===##
##============================================================================##

## tres polinomios del area sobre los mismos 50 avisos: grado 1 (la recta),
## grado 3 y grado 12. poly() usa el polinomio ortogonal, que es numericamente
## estable con grados altos; la curva ajustada es la misma que con area^k
m1  <- lm(precio ~ poly(area, 1),  data = vivienda)
m3  <- lm(precio ~ poly(area, 3),  data = vivienda)
m12 <- lm(precio ~ poly(area, 12), data = vivienda)

## una malla de areas (30, 31, ..., 200) para dibujar cada curva: predict()
## evalua cada modelo en areas que no tienen por que estar en la muestra
malla <- data.frame(area = seq(30, 200, by = 1))
malla$grado_1  <- predict(m1, malla)
malla$grado_3  <- predict(m3, malla)
malla$grado_12 <- predict(m12, malla)

## la diapositiva, con datos: la recta no sigue la curvatura (sesgo); la de
## grado 12 persigue los puntos, sobre todo en los extremos (varianza)
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

## error dentro de la muestra: el rmse sobre los mismos 50 avisos con que se
## ajusto cada modelo. baja siempre con el grado, porque un polinomio de grado
## mayor contiene al de grado menor. "ajustar mejor" no es "predecir mejor"
## (regla de oro #1: el error de entrenamiento es optimista)
ajuste <- data.frame(modelo      = c("grado 1", "grado 3", "grado 12"),
                     rmse_dentro = c(rmse(vivienda$precio, fitted(m1)),
                                     rmse(vivienda$precio, fitted(m3)),
                                     rmse(vivienda$precio, fitted(m12))))
ajuste

##============================================================================##
##=== 3. Llega un dato nuevo                                               ===##
##============================================================================##

## los avisos que llegan despues; nos quedamos con el primero. ningun modelo
## lo vio, asi que el error en el si dice algo sobre predecir
nuevos <- import("input/nuevos.rds", trust = T)
nuevo  <- nuevos %>% slice(1)
nuevo

## ¿que precio le pone cada modelo? predict() con los modelos ya ajustados,
## sin tocarlos; la ultima columna es el precio observado del aviso
prediccion <- data.frame(modelo    = c("grado 1", "grado 3", "grado 12"),
                         predicho  = c(predict(m1, nuevo), predict(m3, nuevo), predict(m12, nuevo)),
                         observado = nuevo$precio)
prediccion

## ahora lo agregamos a la muestra (51 avisos) y volvemos a estimar los tres
## modelos desde cero: ¿cuanto cambia cada curva por un solo dato?
vivienda_mas <- bind_rows(vivienda, nuevo)
m1_mas  <- lm(precio ~ poly(area, 1),  data = vivienda_mas)
m3_mas  <- lm(precio ~ poly(area, 3),  data = vivienda_mas)
m12_mas <- lm(precio ~ poly(area, 12), data = vivienda_mas)
malla$grado_1_mas  <- predict(m1_mas, malla)
malla$grado_3_mas  <- predict(m3_mas, malla)
malla$grado_12_mas <- predict(m12_mas, malla)

## antes (punteada) y despues (azul) de agregar el aviso nuevo (rojo): la recta
## y la cubica casi no se mueven; la de grado 12 cambia de forma
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

## el cambio maximo de cada curva sobre la malla, en millones: cuanto depende
## cada modelo de un solo dato
cambio <- data.frame(modelo        = c("grado 1", "grado 3", "grado 12"),
                     cambio_maximo = c(max(abs(malla$grado_1_mas - malla$grado_1)),
                                       max(abs(malla$grado_3_mas - malla$grado_3)),
                                       max(abs(malla$grado_12_mas - malla$grado_12))))
cambio

##============================================================================##
##=== 4. Llegan diez avisos nuevos                                         ===##
##============================================================================##

## error dentro (los 50 avisos del ajuste) vs. error en los diez avisos nuevos
## (que no participaron en el ajuste): el orden se invierte. el que mejor
## ajustaba dentro es el que peor predice fuera
error <- data.frame(modelo      = c("grado 1", "grado 3", "grado 12"),
                    rmse_dentro = ajuste$rmse_dentro,
                    rmse_nuevos = c(rmse(nuevos$precio, predict(m1, nuevos)),
                                    rmse(nuevos$precio, predict(m3, nuevos)),
                                    rmse(nuevos$precio, predict(m12, nuevos))))
error

## y si el aviso que llega es otro: reajustamos con cada uno de los diez, uno
## a la vez (50 avisos + el aviso i), y dibujamos las diez curvas de cada
## modelo. diez rectas iguales (siempre el mismo error: sesgo), diez cubicas
## casi iguales, diez curvas de grado 12 distintas (varianza)
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
