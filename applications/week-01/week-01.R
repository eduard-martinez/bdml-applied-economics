##============================================================================##
# week-01.R  —  Predecir, no explicar: sesgo y varianza con datos
#------------------------------------------------------------------------------#
# Lee input/vivienda.rds (grano: aviso; avisos simulados con una f conocida) y
# muestra, con polinomios del area:
#   1. bondad de ajuste dentro vs. fuera de muestra (tres modelos)
#   2. la curva U
#   3. que pasa cuando llega un dato nuevo
#   4. sesgo y varianza repitiendo el muestreo
##============================================================================##

## configuracion inicial
rm(list = ls())
if (!require(pacman)) install.packages("pacman")
pacman::p_load(rio, dplyr)
set.seed(10993)

## path
inp <- "input"

##============================================================================##
##=== 1. Datos y particion  (grano: aviso)                                 ===##
##============================================================================##

## avisos simulados: precio (millones de pesos) y area (m2)
vivienda <- import(file.path(inp, "vivienda.rds"), trust = T) %>%
            mutate(log_precio = log(precio),
                   log_area   = log(area))
head(vivienda)
summary(vivienda)

## en logaritmos la relacion es casi una recta. casi.
plot(vivienda$log_area, vivienda$log_precio, pch = 16, col = "grey70",
     xlab = "log(area)", ylab = "log(precio)")

## particion antes de tocar cualquier modelo: 100 avisos para entrenar, el resto para probar
id_train <- sample(nrow(vivienda), 100)
train    <- vivienda[id_train, ]
test     <- vivienda[-id_train, ]
nrow(train)
nrow(test)

##============================================================================##
##=== 2. Tres modelos sobre los mismos 100 puntos                          ===##
##============================================================================##

## polinomios del log(area) de grado 1, 3 y 10
m1  <- lm(log_precio ~ poly(log_area, 1), data = train)
m3  <- lm(log_precio ~ poly(log_area, 3), data = train)
m10 <- lm(log_precio ~ poly(log_area, 10), data = train)

## malla para dibujar las curvas
malla <- data.frame(log_area = seq(min(train$log_area), max(train$log_area), length.out = 200))

plot(train$log_area, train$log_precio, pch = 16, xlab = "log(area)", ylab = "log(precio)")
lines(malla$log_area, predict(m1, malla), col = "red", lwd = 2)
lines(malla$log_area, predict(m3, malla), col = "darkgreen", lwd = 2)
lines(malla$log_area, predict(m10, malla), col = "blue", lwd = 2)
legend("topleft", legend = c("grado 1", "grado 3", "grado 10"),
       col = c("red", "darkgreen", "blue"), lwd = 2)

## bondad de ajuste dentro de la muestra y en la prueba
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
r2   <- function(y, yhat) 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)

bondad <- data.frame(modelo     = c("grado 1", "grado 3", "grado 10"),
                     r2_train   = c(r2(train$log_precio, fitted(m1)),
                                    r2(train$log_precio, fitted(m3)),
                                    r2(train$log_precio, fitted(m10))),
                     r2_test    = c(r2(test$log_precio, predict(m1, test)),
                                    r2(test$log_precio, predict(m3, test)),
                                    r2(test$log_precio, predict(m10, test))),
                     rmse_train = c(rmse(train$log_precio, fitted(m1)),
                                    rmse(train$log_precio, fitted(m3)),
                                    rmse(train$log_precio, fitted(m10))),
                     rmse_test  = c(rmse(test$log_precio, predict(m1, test)),
                                    rmse(test$log_precio, predict(m3, test)),
                                    rmse(test$log_precio, predict(m10, test))))
bondad

## pregunta para la sala: cual de los tres entregarian y con que numero lo defenderian?

##============================================================================##
##=== 3. La curva U: grado 1 a 12                                          ===##
##============================================================================##

grados     <- 1:12
rmse_train <- rep(NA, 12)
rmse_test  <- rep(NA, 12)
for (d in grados) {
  m             <- lm(log_precio ~ poly(log_area, d), data = train)
  rmse_train[d] <- rmse(train$log_precio, fitted(m))
  rmse_test[d]  <- rmse(test$log_precio, predict(m, test))
}
curva_u <- data.frame(grado = grados, rmse_train = rmse_train, rmse_test = rmse_test)
curva_u

## el error de entrenamiento baja siempre; el de prueba tiene forma de U
plot(grados, rmse_train, type = "b", col = "blue", pch = 16, ylim = c(0.15, 0.6),
     xlab = "grado del polinomio", ylab = "RMSE de log(precio)")
lines(grados, rmse_test, type = "b", col = "red", pch = 16)
legend("topleft", legend = c("entrenamiento", "prueba"), col = c("blue", "red"), lwd = 1)

##============================================================================##
##=== 4. Que pasa cuando llega un dato nuevo                               ===##
##============================================================================##

## llega un aviso que ningun modelo vio: uno de la prueba, de unos 80 m2
nuevo <- test %>%
         filter(area >= 78, area <= 82) %>%
         slice(1)
nuevo

## que precio le pone cada modelo
prediccion <- data.frame(modelo    = c("grado 1", "grado 3", "grado 10"),
                         predicho  = exp(c(predict(m1, nuevo), predict(m3, nuevo), predict(m10, nuevo))),
                         observado = nuevo$precio)
prediccion

## lo agregamos al entrenamiento y reajustamos
train_mas <- bind_rows(train, nuevo)
m1_mas    <- lm(log_precio ~ poly(log_area, 1), data = train_mas)
m10_mas   <- lm(log_precio ~ poly(log_area, 10), data = train_mas)

## antes (punteado) y despues (continuo) de agregar el punto rojo
par(mfrow = c(1, 2))
plot(train$log_area, train$log_precio, pch = 16, col = "grey60", main = "grado 1",
     xlab = "log(area)", ylab = "log(precio)")
points(nuevo$log_area, nuevo$log_precio, col = "red", pch = 16, cex = 1.5)
lines(malla$log_area, predict(m1, malla), lty = 2)
lines(malla$log_area, predict(m1_mas, malla), col = "red", lwd = 2)
plot(train$log_area, train$log_precio, pch = 16, col = "grey60", main = "grado 10",
     xlab = "log(area)", ylab = "log(precio)")
points(nuevo$log_area, nuevo$log_precio, col = "red", pch = 16, cex = 1.5)
lines(malla$log_area, predict(m10, malla), lty = 2)
lines(malla$log_area, predict(m10_mas, malla), col = "blue", lwd = 2)
par(mfrow = c(1, 1))

## ahora un aviso poco comun: un apartamento grande
grande <- test %>%
          filter(area >= 250) %>%
          slice(1)
grande
exp(c(predict(m1, grande), predict(m3, grande), predict(m10, grande)))

train_mas <- bind_rows(train, grande)
m1_mas    <- lm(log_precio ~ poly(log_area, 1), data = train_mas)
m10_mas   <- lm(log_precio ~ poly(log_area, 10), data = train_mas)
malla_mas <- data.frame(log_area = seq(min(train_mas$log_area), max(train_mas$log_area), length.out = 200))

par(mfrow = c(1, 2))
plot(train$log_area, train$log_precio, pch = 16, col = "grey60", main = "grado 1",
     xlab = "log(area)", ylab = "log(precio)", xlim = range(malla_mas$log_area))
points(grande$log_area, grande$log_precio, col = "red", pch = 16, cex = 1.5)
lines(malla_mas$log_area, predict(m1, malla_mas), lty = 2)
lines(malla_mas$log_area, predict(m1_mas, malla_mas), col = "red", lwd = 2)
plot(train$log_area, train$log_precio, pch = 16, col = "grey60", main = "grado 10",
     xlab = "log(area)", ylab = "log(precio)", xlim = range(malla_mas$log_area))
points(grande$log_area, grande$log_precio, col = "red", pch = 16, cex = 1.5)
lines(malla_mas$log_area, predict(m10, malla_mas), lty = 2)
lines(malla_mas$log_area, predict(m10_mas, malla_mas), col = "blue", lwd = 2)
par(mfrow = c(1, 1))

## veinte veces: diez avisos nuevos al azar, y se reajusta. esto es la varianza
par(mfrow = c(1, 2))
plot(train$log_area, train$log_precio, pch = 16, col = "grey70", main = "grado 1",
     xlab = "log(area)", ylab = "log(precio)")
for (r in 1:20) {
  train_r <- bind_rows(train, test[sample(nrow(test), 10), ])
  m_r     <- lm(log_precio ~ poly(log_area, 1), data = train_r)
  lines(malla$log_area, predict(m_r, malla), col = adjustcolor("red", 0.4))
}
plot(train$log_area, train$log_precio, pch = 16, col = "grey70", main = "grado 10",
     xlab = "log(area)", ylab = "log(precio)")
for (r in 1:20) {
  train_r <- bind_rows(train, test[sample(nrow(test), 10), ])
  m_r     <- lm(log_precio ~ poly(log_area, 10), data = train_r)
  lines(malla$log_area, predict(m_r, malla), col = adjustcolor("blue", 0.4))
}
par(mfrow = c(1, 1))

##============================================================================##
##=== 5. Sesgo y varianza repitiendo el muestreo                           ===##
##============================================================================##

## los datos se simularon con f(la) = 6.2 + 0.9(la - 4.5) - 0.30(la - 4.5)^2 + 0.15(la - 4.5)^3
## y ruido con sd = 0.25 (ver 00_generar_datos.R). fijamos x0 = un apartamento de 80 m2
x0   <- log(80)
f_x0 <- 6.2 + 0.9 * (x0 - 4.5) - 0.30 * (x0 - 4.5)^2 + 0.15 * (x0 - 4.5)^3

## 200 muestras de entrenamiento de 100 avisos; prediccion en x0 con cada grado
pred_1  <- rep(NA, 200)
pred_3  <- rep(NA, 200)
pred_10 <- rep(NA, 200)
for (r in 1:200) {
  muestra    <- vivienda[sample(nrow(vivienda), 100), ]
  pred_1[r]  <- predict(lm(log_precio ~ poly(log_area, 1), data = muestra), data.frame(log_area = x0))
  pred_3[r]  <- predict(lm(log_precio ~ poly(log_area, 3), data = muestra), data.frame(log_area = x0))
  pred_10[r] <- predict(lm(log_precio ~ poly(log_area, 10), data = muestra), data.frame(log_area = x0))
}

## sesgo al cuadrado, varianza y error total en x0
sesgo_varianza <- data.frame(modelo   = c("grado 1", "grado 3", "grado 10"),
                             sesgo2   = c((mean(pred_1) - f_x0)^2, (mean(pred_3) - f_x0)^2, (mean(pred_10) - f_x0)^2),
                             varianza = c(var(pred_1), var(pred_3), var(pred_10)),
                             sigma2   = 0.25^2) %>%
                  mutate(mse = sesgo2 + varianza + sigma2)
sesgo_varianza

## las cuatro dianas, con datos: la linea punteada es f(x0)
par(mfrow = c(3, 1))
hist(pred_1, breaks = 30, col = "salmon", main = "grado 1", xlab = "prediccion en x0", xlim = c(5.7, 6.5))
abline(v = f_x0, lty = 2, lwd = 2)
hist(pred_3, breaks = 30, col = "lightgreen", main = "grado 3", xlab = "prediccion en x0", xlim = c(5.7, 6.5))
abline(v = f_x0, lty = 2, lwd = 2)
hist(pred_10, breaks = 30, col = "lightblue", main = "grado 10", xlab = "prediccion en x0", xlim = c(5.7, 6.5))
abline(v = f_x0, lty = 2, lwd = 2)
par(mfrow = c(1, 1))
