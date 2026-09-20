##============================================================================##
# week-02.R  —  La regresión lineal como máquina de predecir
#------------------------------------------------------------------------------#
# El problema del curso: predecir el voto de un puesto de votación a partir del
# entorno del puesto (segunda vuelta presidencial de 2022, Petro vs. Hernández).
# Qué lee (carpeta input/, grano: puesto de votación):
#   - puestos_valle_2022.rds     los 1.060 puestos del Valle del Cauca
#   - puestos_colombia_2022.rds  los 12.001 puestos del país (para el cierre)
#   - diccionario_puestos.csv    qué es cada variable
# Qué produce: gráficos en pantalla y tablas en consola; no exporta nada.
# Fijar el directorio de trabajo en applications/week-02.
##============================================================================##

## configuracion inicial
rm(list = ls())
if (!require(pacman)) install.packages("pacman")
pacman::p_load(rio, dplyr)

## tres metricas de prueba que usamos en todo el curso
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
mae <- function(y, yhat) mean(abs(y - yhat))
r2_prueba <- function(y, yhat, y_train) 1 - sum((y - yhat)^2) / sum((y - mean(y_train))^2)

##============================================================================##
##=== 1. Los datos: Cali y su área metropolitana                            ===##
##============================================================================##

## el Valle del Cauca: un puesto por fila, con sus votos y su entorno censal
valle <- import("input/puestos_valle_2022.rds", trust = T)
diccionario <- import("input/diccionario_puestos.csv")
diccionario

## empezamos por Cali, Palmira, Yumbo y Jamundi: 343 puestos que todos conocemos
metro <- valle %>%
         filter(municipio %in% c("CALI", "PALMIRA", "YUMBO", "JAMUNDI"))
nrow(metro)
summary(metro[, c("voto_petro", "participacion", "educ_superior", "estrato_bajo", "estrato_alto", "afro")])

## la y: voto por Petro sobre votos por los dos candidatos; va del 15% (Pance) al 94% (La Meseta, en Jamundi)
hist(metro$voto_petro, breaks = 20, col = "gray85", xlab = "Voto por Petro (%)", main = "343 puestos: la variable que queremos predecir")
metro %>% arrange(voto_petro) %>% select(nombre, municipio, voto_petro, educ_superior, estrato_alto) %>% head(3)
metro %>% arrange(desc(voto_petro)) %>% select(nombre, municipio, voto_petro, educ_superior, estrato_alto) %>% head(3)

## el mapa: el sur y el oriente de Cali votan distinto del sur-occidente
plot(metro$lon, metro$lat, pch = 16, col = ifelse(metro$voto_petro > 50, "firebrick", "steelblue"),
     xlab = "Longitud", ylab = "Latitud", main = "Petro gana (rojo) o pierde (azul) el puesto")

## la particion: 75% para aprender, 25% bajo llave hasta la seccion 5
set.seed(2026)
prueba_id <- sample(nrow(metro), size = round(0.25 * nrow(metro)))
train <- metro[-prueba_id, ]
test <- metro[prueba_id, ]
nrow(train)
nrow(test)

##============================================================================##
##=== 2. La referencia y la recta con una variable                          ===##
##============================================================================##

## la referencia que no usa x: predecir la media del entrenamiento para todos
media_train <- mean(train$voto_petro)
media_train

## la recta: el voto cae con la educacion superior del entorno
m1 <- lm(voto_petro ~ educ_superior, data = train)
summary(m1)

plot(train$educ_superior, train$voto_petro, pch = 16, xlab = "Educación superior en el entorno (%)",
     ylab = "Voto por Petro (%)", main = "Una recta explica el 64% de la variación")
abline(m1, col = "blue", lwd = 2)

## el error dentro del entrenamiento y en la prueba (una sola vez cada modelo)
rmse(train$voto_petro, predict(m1, train))
rmse(test$voto_petro, predict(m1, test))

##============================================================================##
##=== 3. Predecir la media o predecir un puesto nuevo                       ===##
##============================================================================##

## el intervalo de confianza cubre la media f(x); el de prediccion cubre un
## puesto nuevo y carga el ruido sigma: en un puesto con 10% de educacion superior
malla <- data.frame(educ_superior = seq(0, 60, by = 1))
ic <- predict(m1, malla, interval = "confidence")
ip <- predict(m1, malla, interval = "prediction")

plot(train$educ_superior, train$voto_petro, pch = 16, col = adjustcolor("black", 0.5),
     xlab = "Educación superior en el entorno (%)", ylab = "Voto por Petro (%)",
     main = "Intervalo de confianza (azul) y de predicción (rojo)", ylim = c(0, 100))
lines(malla$educ_superior, ic[, "fit"], col = "blue", lwd = 2)
lines(malla$educ_superior, ic[, "lwr"], col = "blue", lty = 2)
lines(malla$educ_superior, ic[, "upr"], col = "blue", lty = 2)
lines(malla$educ_superior, ip[, "lwr"], col = "red", lty = 2)
lines(malla$educ_superior, ip[, "upr"], col = "red", lty = 2)

## con la cuarta parte de los puestos la banda azul se abre; la roja casi no cambia:
## el ruido no se aprende con mas datos
set.seed(2026)
pocos <- train[sample(nrow(train), size = 65), ]
m1_pocos <- lm(voto_petro ~ educ_superior, data = pocos)
ic_pocos <- predict(m1_pocos, malla, interval = "confidence")
ip_pocos <- predict(m1_pocos, malla, interval = "prediction")

fila <- which(malla$educ_superior == 10)
anchos <- data.frame(n = c(nrow(pocos), nrow(train)),
                     ancho_confianza = c(ic_pocos[fila, "upr"] - ic_pocos[fila, "lwr"], ic[fila, "upr"] - ic[fila, "lwr"]),
                     ancho_prediccion = c(ip_pocos[fila, "upr"] - ip_pocos[fila, "lwr"], ip[fila, "upr"] - ip[fila, "lwr"]))
anchos

##============================================================================##
##=== 4. La regresión múltiple y los vecinos                                ===##
##============================================================================##

## nueve variables del entorno: estrato, edad, mujeres, afro, servicios y tamaño
m9 <- lm(voto_petro ~ educ_superior + estrato_bajo + estrato_alto + edad_20_29 + edad_60_74 +
                      mujeres + afro + servicios + log(registrados), data = train)
round(coef(summary(m9)), 3)
summary(m9)$r.squared

## k vecinos mas cercanos en el mapa: para cada puesto nuevo, promediar el voto de
## los k puestos de entrenamiento mas cercanos (se usa varias veces abajo)
knn_reg <- function(coord_train, y_train, coord_nueva, k) {
           prediccion <- rep(NA, nrow(coord_nueva))
           for (i in 1:nrow(coord_nueva)) {
             distancia <- sqrt((coord_train[, 1] - coord_nueva[i, 1])^2 + (coord_train[, 2] - coord_nueva[i, 2])^2)
             vecinos <- order(distancia)[1:k]
             prediccion[i] <- mean(y_train[vecinos])
           }
           return(prediccion)
}
coord_train <- as.matrix(train[, c("lon", "lat")])
coord_test <- as.matrix(test[, c("lon", "lat")])

##============================================================================##
##=== 5. La tabla canónica: siete modelos, una misma prueba                 ===##
##============================================================================##

## dos polinomios en la educacion superior, como en la semana 1
m3 <- lm(voto_petro ~ poly(educ_superior, 3), data = train)
m12 <- lm(voto_petro ~ poly(educ_superior, 12), data = train)

## predicciones de cada modelo sobre los 86 puestos de prueba (una sola vez)
pred_test <- list(rep(media_train, nrow(test)),
                  predict(m1, test),
                  predict(m3, test),
                  predict(m12, test),
                  predict(m9, test),
                  knn_reg(coord_train, train$voto_petro, coord_test, k = 1),
                  knn_reg(coord_train, train$voto_petro, coord_test, k = 5),
                  knn_reg(coord_train, train$voto_petro, coord_test, k = 25))
pred_train <- list(rep(media_train, nrow(train)),
                   predict(m1, train),
                   predict(m3, train),
                   predict(m12, train),
                   predict(m9, train),
                   knn_reg(coord_train, train$voto_petro, coord_train, k = 1),
                   knn_reg(coord_train, train$voto_petro, coord_train, k = 5),
                   knn_reg(coord_train, train$voto_petro, coord_train, k = 25))

## la tabla: una fila por modelo, la misma prueba para todos; k = 1 acierta todo
## dentro y es de los peores fuera
tabla <- data.frame(modelo = c("M0: media del entrenamiento", "M1: recta (educación superior)", "M3: cúbica", "M12: grado 12",
                               "M9: nueve variables",
                               "k-NN en el mapa, k = 1", "k-NN en el mapa, k = 5", "k-NN en el mapa, k = 25"),
                    rmse_dentro = sapply(pred_train, function(p) rmse(train$voto_petro, p)),
                    rmse_prueba = sapply(pred_test, function(p) rmse(test$voto_petro, p)),
                    mae_prueba = sapply(pred_test, function(p) mae(test$voto_petro, p)),
                    r2_prueba = sapply(pred_test, function(p) r2_prueba(test$voto_petro, p, train$voto_petro)))
tabla

## predicho vs. observado en la prueba: la recta (azul) y las nueve variables (verde)
plot(test$voto_petro, pred_test[[2]], pch = 16, col = "blue", xlim = c(0, 100), ylim = c(0, 100),
     xlab = "Voto por Petro observado (%)", ylab = "Voto por Petro predicho (%)",
     main = "Predicho vs. observado en los 86 puestos de prueba")
points(test$voto_petro, pred_test[[5]], pch = 17, col = "darkgreen")
abline(a = 0, b = 1, lty = 2)

## el error por quintil del voto observado: la recta falla mas en los extremos
test$quintil <- ntile(test$voto_petro, 5)
test$error_recta <- test$voto_petro - pred_test[[2]]
test %>%
  group_by(quintil) %>%
  summarise(puestos = n(), voto_medio = mean(voto_petro), rmse_recta = sqrt(mean(error_recta^2)), .groups = "drop")

## los residuos de la recta en el mapa: manchas de un mismo signo delatan lo que
## la educacion no captura (la ubicacion)
train$residuo <- train$voto_petro - predict(m1, train)
plot(train$lon, train$lat, pch = 16, col = ifelse(train$residuo > 0, "firebrick", "steelblue"),
     cex = 0.5 + abs(train$residuo) / 10, xlab = "Longitud", ylab = "Latitud",
     main = "Residuos de la recta: Petro votó más (rojo) o menos (azul) de lo que dice la educación")

##============================================================================##
##=== 6. La regla vigente: el voto de 2018 en el mismo puesto               ===##
##============================================================================##

## una campaña ya tiene una regla para predecir: lo que paso en el puesto la vez
## anterior; en 319 de los 343 puestos existe el voto de 2018 (Petro vs. Duque)
con_2018 <- metro %>% filter(!is.na(voto_petro_2018))
nrow(con_2018)
c(voto_2018 = mean(con_2018$voto_petro_2018), voto_2022 = mean(con_2018$voto_petro))

## tal cual (Petro crecio 11 puntos: la regla se queda corta en todos) y con una recta
## sobre 2018; contra la recta con educacion en los mismos puestos
m_2018 <- lm(voto_petro ~ voto_petro_2018, data = con_2018)
m_educ <- lm(voto_petro ~ educ_superior, data = con_2018)
data.frame(regla = c("el voto de 2018 tal cual", "recta sobre el voto de 2018", "recta sobre educación superior"),
           rmse_dentro = c(rmse(con_2018$voto_petro, con_2018$voto_petro_2018),
                           rmse(con_2018$voto_petro, predict(m_2018, con_2018)),
                           rmse(con_2018$voto_petro, predict(m_educ, con_2018))))

##============================================================================##
##=== 7. Extrapolar: un puesto que no se parece a ninguno                   ===##
##============================================================================##

## un entorno con 80% de educacion superior, fuera del rango 0-58 del entrenamiento:
## la recta sigue derecho, la cubica y el grado 12 se disparan
raro <- data.frame(educ_superior = 80)
extrapolacion <- data.frame(modelo = c("recta", "cúbica", "grado 12"),
                            prediccion = c(predict(m1, raro), predict(m3, raro), predict(m12, raro)))
extrapolacion

##============================================================================##
##=== 8. La limitación: la recta no viaja                                   ===##
##============================================================================##

## la recta del area metropolitana aplicada al resto del Valle: en los puestos
## rurales y en Buenaventura predice mal, y no por el ruido
resto <- valle %>%
         filter(!municipio %in% c("CALI", "PALMIRA", "YUMBO", "JAMUNDI")) %>%
         mutate(grupo = case_when(municipio == "BUENAVENTURA" ~ "Buenaventura",
                                  zona == "rural" ~ "resto del Valle, rural",
                                  TRUE ~ "resto del Valle, cabeceras"))
resto$pred_recta <- predict(m1, resto)
resto$pred_m9 <- predict(m9, resto)
resto %>%
  group_by(grupo) %>%
  summarise(puestos = n(), voto_real = mean(voto_petro), voto_predicho = mean(pred_recta),
            rmse_recta = rmse(voto_petro, pred_recta), rmse_nueve = rmse(voto_petro, pred_m9),
            sd_local = sd(voto_petro), .groups = "drop")

## y en el pais: la misma recta, estimada con los 12.001 puestos, no explica nada
colombia <- import("input/puestos_colombia_2022.rds", trust = T)
m_pais <- lm(voto_petro ~ educ_superior, data = colombia)
summary(m_pais)$r.squared
colombia %>%
  group_by(departamento) %>%
  summarise(puestos = n(), voto_petro = mean(voto_petro), educ_superior = mean(educ_superior, na.rm = T)) %>%
  arrange(voto_petro) %>%
  print(n = 33)
