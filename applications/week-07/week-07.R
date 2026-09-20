##============================================================================##
# week-07.R  —  Representar: PCA, k-medias y una red neuronal
#------------------------------------------------------------------------------#
# La pregunta de hoy: la semana pasada vimos que las 37 columnas del censo son
# redundantes entre sí. ¿Pueden los datos construir sus propios resúmenes?
# Sin y: el PCA comprime la ficha en pocos índices y k-medias agrupa los puestos
# en tipologías de territorio. Con y: una red neuronal aprende la representación
# y la predicción a la vez, y se mide contra el bosque.
# Qué lee (carpeta input/, grano: puesto de votación):
#   - ficha_puestos_2022.rds  los puestos de 2022 con la ficha de 37 predictores,
#                             voto_petro, gana_petro y la partición fija (muestra)
# Qué produce: gráficos en pantalla y tablas en consola; no exporta nada.
# Fijar el directorio de trabajo en applications/week-07. Tarda unos 5 minutos:
# las redes se entrenan tres veces por configuración.
##============================================================================##

## configuracion inicial
rm(list = ls())
if (!require(pacman)) install.packages("pacman")
pacman::p_load(rio, dplyr, nnet)

## la metrica del curso
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

##============================================================================##
##=== 1. Los datos y la ficha estandarizada                                ===##
##============================================================================##

## los puestos con su ficha; la misma particion de siempre
puestos <- import("input/ficha_puestos_2022.rds", trust = T)
x_ficha <- setdiff(names(puestos), c("puesto", "nombre", "departamento", "municipio", "cod_municipio", "zona", "lon", "lat",
                                     "registrados", "voto_petro", "gana_petro", "participacion", "voto_petro_2018", "muestra"))
train <- puestos %>% filter(muestra == "entrenamiento")
test <- puestos %>% filter(muestra == "prueba")

## el PCA vive en distancias: hay que estandarizar, y con las medias y desviaciones
## DEL ENTRENAMIENTO (regla de oro #3: tambien sin y hay pipeline)
medias <- colMeans(train[, x_ficha])
desviaciones <- apply(train[, x_ficha], 2, sd)
z_train <- scale(as.matrix(train[, x_ficha]), center = medias, scale = desviaciones)
z_test <- scale(as.matrix(test[, x_ficha]), center = medias, scale = desviaciones)

##============================================================================##
##=== 2. PCA: las direcciones que concentran la varianza                   ===##
##============================================================================##

## las 37 columnas giradas a 37 componentes sin correlacion, ordenadas por varianza
pca <- prcomp(z_train, center = F, scale. = F)
varianza <- pca$sdev^2 / sum(pca$sdev^2)
round(rbind(componente = 1:8, varianza = varianza[1:8] * 100, acumulada = cumsum(varianza)[1:8] * 100), 1)
plot(varianza[1:15] * 100, type = "b", pch = 16, xlab = "componente", ylab = "% de la varianza",
     main = "Scree: la primera componente concentra un tercio de la ficha")

## ¿cuantas para el 80 y el 90 por ciento?
c(para_80 = which(cumsum(varianza) >= 0.8)[1], para_90 = which(cumsum(varianza) >= 0.9)[1])

## que es cada componente: las cargas mas grandes en valor absoluto
cargas <- function(k, n = 8) {
          v <- pca$rotation[, k]
          round(v[order(-abs(v))][1:n], 2)
}
cargas(1)
cargas(2)
cargas(3)

## PC1 es el eje urbano-rural: servicios y escala de un lado, ruralidad del otro.
## el puntaje de cada puesto en PC1 es un indice de carencias construido sin ver el voto
train$indice <- as.numeric(z_train %*% pca$rotation[, 1])
c(cor_con_rural = cor(train$indice, train$rural), cor_con_educacion = cor(train$indice, train$educ_superior),
  cor_con_participacion = cor(train$indice, train$participacion), cor_con_voto = cor(train$indice, train$voto_petro))

## el indice ordena carencias... y el voto no es lineal en el: los dos extremos votan Petro
train %>%
mutate(decil = ntile(indice, 10)) %>%
group_by(decil) %>%
summarise(educ_superior = mean(educ_superior), alcantarillado = mean(serv_alcantarillado), voto = mean(voto_petro), .groups = "drop")
plot(tapply(train$voto_petro, ntile(train$indice, 10), mean), type = "b", pch = 16, xlab = "decil del indice (PC1)",
     ylab = "voto por Petro", main = "La U: los dos extremos del indice votan Petro")

## regresion sobre componentes (PCR): ¿cuantas componentes recuperan a la ficha?
puntajes_train <- z_train %*% pca$rotation
puntajes_test <- z_test %*% pca$rotation
for (k in c(1, 2, 5, 10, 20, 37)) {
  modelo <- lm(train$voto_petro ~ puntajes_train[, 1:k, drop = F])
  prediccion <- cbind(1, puntajes_test[, 1:k, drop = F]) %*% coef(modelo)
  cat("PCR con", k, "componentes: RMSE de prueba", round(rmse(test$voto_petro, prediccion), 2), "\n")
}

##============================================================================##
##=== 3. k-medias: tipologias de territorio                                ===##
##============================================================================##

## el codo: la suma de cuadrados dentro de los grupos, para k de 2 a 8
codo <- c()
for (k in 2:8) {
  set.seed(2026)
  km <- kmeans(z_train, centers = k, nstart = 10)
  codo <- c(codo, km$tot.withinss / km$totss)
}
round(rbind(k = 2:8, dentro_sobre_total = codo), 3)
plot(2:8, codo, type = "b", pch = 16, xlab = "k (grupos)", ylab = "SS dentro / SS total", main = "El codo")

## cuatro tipologias, legibles (nstart alto: k-medias depende del arranque)
set.seed(2026)
tipologia <- kmeans(z_train, centers = 4, nstart = 25)
train$grupo <- tipologia$cluster
train %>%
group_by(grupo) %>%
summarise(puestos = n(), rural = mean(rural), educ_superior = mean(educ_superior),
          estrato_4_a_6 = mean(estrato_4 + estrato_5 + estrato_6), afro = mean(afro), indigena = mean(indigena),
          alcantarillado = mean(serv_alcantarillado), voto = mean(voto_petro), .groups = "drop")

## el mapa de las tipologias: el agrupamiento no uso las coordenadas y aun asi dibuja el pais
plot(train$lon, train$lat, col = train$grupo, pch = 16, cex = 0.4, xlab = "longitud", ylab = "latitud",
     main = "Cuatro tipologias del censo, en el mapa (sin usar coordenadas)")

##============================================================================##
##=== 4. La red: aprender la representacion y la prediccion a la vez       ===##
##============================================================================##

## una capa oculta de 16 neuronas sobre la ficha estandarizada y el departamento;
## y se reescala a [0, 1] porque nnet trabaja mejor asi
x_red <- cbind(z_train, model.matrix(~ departamento - 1, data = train %>% mutate(departamento = factor(departamento))))
x_red_test <- cbind(z_test, model.matrix(~ departamento - 1,
                                         data = test %>% mutate(departamento = factor(departamento, levels = levels(factor(train$departamento))))))
y_red <- train$voto_petro / 100

## tres semillas: la red depende del arranque aleatorio (el descenso de gradiente
## no encuentra siempre el mismo minimo); decay es la penalizacion de la semana 3
for (semilla in 1:3) {
  set.seed(semilla)
  red <- nnet(x_red, y_red, size = 16, decay = 0.1, maxit = 500, linout = T, trace = F, MaxNWts = 5000)
  cat("semilla", semilla, ": RMSE de prueba", round(rmse(test$voto_petro, predict(red, x_red_test) * 100), 2), "\n")
}

## con las coordenadas (estandarizadas con el entrenamiento), como el bosque
lon_media <- mean(train$lon); lon_sd <- sd(train$lon)
lat_media <- mean(train$lat); lat_sd <- sd(train$lat)
x_red2 <- cbind(x_red, lon = (train$lon - lon_media) / lon_sd, lat = (train$lat - lat_media) / lat_sd)
x_red2_test <- cbind(x_red_test, lon = (test$lon - lon_media) / lon_sd, lat = (test$lat - lat_media) / lat_sd)
mejor_red <- NULL
mejor_rmse <- Inf
for (semilla in 1:3) {
  set.seed(semilla)
  red <- nnet(x_red2, y_red, size = 16, decay = 0.1, maxit = 500, linout = T, trace = F, MaxNWts = 5000)
  error <- rmse(test$voto_petro, predict(red, x_red2_test) * 100)
  cat("semilla", semilla, "(con coordenadas): RMSE de prueba", round(error, 2), "\n")
  if (error < mejor_rmse) { mejor_rmse <- error; mejor_red <- red }
}

##============================================================================##
##=== 5. La tabla del lider: la red no destrona al bosque                  ===##
##============================================================================##

## el RMSE de prueba de la mejor semilla, contra los lideres de las semanas 3 a 5
lider <- data.frame(semana = c(3, 3, 5, 5, 7, 7),
                    modelo = c("Lasso, ficha larga", "k-NN en el mapa (k = 10)", "bosque (+ coordenadas)",
                               "boosting (+ coordenadas)", "red (ficha + dpto)", "red (+ coordenadas, mejor semilla)"),
                    rmse_prueba = c(14.75, 12.13, 12.10, 12.24, 13.74, round(mejor_rmse, 2)))
lider %>% arrange(rmse_prueba)
