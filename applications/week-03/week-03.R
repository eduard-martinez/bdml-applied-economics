##============================================================================##
# week-03.R  —  Validación cruzada y regularización: el pipeline honesto
#------------------------------------------------------------------------------#
# La pregunta de hoy: el censo describe el entorno de cada puesto con 37
# predictores (y más de mil si pesan distinto en cada departamento). ¿Cuáles
# usar, y cómo saber, sin gastar la prueba, que una decisión mejora la predicción?
# Qué lee (carpeta input/, grano: puesto de votación):
#   - puestos_colombia_2022.rds  los 12.001 puestos, con la partición fija (muestra)
#   - censo_puestos_2022.rds     las columnas crudas del censo 2018 por puesto
#   - puestos_colombia_2018.rds  la segunda vuelta de 2018 (Petro vs. Duque)
# Qué produce: gráficos en pantalla y tablas en consola; no exporta nada.
# Fijar el directorio de trabajo en applications/week-03.
##============================================================================##

## configuracion inicial
rm(list = ls())
if (!require(pacman)) install.packages("pacman")
pacman::p_load(rio, dplyr, glmnet)

## la metrica del curso
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

##============================================================================##
##=== 1. La ficha completa: de conteos a 37 predictores                    ===##
##============================================================================##

## los puestos y el censo crudo de su entorno
puestos <- import("input/puestos_colombia_2022.rds", trust = T)
censo <- import("input/censo_puestos_2022.rds", trust = T)

## ingenieria de variables: los conteos se vuelven proporciones y logaritmos
## (una categoria por bloque queda por fuera: es la referencia)
ficha <- censo %>%
         mutate(con_nivel = educ_ninguna + educ_basica + educ_tecnica + educ_superior,
                con_etnia = etnia_ninguna + indigena + rom + raizal + palenquero + afro) %>%
         transmute(puesto,
                   educ_ninguna = educ_ninguna / con_nivel * 100, educ_tecnica = educ_tecnica / con_nivel * 100,
                   educ_superior = educ_superior / con_nivel * 100,
                   edad_0_19 = edad_0_19 / personas * 100, edad_20_29 = edad_20_29 / personas * 100,
                   edad_45_59 = edad_45_59 / personas * 100, edad_60_74 = edad_60_74 / personas * 100,
                   edad_75 = edad_75 / personas * 100,
                   mujeres = mujeres / personas * 100, edad_votar = edad_votar / personas * 100,
                   indigena = indigena / con_etnia * 100, rom = rom / con_etnia * 100, raizal = raizal / con_etnia * 100,
                   palenquero = palenquero / con_etnia * 100, afro = afro / con_etnia * 100,
                   serv_agua = serv_agua / viviendas * 100, serv_alcantarillado = serv_alcantarillado / viviendas * 100,
                   serv_energia = serv_energia / viviendas * 100, serv_gas = serv_gas / viviendas * 100,
                   serv_internet = serv_internet / viviendas * 100,
                   estrato_0 = estrato_0 / viviendas * 100, estrato_1 = estrato_1 / viviendas * 100,
                   estrato_2 = estrato_2 / viviendas * 100, estrato_3 = estrato_3 / viviendas * 100,
                   estrato_4 = estrato_4 / viviendas * 100, estrato_5 = estrato_5 / viviendas * 100,
                   estrato_6 = estrato_6 / viviendas * 100,
                   hogares_por_vivienda = hogares / viviendas, personas_por_hogar = personas / hogares,
                   log_poblacion = log(personas), log_viviendas = log(viviendas), log_manzanas = log(manzanas),
                   manzanas_compartidas = manzanas_compartidas / manzanas * 100,
                   log_distancia = log(1 + distancia_media),
                   desfase = pmin(pmax(desfase, -100), 300))

## la base de trabajo: identificacion, y, las variables de la semana 2 y la ficha
base <- puestos %>%
        select(puesto, nombre, departamento, municipio, cod_municipio, zona, lon, lat, registrados,
               voto_petro, estrato_bajo, estrato_alto, muestra) %>%
        inner_join(ficha, by = "puesto") %>%
        mutate(log_registrados = log(registrados), rural = as.integer(zona == "rural")) %>%
        filter(!is.na(educ_superior), !is.na(afro))
x_ficha <- c(setdiff(names(ficha), "puesto"), "log_registrados", "rural")
length(x_ficha)

## las formulas de hoy
f_recta <- voto_petro ~ educ_superior
f_nueve <- voto_petro ~ educ_superior + estrato_bajo + estrato_alto + edad_20_29 + edad_60_74 + mujeres + afro + serv_gas + log_registrados
f_ficha <- as.formula(paste("voto_petro ~", paste(x_ficha, collapse = " + ")))

##============================================================================##
##=== 2. Primer acto: 37 predictores para 257 puestos                      ===##
##============================================================================##

## el area metropolitana de la semana 2, con la misma particion
metro <- base %>% filter(departamento == "VALLE", municipio %in% c("CALI", "PALMIRA", "YUMBO", "JAMUNDI"))
set.seed(2026)
prueba_id <- sample(nrow(metro), size = round(0.25 * nrow(metro)))
train <- metro[-prueba_id, ]
test <- metro[prueba_id, ]

## la recta, las nueve variables y la ficha completa: dentro del entrenamiento todo mejora
m_recta <- lm(f_recta, data = train)
m_nueve <- lm(f_nueve, data = train)
m_ficha <- lm(f_ficha, data = train)
dentro <- c(recta = rmse(train$voto_petro, predict(m_recta, train)),
            nueve = rmse(train$voto_petro, predict(m_nueve, train)),
            ficha = rmse(train$voto_petro, predict(m_ficha, train)))
dentro
summary(m_ficha)$r.squared
sum(summary(m_ficha)$coefficients[-1, 4] < 0.05)

## el conjunto de validacion: apartar 64 puestos del entrenamiento y elegir el mejor;
## con cinco semillas distintas el ganador y su error cambian
for (semilla in 1:5) {
  set.seed(semilla)
  val_id <- sample(nrow(train), size = 64)
  ajuste <- train[-val_id, ]
  valida <- train[val_id, ]
  errores <- c(recta = rmse(valida$voto_petro, predict(lm(f_recta, ajuste), valida)),
               nueve = rmse(valida$voto_petro, predict(lm(f_nueve, ajuste), valida)),
               ficha = rmse(valida$voto_petro, predict(lm(f_ficha, ajuste), valida)))
  cat("semilla", semilla, ":", round(errores, 2), " gana", names(which.min(errores)), "\n")
}

## validacion cruzada de 5 pliegues, a mano: cada puesto se predice una vez con un
## modelo que no lo vio
set.seed(2026)
pliegue <- sample(rep(1:5, length.out = nrow(train)))
cv_lm <- function(formula, datos, pliegue) {
         error2 <- c()
         for (k in sort(unique(pliegue))) {
           modelo <- lm(formula, data = datos[pliegue != k, ])
           error2 <- c(error2, (datos$voto_petro[pliegue == k] - predict(modelo, datos[pliegue == k, ]))^2)
         }
         return(sqrt(mean(error2)))
}
cv <- c(recta = cv_lm(f_recta, train, pliegue), nueve = cv_lm(f_nueve, train, pliegue), ficha = cv_lm(f_ficha, train, pliegue))

## leave-one-out: 257 ajustes, o uno solo con el atajo de la matriz sombrero (apendice A1)
loocv_lm <- function(modelo) sqrt(mean((residuals(modelo) / (1 - hatvalues(modelo)))^2))
loocv <- c(recta = loocv_lm(m_recta), nueve = loocv_lm(m_nueve), ficha = loocv_lm(m_ficha))
rbind(dentro, cv, loocv)

## k-NN en el mapa: la k se elige por CV, no a ojo
knn_reg <- function(coord_train, y_train, coord_nueva, k) {
           prediccion <- rep(NA, nrow(coord_nueva))
           for (i in 1:nrow(coord_nueva)) {
             distancia <- (coord_train[, 1] - coord_nueva[i, 1])^2 + (coord_train[, 2] - coord_nueva[i, 2])^2
             prediccion[i] <- mean(y_train[order(distancia)[1:k]])
           }
           return(prediccion)
}
cv_knn <- function(datos, pliegue, k) {
          error2 <- c()
          for (j in sort(unique(pliegue))) {
            a <- datos[pliegue != j, ]
            b <- datos[pliegue == j, ]
            error2 <- c(error2, (b$voto_petro - knn_reg(as.matrix(a[, c("lon", "lat")]), a$voto_petro, as.matrix(b[, c("lon", "lat")]), k))^2)
          }
          return(sqrt(mean(error2)))
}
ks <- c(1, 2, 3, 5, 7, 10, 15, 20, 30, 50)
cv_k <- sapply(ks, function(k) cv_knn(train, pliegue, k))

## la misma curva con su error estandar (los cinco pliegues dan cinco errores), el
## error dentro y, solo para la foto, el de la prueba
ee_k <- sapply(ks, function(k) {
        mse_pliegue <- sapply(1:5, function(j) {
                       a <- train[pliegue != j, ]
                       b <- train[pliegue == j, ]
                       mean((b$voto_petro - knn_reg(as.matrix(a[, c("lon", "lat")]), a$voto_petro, as.matrix(b[, c("lon", "lat")]), k))^2)
        })
        sd(sqrt(mse_pliegue)) / sqrt(5)
})
coord_train <- as.matrix(train[, c("lon", "lat")])
dentro_k <- sapply(ks, function(k) rmse(train$voto_petro, knn_reg(coord_train, train$voto_petro, coord_train, k)))
prueba_k <- sapply(ks, function(k) rmse(test$voto_petro, knn_reg(coord_train, train$voto_petro, as.matrix(test[, c("lon", "lat")]), k)))
curva_k <- data.frame(k = ks, dentro = dentro_k, cv = cv_k, ee = ee_k, prueba = prueba_k)
round(curva_k, 2)

## el minimo y la regla de un error estandar (el modelo mas simple, el k mas grande,
## a menos de un error estandar del minimo)
k_min <- ks[which.min(cv_k)]
k_1se <- max(ks[cv_k <= min(cv_k) + ee_k[which.min(cv_k)]])
c(k_min = k_min, k_1se = k_1se)
plot(ks, cv_k, type = "b", pch = 16, col = "red", log = "x", ylim = c(0, 12), xlab = "k (vecinos)", ylab = "RMSE",
     main = "k-NN en el mapa: dentro (gris), CV (rojo) y prueba (azul)")
lines(ks, dentro_k, type = "b", pch = 16, col = "gray50")
lines(ks, prueba_k, type = "b", pch = 16, col = "blue", lty = 2)

## regularizacion: Lasso, Elastic Net y Ridge sobre la ficha, con lambda por CV
x_train <- as.matrix(train[, x_ficha])
x_test <- as.matrix(test[, x_ficha])
set.seed(2026)
cv_lasso <- cv.glmnet(x_train, train$voto_petro, alpha = 1, nfolds = 5)
set.seed(2026)
cv_en <- cv.glmnet(x_train, train$voto_petro, alpha = 0.5, nfolds = 5)
set.seed(2026)
cv_ridge <- cv.glmnet(x_train, train$voto_petro, alpha = 0, nfolds = 5)

## el camino de los coeficientes y la curva de CV del Lasso
plot(cv_lasso$glmnet.fit, xvar = "lambda", main = "Lasso: el camino de los coeficientes")
plot(cv_lasso)
c(lambda_min = cv_lasso$lambda.min, lambda_1se = cv_lasso$lambda.1se)

## el modelo parsimonioso: lo que queda con la regla de un error estandar
coef_1se <- coef(cv_lasso, s = "lambda.1se")
round(coef_1se[coef_1se[, 1] != 0, , drop = F], 3)

## el orden de entrada al camino: lo que mas aporta dado lo que ya esta
betas <- as.matrix(cv_lasso$glmnet.fit$beta)
entrada <- apply(betas != 0, 1, function(fila) which(fila)[1])
names(sort(entrada))[1:8]

## el proxy: educacion superior y estrato alto cuentan casi la misma historia
cor(train$educ_superior, train$estrato_5 + train$estrato_6)

## la tabla del primer acto: dentro, CV y prueba (la prueba, una sola vez)
penalizado <- function(cv_obj, s) {
              c(coeficientes = sum(coef(cv_obj, s = s) != 0) - 1,
                dentro = rmse(train$voto_petro, predict(cv_obj, x_train, s = s)),
                cv = sqrt(cv_obj$cvm[cv_obj$lambda == cv_obj[[s]]]),
                prueba = rmse(test$voto_petro, predict(cv_obj, x_test, s = s)))
}
acto_1 <- rbind(recta = c(1, dentro["recta"], cv["recta"], rmse(test$voto_petro, predict(m_recta, test))),
                nueve = c(9, dentro["nueve"], cv["nueve"], rmse(test$voto_petro, predict(m_nueve, test))),
                ficha_ols = c(37, dentro["ficha"], cv["ficha"], rmse(test$voto_petro, predict(m_ficha, test))),
                ridge_min = penalizado(cv_ridge, "lambda.min"),
                en_min = penalizado(cv_en, "lambda.min"),
                lasso_min = penalizado(cv_lasso, "lambda.min"),
                lasso_1se = penalizado(cv_lasso, "lambda.1se"))
colnames(acto_1) <- c("coeficientes", "dentro", "cv", "prueba")
round(acto_1, 2)

## ¿6,7 contra 6,8 es una diferencia? bootstrap sobre los 86 puestos de prueba:
## se remuestrea la prueba y se recalcula la metrica (y la diferencia entre dos modelos)
set.seed(2026)
pred_recta <- predict(m_recta, test)
pred_ficha <- predict(m_ficha, test)
replicas <- t(sapply(1:1000, function(b) {
            i <- sample(nrow(test), replace = T)
            c(recta = rmse(test$voto_petro[i], pred_recta[i]), ficha = rmse(test$voto_petro[i], pred_ficha[i]))
}))
round(rbind(recta = quantile(replicas[, "recta"], c(0.025, 0.975)),
            ficha = quantile(replicas[, "ficha"], c(0.025, 0.975)),
            diferencia = quantile(replicas[, "ficha"] - replicas[, "recta"], c(0.025, 0.975))), 2)

## ¿cuales elige el Lasso? la seleccion cambia con la muestra: 20 remuestras bootstrap
set.seed(2026)
elegidas <- matrix(0, nrow = 20, ncol = length(x_ficha), dimnames = list(NULL, x_ficha))
for (b in 1:20) {
  remuestra <- sample(nrow(train), replace = T)
  ajuste <- cv.glmnet(x_train[remuestra, ], train$voto_petro[remuestra], alpha = 1, nfolds = 5)
  elegidas[b, ] <- as.numeric(coef(ajuste, s = "lambda.1se")[-1, 1] != 0)
}
sort(colMeans(elegidas), decreasing = T)[1:12]

##============================================================================##
##=== 3. Segundo acto: el país, con la partición fija del curso             ===##
##============================================================================##

## 9.602 puestos para aprender, 2.398 bajo llave
train_pais <- base %>% filter(muestra == "entrenamiento")
test_pais <- base %>% filter(muestra == "prueba")

## de la recta a la ficha con departamentos
f_ficha_dpto <- update(f_ficha, . ~ . + departamento)
modelos <- list(recta = lm(f_recta, train_pais), nueve = lm(f_nueve, train_pais),
                ficha = lm(f_ficha, train_pais), ficha_dpto = lm(f_ficha_dpto, train_pais))
pais <- data.frame(modelo = c("media", names(modelos)),
                   coeficientes = c(0, sapply(modelos, function(m) length(coef(m)) - 1)),
                   dentro = c(rmse(train_pais$voto_petro, mean(train_pais$voto_petro)),
                              sapply(modelos, function(m) rmse(train_pais$voto_petro, predict(m, train_pais)))),
                   prueba = c(rmse(test_pais$voto_petro, mean(train_pais$voto_petro)),
                              sapply(modelos, function(m) rmse(test_pais$voto_petro, predict(m, test_pais)))))
pais

## la ficha larga: cada predictor con una pendiente distinta en cada departamento
f_larga <- as.formula(paste("~ (", paste(x_ficha, collapse = " + "), ") * departamento"))
x_larga <- model.matrix(f_larga, data = rbind(train_pais, test_pais))[, -1]
x_larga_train <- x_larga[1:nrow(train_pais), ]
x_larga_test <- x_larga[-(1:nrow(train_pais)), ]
ncol(x_larga_train)

## OLS con 1.253 columnas: dentro mejora, en la prueba explota
ols_larga <- lm.fit(cbind(1, x_larga_train), train_pais$voto_petro)
beta <- ols_larga$coefficients
beta[is.na(beta)] <- 0
c(dentro = rmse(train_pais$voto_petro, cbind(1, x_larga_train) %*% beta),
  prueba = rmse(test_pais$voto_petro, cbind(1, x_larga_test) %*% beta))

## las mismas 1.253 con penalizacion y lambda por CV (tarda cerca de un minuto cada uno)
set.seed(2026)
cv_lasso_larga <- cv.glmnet(x_larga_train, train_pais$voto_petro, alpha = 1, nfolds = 5)
set.seed(2026)
cv_ridge_larga <- cv.glmnet(x_larga_train, train_pais$voto_petro, alpha = 0, nfolds = 5)
larga <- rbind(lasso_min = c(sum(coef(cv_lasso_larga, s = "lambda.min") != 0) - 1,
                             sqrt(min(cv_lasso_larga$cvm)),
                             rmse(test_pais$voto_petro, predict(cv_lasso_larga, x_larga_test, s = "lambda.min"))),
               lasso_1se = c(sum(coef(cv_lasso_larga, s = "lambda.1se") != 0) - 1,
                             sqrt(cv_lasso_larga$cvm[cv_lasso_larga$lambda == cv_lasso_larga$lambda.1se]),
                             rmse(test_pais$voto_petro, predict(cv_lasso_larga, x_larga_test, s = "lambda.1se"))),
               ridge_min = c(ncol(x_larga_train), sqrt(min(cv_ridge_larga$cvm)),
                             rmse(test_pais$voto_petro, predict(cv_ridge_larga, x_larga_test, s = "lambda.min"))))
colnames(larga) <- c("coeficientes", "cv", "prueba")
round(larga, 2)

## con 2.398 puestos de prueba los intervalos son estrechos: Lasso contra ficha + departamentos
set.seed(2026)
pred_lasso <- as.numeric(predict(cv_lasso_larga, x_larga_test, s = "lambda.min"))
pred_dpto <- predict(modelos$ficha_dpto, test_pais)
replicas <- t(sapply(1:1000, function(b) {
            i <- sample(nrow(test_pais), replace = T)
            c(lasso = rmse(test_pais$voto_petro[i], pred_lasso[i]), ficha_dpto = rmse(test_pais$voto_petro[i], pred_dpto[i]))
}))
round(rbind(lasso = quantile(replicas[, "lasso"], c(0.025, 0.975)),
            ficha_dpto = quantile(replicas[, "ficha_dpto"], c(0.025, 0.975)),
            diferencia = quantile(replicas[, "lasso"] - replicas[, "ficha_dpto"], c(0.025, 0.975))), 2)

##============================================================================##
##=== 4. El pipeline honesto: una trampa y dos esquemas de validación       ===##
##============================================================================##

## los mismos pliegues para todo lo que sigue: al azar y por municipio
set.seed(2026)
pl_azar <- sample(rep(1:5, length.out = nrow(train_pais)))
municipios <- unique(train_pais$cod_municipio)
set.seed(2026)
grupo <- sample(rep(1:5, length.out = length(municipios)))
pl_municipio <- grupo[match(train_pais$cod_municipio, municipios)]

## la trampa: codificar el municipio con su voto promedio usando TODOS los puestos
## (incluido el que se va a predecir) y validar despues
train_pais$media_municipio <- ave(train_pais$voto_petro, train_pais$cod_municipio)
f_trampa <- update(f_ficha, . ~ . + media_municipio)
cv_trampa <- cv_lm(f_trampa, train_pais, pl_azar)

## lo honesto: la media del municipio se calcula dentro de cada pliegue, sin el pliegue
error2 <- c()
for (k in 1:5) {
  a <- train_pais[pl_azar != k, ]
  b <- train_pais[pl_azar == k, ]
  medias <- a %>% group_by(cod_municipio) %>% summarise(media_municipio = mean(voto_petro), .groups = "drop")
  a$media_municipio <- ave(a$voto_petro, a$cod_municipio)
  b <- b %>% select(-media_municipio) %>% left_join(medias, by = "cod_municipio")
  b$media_municipio[is.na(b$media_municipio)] <- mean(a$voto_petro)
  error2 <- c(error2, (b$voto_petro - predict(lm(f_trampa, a), b))^2)
}
cv_honesto <- sqrt(mean(error2))

## y en la prueba, con la media calculada solo con el entrenamiento
medias <- train_pais %>% group_by(cod_municipio) %>% summarise(media_municipio = mean(voto_petro), .groups = "drop")
test_trampa <- test_pais %>% left_join(medias, by = "cod_municipio")
test_trampa$media_municipio[is.na(test_trampa$media_municipio)] <- mean(train_pais$voto_petro)
c(cv_con_trampa = cv_trampa, cv_honesto = cv_honesto,
  prueba = rmse(test_trampa$voto_petro, predict(lm(f_trampa, train_pais), test_trampa)))

## vecinos que se copian: CV al azar contra CV por municipio, para los vecinos del
## mapa y para la ficha con departamentos
x_dpto <- cbind(1, as.matrix(train_pais[, x_ficha]), model.matrix(~ departamento, train_pais)[, -1])
cv_matriz <- function(x, y, pliegue) {
             error2 <- c()
             for (k in 1:5) {
               beta <- lm.fit(x[pliegue != k, ], y[pliegue != k])$coefficients
               beta[is.na(beta)] <- 0
               error2 <- c(error2, (y[pliegue == k] - x[pliegue == k, ] %*% beta)^2)
             }
             return(sqrt(mean(error2)))
}
esquemas <- rbind(knn_mapa = c(cv_knn(train_pais, pl_azar, 10), cv_knn(train_pais, pl_municipio, 10)),
                  ficha_dpto = c(cv_matriz(x_dpto, train_pais$voto_petro, pl_azar), cv_matriz(x_dpto, train_pais$voto_petro, pl_municipio)))
colnames(esquemas) <- c("cv_al_azar", "cv_por_municipio")
round(esquemas, 2)
rmse(test_pais$voto_petro, knn_reg(as.matrix(train_pais[, c("lon", "lat")]), train_pais$voto_petro,
                                   as.matrix(test_pais[, c("lon", "lat")]), 10))

## en el tiempo: el modelo de 2018 (Petro vs. Duque) prediciendo 2022
puestos_2018 <- import("input/puestos_colombia_2018.rds", trust = T) %>%
                mutate(log_registrados = log(registrados)) %>%
                filter(!is.na(educ_superior), !is.na(afro))
f_bloques <- voto_petro ~ educ_superior + estrato_bajo + estrato_alto + servicios + edad_0_19 + edad_20_29 +
                          edad_45_59 + edad_60_74 + edad_75 + mujeres + afro + indigena + log(poblacion) +
                          log_registrados + zona + departamento
bloques_2022 <- puestos %>% mutate(log_registrados = log(registrados)) %>% filter(!is.na(educ_superior), !is.na(afro))
m_2018 <- lm(f_bloques, data = puestos_2018)
m_2022 <- lm(f_bloques, data = bloques_2022 %>% filter(muestra == "entrenamiento"))
test_2022 <- bloques_2022 %>% filter(muestra == "prueba")
salto <- mean(bloques_2022$voto_petro[bloques_2022$muestra == "entrenamiento"]) - mean(puestos_2018$voto_petro)
c(modelo_2022 = rmse(test_2022$voto_petro, predict(m_2022, test_2022)),
  modelo_2018 = rmse(test_2022$voto_petro, predict(m_2018, test_2022)),
  modelo_2018_corrigiendo_el_nivel = rmse(test_2022$voto_petro, predict(m_2018, test_2022) + salto),
  salto_de_nivel = salto)

##============================================================================##
##=== 5. La tabla del líder: nacional, la misma prueba para todos           ===##
##============================================================================##

lider <- data.frame(semana = 3,
                    modelo = c("media", "recta (educación superior)", "las nueve variables de la semana 2",
                               "ficha completa (37)", "ficha + departamentos", "OLS con la ficha larga (1.253)",
                               "Ridge, ficha larga", "Lasso, ficha larga", "k-NN en el mapa (k = 10)"),
                    rmse_prueba = c(pais$prueba, rmse(test_pais$voto_petro, cbind(1, x_larga_test) %*% beta),
                                    larga["ridge_min", "prueba"], larga["lasso_min", "prueba"],
                                    rmse(test_pais$voto_petro, knn_reg(as.matrix(train_pais[, c("lon", "lat")]), train_pais$voto_petro,
                                                                       as.matrix(test_pais[, c("lon", "lat")]), 10))))
lider %>% arrange(rmse_prueba)
