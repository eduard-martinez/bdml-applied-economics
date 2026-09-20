##============================================================================##
# week-06.R  —  Abrir la caja negra
#------------------------------------------------------------------------------#
# La pregunta de hoy: el líder de la semana 5 predice el voto con un error de
# 12 puntos y no tiene coeficientes. ¿Qué aprendió del territorio? ¿Qué variables
# usa, con qué forma, y dónde se equivoca?
# Qué lee (carpeta input/, grano: puesto de votación):
#   - ficha_puestos_2022.rds  los puestos de 2022 con la ficha de 37 predictores,
#                             voto_petro, gana_petro y la partición fija (muestra)
# Qué produce: gráficos en pantalla y tablas en consola; no exporta nada.
# Fijar el directorio de trabajo en applications/week-06. Tarda unos 8 minutos:
# la ablación por bloques reentrena el modelo nueve veces.
##============================================================================##

## configuracion inicial
rm(list = ls())
if (!require(pacman)) install.packages("pacman")
pacman::p_load(rio, dplyr, randomForest, xgboost)

## la metrica del curso
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

##============================================================================##
##=== 1. El modelo de la sesion: el lider de la semana 5                   ===##
##============================================================================##

## los puestos con su ficha; la misma particion de siempre
puestos <- import("input/ficha_puestos_2022.rds", trust = T)
x_ficha <- setdiff(names(puestos), c("puesto", "nombre", "departamento", "municipio", "cod_municipio", "zona", "lon", "lat",
                                     "registrados", "voto_petro", "gana_petro", "participacion", "voto_petro_2018", "muestra"))
train <- puestos %>% filter(muestra == "entrenamiento")
test <- puestos %>% filter(muestra == "prueba")
x_train <- train[, c(x_ficha, "lon", "lat")] %>% mutate(departamento = factor(train$departamento))
x_test <- test[, c(x_ficha, "lon", "lat")] %>% mutate(departamento = factor(test$departamento, levels = levels(x_train$departamento)))

## el boosting de la semana 5, reentrenado igual (rondas por CV con parada temprana);
## para los valores SHAP usamos este y no el bosque: en xgboost el calculo es exacto y rapido
m_train <- model.matrix(~ . - 1, data = x_train)
m_test <- model.matrix(~ . - 1, data = x_test)
d_train <- xgb.DMatrix(m_train, label = train$voto_petro)
parametros <- list(eta = 0.05, max_depth = 5, subsample = 0.8, colsample_bytree = 0.8)
set.seed(2026)
cv_boost <- xgb.cv(params = parametros, data = d_train, nrounds = 2000, nfold = 5, early_stopping_rounds = 50, verbose = 0)
boost <- xgb.train(params = parametros, data = d_train, nrounds = cv_boost$early_stop$best_iteration)
pred <- predict(boost, xgb.DMatrix(m_test))
rmse(test$voto_petro, pred)

## y el bosque de la semana 5, que trae las dos importancias nativas (tarda ~2 min)
set.seed(2026)
bosque <- randomForest(x = x_train, y = train$voto_petro, ntree = 400, importance = T)

##============================================================================##
##=== 2. Importancia: impureza, permutacion, y la trampa                   ===##
##============================================================================##

## las dos importancias del bosque, lado a lado: el ranking no es el mismo
tabla_imp <- data.frame(permutacion = importance(bosque)[, "%IncMSE"], impureza = importance(bosque)[, "IncNodePurity"])
tabla_imp %>% arrange(desc(permutacion)) %>% head(10)
tabla_imp %>% arrange(desc(impureza)) %>% head(10)

## permutacion a mano sobre el boosting, en la prueba: barajar la variable y medir
## cuanto empeora (la funcion sirve para una variable o para un bloque entero)
set.seed(2026)
permutar <- function(columnas) {
            x_perm <- m_test
            for (v in columnas) x_perm[, v] <- x_perm[sample(nrow(x_perm)), v]
            rmse(test$voto_petro, predict(boost, xgb.DMatrix(x_perm)))
}
dpto_cols <- grep("^departamento", colnames(m_test), value = T)
round(c(sin_permutar = rmse(test$voto_petro, pred),
        lon = permutar("lon"), lat = permutar("lat"), lon_y_lat = permutar(c("lon", "lat")),
        departamento = permutar(dpto_cols), toda_la_geografia = permutar(c("lon", "lat", dpto_cols)),
        educ_superior = permutar("educ_superior"), indigena = permutar("indigena"), afro = permutar("afro")), 2)

##============================================================================##
##=== 3. Ablacion por bloques: reentrenar sin cada bloque                  ===##
##============================================================================##

## la permutacion mantiene el modelo fijo; la ablacion pregunta otra cosa: ¿que error
## tendria un modelo que nunca vio ese bloque? (reentrena, con sus rondas por CV)
ajustar_sin <- function(columnas) {
               x_sin <- x_train[, setdiff(names(x_train), columnas), drop = F]
               x_sin_test <- x_test[, setdiff(names(x_test), columnas), drop = F]
               m_sin <- model.matrix(~ . - 1, data = x_sin)
               m_sin_test <- model.matrix(~ . - 1, data = x_sin_test)
               d_sin <- xgb.DMatrix(m_sin, label = train$voto_petro)
               set.seed(2026)
               cv <- xgb.cv(params = parametros, data = d_sin, nrounds = 2000, nfold = 5, early_stopping_rounds = 50, verbose = 0)
               modelo <- xgb.train(params = parametros, data = d_sin, nrounds = cv$early_stop$best_iteration)
               rmse(test$voto_petro, predict(modelo, xgb.DMatrix(m_sin_test)))
}
bloques <- list(geografia = c("lon", "lat", "departamento"),
                educacion = c("educ_ninguna", "educ_tecnica", "educ_superior"),
                edad_y_genero = c("edad_0_19", "edad_20_29", "edad_45_59", "edad_60_74", "edad_75", "edad_votar", "mujeres"),
                etnia = c("indigena", "rom", "raizal", "palenquero", "afro"),
                servicios = c("serv_agua", "serv_alcantarillado", "serv_energia", "serv_gas", "serv_internet"),
                estrato = paste0("estrato_", 0:6),
                escala = c("hogares_por_vivienda", "personas_por_hogar", "log_poblacion", "log_viviendas", "log_manzanas",
                           "manzanas_compartidas", "log_distancia", "desfase", "log_registrados", "rural"))
ablacion <- sapply(bloques, ajustar_sin)
round(sort(ablacion, decreasing = T), 2)

## los dos extremos: solo la geografia, y solo la etnia
c(solo_geografia = ajustar_sin(setdiff(names(x_train), c("lon", "lat", "departamento"))),
  solo_etnia = ajustar_sin(setdiff(names(x_train), bloques$etnia)))

##============================================================================##
##=== 4. Dependencia parcial e ICE: la forma de cada variable              ===##
##============================================================================##

## PDP a mano: fijar educ_superior en un valor para todos los puestos (una muestra
## de 800 para no esperar), predecir y promediar; repetir sobre una malla
set.seed(2026)
muestra_pdp <- m_train[sample(nrow(m_train), 800), ]
pdp <- function(variable, malla) {
       sapply(malla, function(g) {
              x_pdp <- muestra_pdp
              x_pdp[, variable] <- g
              mean(predict(boost, xgb.DMatrix(x_pdp)))
       })
}
malla_educ <- quantile(train$educ_superior, seq(0.02, 0.98, length.out = 12))
malla_afro <- quantile(train$afro, seq(0.02, 0.98, length.out = 12))
malla_edad <- quantile(train$edad_60_74, seq(0.02, 0.98, length.out = 12))
pdp_educ <- pdp("educ_superior", malla_educ)
pdp_afro <- pdp("afro", malla_afro)
pdp_edad <- pdp("edad_60_74", malla_edad)
round(rbind(educ_superior = malla_educ, pdp = pdp_educ), 1)
round(rbind(afro = malla_afro, pdp = pdp_afro), 1)
round(rbind(edad_60_74 = malla_edad, pdp = pdp_edad), 1)

## la sorpresa: la estrella de Cali, plana en el pais (su informacion ya esta en la
## geografia); la edad si conserva su pendiente
plot(malla_educ, pdp_educ, type = "b", pch = 16, ylim = c(48, 58), xlab = "educacion superior (%)", ylab = "prediccion promedio",
     main = "PDP: educacion superior (negro) y edad 60-74 (rojo)")
lines(malla_edad, pdp_edad, type = "b", pch = 16, col = "red")

## ICE: la misma curva, puesto por puesto; el promedio (PDP) esconde pendientes opuestas
malla_ice <- c(0, 5, 10, 20, 30, 40, 50, 60)
ice <- function(fila, variable, malla) {
       sapply(malla, function(g) {
              x_ice <- m_test[fila, , drop = F]
              x_ice[, variable] <- g
              predict(boost, xgb.DMatrix(x_ice))
       })
}
tres <- c(which(test$municipio == "CALI" & test$educ_superior > 40)[1],
          which(test$departamento == "CHOCO")[1],
          which(test$municipio == "BOGOTA DC")[1])
test[tres, c("nombre", "municipio", "voto_petro")]
curvas <- t(sapply(tres, ice, variable = "educ_superior", malla = malla_ice))
round(curvas, 1)
matplot(malla_ice, t(curvas), type = "b", pch = 16, lty = 1, xlab = "educacion superior (%)", ylab = "prediccion",
        main = "ICE: tres puestos, tres pendientes")

##============================================================================##
##=== 5. SHAP: repartir cada prediccion entre las variables                ===##
##============================================================================##

## para arboles el valor de Shapley se calcula exacto (TreeSHAP): una fila por puesto
## de la prueba, una columna por variable, mas la base (la prediccion media)
shap <- predict(boost, xgb.DMatrix(m_test), predcontrib = T)
colnames(shap) <- c(colnames(m_test), "base")
dim(shap)

## la propiedad que lo define: base + suma de contribuciones = prediccion, puesto a puesto
c(base = shap[1, "base"], suma = sum(shap[1, ]), prediccion = pred[1])

## importancia global: el promedio de |contribucion| (las 33 dummies del departamento se suman)
importancia_shap <- colMeans(abs(shap))
importancia_shap <- c(importancia_shap[setdiff(colnames(shap), c(dpto_cols, "base"))], departamento = sum(importancia_shap[dpto_cols]))
round(sort(importancia_shap, decreasing = T)[1:10], 2)

## un puesto, explicado: Capitan (Choco); el mapa empuja 26 puntos hacia arriba
capitan <- which(test$nombre == "CAPITAN")[1]
test[capitan, c("nombre", "departamento", "voto_petro")]
contribuciones <- shap[capitan, setdiff(colnames(shap), "base")]
c(base = shap[capitan, "base"], prediccion = pred[capitan], real = test$voto_petro[capitan])
round(sort(contribuciones[abs(contribuciones) > 1], decreasing = T), 1)

## dependencia SHAP: la contribucion de afro contra su valor (el "PDP local" de SHAP)
test %>%
mutate(shap_afro = shap[, "afro"], quintil = ntile(afro, 5)) %>%
group_by(quintil) %>%
summarise(afro = mean(afro), contribucion = mean(shap_afro), .groups = "drop")

##============================================================================##
##=== 6. Donde se equivoca el lider                                        ===##
##============================================================================##

## por zona: el error del lider vive en lo rural
test %>%
mutate(error = voto_petro - pred) %>%
group_by(zona) %>%
summarise(puestos = n(), rmse = sqrt(mean(error^2)), sesgo = mean(error), .groups = "drop")

## compresion hacia la media: en los extremos el modelo se queda corto
test %>%
mutate(tramo = cut(voto_petro, c(0, 20, 40, 60, 80, 100), include.lowest = T)) %>%
group_by(tramo) %>%
summarise(puestos = n(), real = mean(voto_petro), predicho = mean(pred[cur_group_rows()]), .groups = "drop")
plot(pred, test$voto_petro, pch = 16, cex = 0.4, col = "gray50", xlab = "predicho", ylab = "observado",
     main = "El lider, predicho vs. observado (la linea es la diagonal)")
abline(0, 1)

## por departamento: donde el modelo sabe y donde no
errores_dpto <- test %>%
                mutate(error = voto_petro - pred) %>%
                group_by(departamento) %>%
                summarise(puestos = n(), rmse = sqrt(mean(error^2)), .groups = "drop")
errores_dpto %>% arrange(desc(rmse)) %>% head(4)
errores_dpto %>% arrange(rmse) %>% head(4)

##============================================================================##
##=== 7. La tabla del lider no se movio                                     ===##
##============================================================================##

## hoy no entro ningun modelo nuevo: aprendimos que hay dentro del lider.
## el RMSE de prueba sigue en 12,1 (bosque + coordenadas) y 12,2 (boosting)
c(boosting = rmse(test$voto_petro, pred), bosque = rmse(test$voto_petro, predict(bosque, x_test)))
