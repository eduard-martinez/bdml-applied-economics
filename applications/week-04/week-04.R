##============================================================================##
# week-04.R  —  Clasificación: de la logística a las decisiones
#------------------------------------------------------------------------------#
# La pregunta de hoy: ¿gana Petro el puesto? ¿Con qué probabilidad? ¿Y qué hace
# una campaña con esa probabilidad cuando equivocarse hacia un lado cuesta más
# que hacia el otro?
# Qué lee (carpeta input/, grano: puesto de votación):
#   - ficha_puestos_2022.rds     los puestos de 2022 con la ficha de 37 predictores de la
#                                semana 3, gana_petro y la partición fija (muestra)
#   - puestos_colombia_2010.rds  la segunda vuelta de 2010 (Santos vs. Mockus)
# Qué produce: gráficos en pantalla y tablas en consola; no exporta nada.
# Fijar el directorio de trabajo en applications/week-04.
##============================================================================##

## configuracion inicial
rm(list = ls())
if (!require(pacman)) install.packages("pacman")
pacman::p_load(rio, dplyr, glmnet)

## las metricas de hoy: la AUC (probabilidad de ordenar bien un par, apendice A6) y
## las que salen de la matriz de confusion con un umbral
auc <- function(y, p) {
       rango <- rank(p)
       (sum(rango[y == 1]) - sum(y == 1) * (sum(y == 1) + 1) / 2) / (sum(y == 1) * sum(y == 0))
}
metricas <- function(y, p, umbral = 0.5) {
            pred <- as.integer(p >= umbral)
            vp <- sum(pred == 1 & y == 1)
            fn <- sum(pred == 0 & y == 1)
            fp <- sum(pred == 1 & y == 0)
            vn <- sum(pred == 0 & y == 0)
            c(auc = auc(y, p), exactitud = (vp + vn) / length(y), sensibilidad = vp / (vp + fn),
              especificidad = vn / (vn + fp), precision = vp / (vp + fp), f1 = 2 * vp / (2 * vp + fp + fn),
              balanceada = (vp / (vp + fn) + vn / (vn + fp)) / 2)
}

##============================================================================##
##=== 1. La pregunta, la referencia y la regla vigente                     ===##
##============================================================================##

## los puestos de 2022 con su ficha; la misma particion de la semana 3
puestos <- import("input/ficha_puestos_2022.rds", trust = T)
x_ficha <- setdiff(names(puestos), c("puesto", "nombre", "departamento", "municipio", "cod_municipio", "zona", "lon", "lat",
                                     "registrados", "voto_petro", "gana_petro", "participacion", "voto_petro_2018", "muestra"))
length(x_ficha)
train <- puestos %>% filter(muestra == "entrenamiento")
test <- puestos %>% filter(muestra == "prueba")

## la y de hoy: 1 si Petro saco mas votos que Hernandez en el puesto
c(entrenamiento = mean(train$gana_petro), prueba = mean(test$gana_petro))

## la geografia del problema: hay departamentos donde gano (o perdio) casi todos los puestos
por_dpto <- train %>%
            group_by(departamento) %>%
            summarise(puestos = n(), gana_petro = mean(gana_petro), .groups = "drop") %>%
            arrange(gana_petro)
head(por_dpto, 5)
tail(por_dpto, 5)

## la referencia que no mira los datos: "Petro gana en todos"
mean(test$gana_petro == 1)

## la regla vigente, lo que haria la campaña sin modelo: "gana donde gano en 2018"
## (solo en los puestos que existian en 2018)
test_historia <- test %>% filter(!is.na(voto_petro_2018))
nrow(test_historia)
table(real = test_historia$gana_petro, regla = as.integer(test_historia$voto_petro_2018 >= 50))
metricas(test_historia$gana_petro, test_historia$voto_petro_2018 / 100)

## la regla ordena muy bien y decide mal: Petro crecio entre 2018 y 2022, y el corte de
## 50 puntos quedo viejo (con 40 a 50 puntos en 2018 gano dos de cada tres puestos en 2022)
test_historia %>%
mutate(voto_2018 = cut(voto_petro_2018, c(0, 20, 30, 40, 50, 60, 70, 100), include.lowest = T)) %>%
group_by(voto_2018) %>%
summarise(puestos = n(), gana_2022 = mean(gana_petro), .groups = "drop")

##============================================================================##
##=== 2. El atajo: la regresion de la semana 3 como clasificador           ===##
##============================================================================##

## la ficha con departamentos sobre voto_petro (RMSE 15,6 la semana pasada): gana si predice mas de 50
f_voto <- as.formula(paste("voto_petro ~", paste(x_ficha, collapse = " + "), "+ departamento"))
m_voto <- lm(f_voto, data = train)
voto_pred <- predict(m_voto, test)
table(real = test$gana_petro, pred = as.integer(voto_pred >= 50))
c(exactitud = mean((voto_pred >= 50) == test$gana_petro), auc = auc(test$gana_petro, voto_pred))

## clasifica bien, pero entrega puntos de voto, no una probabilidad. y el camino corto,
## OLS sobre la dummy (el modelo de probabilidad lineal), entrega "probabilidades" imposibles
f_gana <- update(f_voto, gana_petro ~ .)
m_lpm <- lm(f_gana, data = train)
p_lpm <- predict(m_lpm, test)
c(menores_que_0 = sum(p_lpm < 0), mayores_que_1 = sum(p_lpm > 1), minimo = min(p_lpm), maximo = max(p_lpm))

##============================================================================##
##=== 3. La curva en S: el area metropolitana de la semana 2               ===##
##============================================================================##

## Cali, Palmira, Yumbo y Jamundi, con la particion de las semanas 2 y 3
metro <- puestos %>% filter(departamento == "VALLE", municipio %in% c("CALI", "PALMIRA", "YUMBO", "JAMUNDI"))
set.seed(2026)
prueba_id <- sample(nrow(metro), size = round(0.25 * nrow(metro)))
metro_train <- metro[-prueba_id, ]
metro_test <- metro[prueba_id, ]
c(puestos = nrow(metro_train), gana_petro = mean(metro_train$gana_petro), pierde = sum(metro_train$gana_petro == 0))

## la fraccion de puestos ganados por tramos de educacion superior: la S esta en los datos
tramos <- metro_train %>%
          mutate(tramo = cut(educ_superior, c(0, 10, 20, 30, 40, 50, 100), include.lowest = T)) %>%
          group_by(tramo) %>%
          summarise(puestos = n(), educ_superior = mean(educ_superior), gana_petro = mean(gana_petro), .groups = "drop")
tramos

## el modelo de probabilidad lineal y la logistica, con un solo predictor
lpm_metro <- lm(gana_petro ~ educ_superior, data = metro_train)
logit_metro <- glm(gana_petro ~ educ_superior, family = binomial, data = metro_train)
coef(lpm_metro)
summary(logit_metro)$coefficients
sum(fitted(lpm_metro) > 1)

## como se lee: el odds ratio, el punto donde la probabilidad es 0,5 y el efecto
## marginal, que depende de donde este parado el puesto
exp(coef(logit_metro)["educ_superior"])
-coef(logit_metro)[1] / coef(logit_metro)[2]
p_tres <- predict(logit_metro, data.frame(educ_superior = c(10, 30, 45)), type = "response")
rbind(educ_superior = c(10, 30, 45), probabilidad = p_tres, efecto_marginal = coef(logit_metro)[2] * p_tres * (1 - p_tres))

## la foto: los tramos, la recta que se sale de [0,1] y la curva en S
malla <- data.frame(educ_superior = seq(0, 60, 0.5))
plot(metro_train$educ_superior, metro_train$gana_petro, pch = 16, col = "gray70", xlab = "educacion superior (%)",
     ylab = "gana Petro (0/1) y probabilidad", main = "Area metropolitana: LPM (rojo) y logistica (azul)", ylim = c(-0.1, 1.2))
points(tramos$educ_superior, tramos$gana_petro, pch = 17, cex = 1.4)
abline(lpm_metro, col = "red", lwd = 2)
lines(malla$educ_superior, predict(logit_metro, malla, type = "response"), col = "blue", lwd = 2)
abline(h = c(0, 1), lty = 3)

## y la exactitud ya engaña: "Petro gana en todos" contra la logistica, en los 86 de prueba
c(gana_en_todos = mean(metro_test$gana_petro == 1),
  logistica = mean((predict(logit_metro, metro_test, type = "response") >= 0.5) == metro_test$gana_petro),
  pierde_en_la_prueba = sum(metro_test$gana_petro == 0))

##============================================================================##
##=== 4. La logistica en el pais: del predictor unico a la ficha           ===##
##============================================================================##

## la educacion superior sola: como la recta de la semana 2, no viaja fuera de Cali
logit_educ <- glm(gana_petro ~ educ_superior, family = binomial, data = train)
summary(logit_educ)$coefficients

## nueve variables (las de la semana 2): casi todas significativas
train <- train %>% mutate(estrato_bajo = estrato_1 + estrato_2, estrato_alto = estrato_5 + estrato_6)
test <- test %>% mutate(estrato_bajo = estrato_1 + estrato_2, estrato_alto = estrato_5 + estrato_6)
logit_nueve <- glm(gana_petro ~ educ_superior + estrato_bajo + estrato_alto + edad_20_29 + edad_60_74 + mujeres + afro +
                                serv_gas + log_registrados, family = binomial, data = train)
round(cbind(summary(logit_nueve)$coefficients, odds_ratio = exp(coef(logit_nueve))), 4)

## la ficha completa y la ficha con departamentos (el departamento pesa: Petro gano el
## 3 % de los puestos de Casanare y el 97 % de los de Putumayo)
f_ficha <- as.formula(paste("gana_petro ~", paste(x_ficha, collapse = " + ")))
logit_ficha <- glm(f_ficha, family = binomial, data = train)
logit_dpto <- glm(f_gana, family = binomial, data = train)

## las probabilidades en la prueba
p_educ <- predict(logit_educ, test, type = "response")
p_nueve <- predict(logit_nueve, test, type = "response")
p_ficha <- predict(logit_ficha, test, type = "response")
p_dpto <- predict(logit_dpto, test, type = "response")
round(rbind(educ = metricas(test$gana_petro, p_educ), nueve = metricas(test$gana_petro, p_nueve),
            ficha = metricas(test$gana_petro, p_ficha), ficha_dpto = metricas(test$gana_petro, p_dpto)), 3)

## la flexibilidad se paga: la AUC dentro y en la prueba
c(dentro = auc(train$gana_petro, fitted(logit_dpto)), prueba = auc(test$gana_petro, p_dpto))

##============================================================================##
##=== 5. Dos contrastes con las herramientas de la semana 3                ===##
##============================================================================##

## la logistica penalizada sobre la ficha larga: las 1.253 columnas de la semana 3
## (cada predictor con una pendiente por departamento), lambda por CV con la AUC
f_larga <- as.formula(paste("~ (", paste(x_ficha, collapse = " + "), ") * departamento"))
x_larga <- model.matrix(f_larga, data = rbind(train, test))[, -1]
x_larga_train <- x_larga[1:nrow(train), ]
x_larga_test <- x_larga[-(1:nrow(train)), ]
ncol(x_larga_train)
set.seed(2026)
cv_penal <- cv.glmnet(x_larga_train, train$gana_petro, family = "binomial", alpha = 1, nfolds = 5, type.measure = "auc",
                      nlambda = 25, lambda.min.ratio = 0.005, control = list(thresh = 1e-5))
plot(cv_penal)
c(coeficientes = sum(coef(cv_penal, s = "lambda.min") != 0) - 1, auc_cv = max(cv_penal$cvm))
p_penal <- as.numeric(predict(cv_penal, x_larga_test, s = "lambda.min", type = "response"))

## k-NN en el mapa: la misma funcion de las semanas 2 y 3. el promedio de una y binaria
## entre los k vecinos es la fraccion de vecinos donde gano Petro: una probabilidad
knn_reg <- function(coord_train, y_train, coord_nueva, k) {
           prediccion <- rep(NA, nrow(coord_nueva))
           for (i in 1:nrow(coord_nueva)) {
             distancia <- (coord_train[, 1] - coord_nueva[i, 1])^2 + (coord_train[, 2] - coord_nueva[i, 2])^2
             prediccion[i] <- mean(y_train[order(distancia)[1:k]])
           }
           return(prediccion)
}

## la k se elige por CV de 5 pliegues con la AUC (tarda cerca de medio minuto)
set.seed(2026)
pliegue <- sample(rep(1:5, length.out = nrow(train)))
ks <- c(1, 3, 5, 10, 15, 25, 50, 100)
auc_k <- c()
for (k in ks) {
  p_cv <- rep(NA, nrow(train))
  for (j in 1:5) {
    a <- train[pliegue != j, ]
    b <- train[pliegue == j, ]
    p_cv[pliegue == j] <- knn_reg(as.matrix(a[, c("lon", "lat")]), a$gana_petro, as.matrix(b[, c("lon", "lat")]), k)
  }
  auc_k <- c(auc_k, auc(train$gana_petro, p_cv))
}
round(rbind(k = ks, auc_cv = auc_k), 3)
k_cv <- ks[which.max(auc_k)]
p_knn <- knn_reg(as.matrix(train[, c("lon", "lat")]), train$gana_petro, as.matrix(test[, c("lon", "lat")]), k_cv)

##============================================================================##
##=== 6. Las metricas: la misma prueba para todos                          ===##
##============================================================================##

## la matriz de confusion de la ficha con departamentos, con el umbral 0,5
table(real = test$gana_petro, pred = as.integer(p_dpto >= 0.5))

## la tabla de hoy (el atajo de la regresion no entrega probabilidad: su umbral son 50 puntos)
tabla <- rbind(gana_en_todos = metricas(test$gana_petro, rep(1, nrow(test))),
               logit_educ = metricas(test$gana_petro, p_educ),
               logit_nueve = metricas(test$gana_petro, p_nueve),
               logit_ficha = metricas(test$gana_petro, p_ficha),
               regresion_semana_3 = metricas(test$gana_petro, voto_pred / 100),
               logit_ficha_dpto = metricas(test$gana_petro, p_dpto),
               logit_penalizada = metricas(test$gana_petro, p_penal),
               knn_mapa = metricas(test$gana_petro, p_knn))
round(tabla, 3)

## cuando una clase domina: 2010, Santos contra Mockus
puestos_2010 <- import("input/puestos_colombia_2010.rds", trust = T) %>%
                filter(!is.na(educ_superior), !is.na(afro), !is.na(estrato_bajo))
train_2010 <- puestos_2010 %>% filter(muestra == "entrenamiento")
test_2010 <- puestos_2010 %>% filter(muestra == "prueba")
mean(test_2010$gana_santos)
logit_2010 <- glm(gana_santos ~ educ_superior + estrato_bajo + estrato_alto + servicios + edad_0_19 + edad_20_29 + edad_45_59 +
                                edad_60_74 + edad_75 + mujeres + afro + indigena + log(poblacion) + log(registrados) + zona +
                                departamento, family = binomial, data = train_2010)
p_2010 <- predict(logit_2010, test_2010, type = "response")
table(real = test_2010$gana_santos, pred = as.integer(p_2010 >= 0.5))

## dos elecciones, dos prevalencias, las mismas metricas
round(cbind(eleccion_2010 = c(prevalencia = mean(test_2010$gana_santos), metricas(test_2010$gana_santos, p_2010)),
            eleccion_2022 = c(prevalencia = mean(test$gana_petro), metricas(test$gana_petro, p_dpto))), 3)

## los positivos raros: buscar los puestos de Mockus (12 % de la prueba). la tasa de falsos
## positivos casi no se mueve; la precision cuenta cuantas alarmas eran ciertas
p_mockus <- 1 - p_2010
gana_mockus <- 1 - test_2010$gana_santos
umbrales <- c(0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
round(data.frame(umbral = umbrales,
                 alarmas = sapply(umbrales, function(u) sum(p_mockus >= u)),
                 recall = sapply(umbrales, function(u) mean(p_mockus[gana_mockus == 1] >= u)),
                 precision = sapply(umbrales, function(u) mean(gana_mockus[p_mockus >= u])),
                 tasa_fp = sapply(umbrales, function(u) mean(p_mockus[gana_mockus == 0] >= u))), 3)

##============================================================================##
##=== 7. El umbral es una decision: la campaña y sus costos                ===##
##============================================================================##

## la campaña "da por seguro" un puesto si el modelo dice que gana, y no manda recursos.
## dar por seguro un puesto que se pierde cuesta 5; trabajar un puesto que igual se ganaba, 1
c_fp <- 5
c_fn <- 1
umbral_costos <- c_fp / (c_fp + c_fn)
umbral_costos
costo <- function(y, p, umbral) c_fp * sum(p >= umbral & y == 0) + c_fn * sum(p < umbral & y == 1)

## la misma logistica, las mismas probabilidades, dos umbrales
table(real = test$gana_petro, pred = as.integer(p_dpto >= 0.5))
table(real = test$gana_petro, pred = as.integer(p_dpto >= umbral_costos))
round(rbind(umbral_0.5 = metricas(test$gana_petro, p_dpto, 0.5), umbral_costos = metricas(test$gana_petro, p_dpto, umbral_costos)), 3)
c(trabajar_todos = costo(test$gana_petro, p_dpto, 1.1), dar_todos_por_seguros = costo(test$gana_petro, p_dpto, 0),
  umbral_0.5 = costo(test$gana_petro, p_dpto, 0.5), umbral_costos = costo(test$gana_petro, p_dpto, umbral_costos))

## la curva del costo total contra el umbral: el minimo cae donde dijo la formula
malla_umbral <- seq(0.05, 0.95, 0.05)
costo_umbral <- sapply(malla_umbral, function(u) costo(test$gana_petro, p_dpto, u))
plot(malla_umbral, costo_umbral, type = "b", pch = 16, xlab = "umbral", ylab = "costo total", main = "El costo de la campaña segun el umbral")
abline(v = umbral_costos, lty = 2)

## y con mejores probabilidades, menor costo con el mismo umbral
c(ficha_dpto = costo(test$gana_petro, p_dpto, umbral_costos), penalizada = costo(test$gana_petro, p_penal, umbral_costos),
  knn_mapa = costo(test$gana_petro, p_knn, umbral_costos))

## la curva ROC: todos los umbrales a la vez; los dos de hoy son dos puntos de la misma curva
roc <- function(y, p) {
       umbrales <- sort(unique(c(p, 2)), decreasing = T)
       data.frame(tasa_fp = sapply(umbrales, function(u) mean(p[y == 0] >= u)),
                  tasa_vp = sapply(umbrales, function(u) mean(p[y == 1] >= u)))
}
roc_dpto <- roc(test$gana_petro, p_dpto)
roc_penal <- roc(test$gana_petro, p_penal)
roc_nueve <- roc(test$gana_petro, p_nueve)
plot(roc_dpto$tasa_fp, roc_dpto$tasa_vp, type = "l", lwd = 2, col = "blue", xlab = "tasa de falsos positivos",
     ylab = "tasa de verdaderos positivos", main = "ROC: nueve (gris), ficha + dpto (azul), penalizada (rojo)")
lines(roc_penal$tasa_fp, roc_penal$tasa_vp, lwd = 2, col = "red")
lines(roc_nueve$tasa_fp, roc_nueve$tasa_vp, lwd = 2, col = "gray50")
abline(0, 1, lty = 3)
points(c(mean(p_dpto[test$gana_petro == 0] >= 0.5), mean(p_dpto[test$gana_petro == 0] >= umbral_costos)),
       c(mean(p_dpto[test$gana_petro == 1] >= 0.5), mean(p_dpto[test$gana_petro == 1] >= umbral_costos)), pch = 16, cex = 1.3)

##============================================================================##
##=== 8. Calibracion: ¿0,8 significa 80 %?                                  ===##
##============================================================================##

## por deciles de la probabilidad predicha: la media predicha contra la frecuencia observada
calibracion <- function(y, p) {
               decil <- ntile(p, 10)
               rbind(predicha = tapply(p, decil, mean), observada = tapply(y, decil, mean))
}
round(calibracion(test$gana_petro, p_dpto), 3)
round(calibracion(test$gana_petro, p_penal), 3)
round(calibracion(test$gana_petro, p_knn), 3)

## dos metricas de probabilidad: el Brier (el MSE de las probabilidades) y la log-loss
## (lo que la logistica minimiza). k-NN dice 0 o 1 exactos cuando los diez vecinos coinciden
## y a veces se equivoca: sin recortar las probabilidades su log-loss seria infinita
brier <- function(y, p) mean((p - y)^2)
logloss <- function(y, p) {
           p <- pmin(pmax(p, 0.001), 0.999)
           -mean(y * log(p) + (1 - y) * log(1 - p))
}
round(rbind(brier = c(referencia = brier(test$gana_petro, mean(train$gana_petro)), ficha_dpto = brier(test$gana_petro, p_dpto),
                      penalizada = brier(test$gana_petro, p_penal), knn_mapa = brier(test$gana_petro, p_knn)),
            logloss = c(logloss(test$gana_petro, mean(train$gana_petro)), logloss(test$gana_petro, p_dpto),
                        logloss(test$gana_petro, p_penal), logloss(test$gana_petro, p_knn))), 3)

##============================================================================##
##=== 9. Donde falla: los puestos reñidos                                   ===##
##============================================================================##

## la exactitud segun el margen con que se definio el puesto: cerca del empate nadie acierta
test %>%
mutate(acierta = as.integer((p_penal >= 0.5) == gana_petro),
       margen = cut(abs(voto_petro - 50), c(0, 5, 10, 20, 30, 50), include.lowest = T)) %>%
group_by(margen) %>%
summarise(puestos = n(), exactitud = mean(acierta), .groups = "drop")

## y por zona
test %>%
mutate(acierta = as.integer((p_penal >= 0.5) == gana_petro)) %>%
group_by(zona) %>%
summarise(puestos = n(), gana_petro = mean(gana_petro), exactitud = mean(acierta), .groups = "drop")

##============================================================================##
##=== 10. La tabla del líder: clasificación, la misma prueba para todos     ===##
##============================================================================##

lider <- data.frame(semana = 4,
                    modelo = c("Petro gana en todos", "logística, educación superior", "logística, nueve variables",
                               "logística, ficha completa (37)", "regresión de la semana 3 (gana si predice más de 50)",
                               "logística, ficha + departamentos", "logística penalizada, ficha larga",
                               paste0("k-NN en el mapa (k = ", k_cv, ")")),
                    auc_prueba = tabla[, "auc"], exactitud = tabla[, "exactitud"])
rownames(lider) <- NULL
lider %>% arrange(desc(auc_prueba))

## la regla vigente y la ficha con departamentos, en los mismos puestos con historia
con_historia <- !is.na(test$voto_petro_2018)
round(rbind(gano_en_2018 = metricas(test$gana_petro[con_historia], test$voto_petro_2018[con_historia] / 100),
            ficha_dpto = metricas(test$gana_petro[con_historia], p_dpto[con_historia]),
            penalizada = metricas(test$gana_petro[con_historia], p_penal[con_historia]),
            knn_mapa = metricas(test$gana_petro[con_historia], p_knn[con_historia])), 3)
