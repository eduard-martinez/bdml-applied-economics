##============================================================================##
# week-05.R  —  Árboles, bosques y boosting
#------------------------------------------------------------------------------#
# La pregunta de hoy: la semana 3 escribimos 1.253 columnas a mano para que la
# educación pesara distinto en cada departamento. ¿Puede un algoritmo encontrar
# solo las particiones del territorio que importan, y mejorar a los dos líderes
# (Lasso 14,7 y k-NN 12,1; en clasificación, 0,937 y 0,945)?
# Qué lee (carpeta input/, grano: puesto de votación):
#   - ficha_puestos_2022.rds  los puestos de 2022 con la ficha de 37 predictores,
#                             voto_petro, gana_petro y la partición fija (muestra)
# Qué produce: gráficos en pantalla y tablas en consola; no exporta nada.
# Fijar el directorio de trabajo en applications/week-05. Tarda unos 5 minutos:
# los bosques de 9.602 puestos no son gratis.
##============================================================================##

## configuracion inicial
rm(list = ls())
if (!require(pacman)) install.packages("pacman")
pacman::p_load(rio, dplyr, rpart, randomForest, xgboost)

## las metricas del curso
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
auc <- function(y, p) {
       rango <- rank(p)
       (sum(rango[y == 1]) - sum(y == 1) * (sum(y == 1) + 1) / 2) / (sum(y == 1) * sum(y == 0))
}

##============================================================================##
##=== 1. Los datos: la ficha, la particion y las formulas                  ===##
##============================================================================##

## los puestos de 2022 con su ficha; la misma particion de las semanas 3 y 4
puestos <- import("input/ficha_puestos_2022.rds", trust = T)
x_ficha <- setdiff(names(puestos), c("puesto", "nombre", "departamento", "municipio", "cod_municipio", "zona", "lon", "lat",
                                     "registrados", "voto_petro", "gana_petro", "participacion", "voto_petro_2018", "muestra"))
train <- puestos %>% filter(muestra == "entrenamiento")
test <- puestos %>% filter(muestra == "prueba")

## la ficha sola (para arboles que se puedan leer) y la ficha con departamento
f_ficha <- as.formula(paste("voto_petro ~", paste(x_ficha, collapse = " + ")))
f_dpto <- as.formula(paste("voto_petro ~", paste(c(x_ficha, "departamento"), collapse = " + ")))

##============================================================================##
##=== 2. Un corte a mano: diez puestos de Cali                             ===##
##============================================================================##

## los cinco puestos de Cali con menos educacion superior y los cinco con mas
cali <- puestos %>% filter(municipio == "CALI")
toy <- bind_rows(cali %>% arrange(educ_superior) %>% head(5),
                 cali %>% arrange(desc(educ_superior)) %>% head(5)) %>%
       arrange(educ_superior) %>%
       select(nombre, educ_superior, voto_petro)
toy

## la referencia: sin cortar, la prediccion es la media y el RSS mide lo que falta
rss <- function(y) sum((y - mean(y))^2)
c(media = mean(toy$voto_petro), rss = rss(toy$voto_petro))

## los nueve cortes posibles (el punto medio entre puestos consecutivos): para cada uno,
## la media a cada lado y el RSS que queda. la maquina hace exactamente esta cuenta
cortes <- data.frame(corte = (toy$educ_superior[1:9] + toy$educ_superior[2:10]) / 2)
for (i in 1:9) {
  izq <- toy$voto_petro[toy$educ_superior < cortes$corte[i]]
  der <- toy$voto_petro[toy$educ_superior >= cortes$corte[i]]
  cortes$media_izq[i] <- mean(izq)
  cortes$media_der[i] <- mean(der)
  cortes$rss[i] <- rss(izq) + rss(der)
}
round(cortes, 1)

## el mejor corte parte a Cali en dos ciudades
plot(toy$educ_superior, toy$voto_petro, pch = 16, xlab = "educacion superior (%)", ylab = "voto por Petro (pp)",
     main = "Diez puestos de Cali y el corte que elige el arbol")
abline(v = cortes$corte[which.min(cortes$rss)], lty = 2)

##============================================================================##
##=== 3. El arbol del pais: crecer, leer y podar                           ===##
##============================================================================##

## un arbol chico sobre la ficha: la misma cuenta, 9.602 puestos y 37 variables a la vez
arbol <- rpart(f_ficha, data = train, control = rpart.control(cp = 0.01))
arbol
c(hojas = sum(arbol$frame$var == "<leaf>"),
  dentro = rmse(train$voto_petro, predict(arbol, train)), prueba = rmse(test$voto_petro, predict(arbol, test)))
plot(arbol, margin = 0.05)
text(arbol, cex = 0.7)

## el primer corte del pais: afro. comparen con el mejor corte de otras variables
## (la particion binaria recursiva evalua todos los cortes de todas las variables)
mejor_corte <- function(x, y) {
               candidatos <- unique(quantile(x, seq(0.02, 0.98, 0.02)))
               candidatos <- candidatos[candidatos > min(x)]
               perdida <- sapply(candidatos, function(s) rss(y[x < s]) + rss(y[x >= s]))
               c(corte = candidatos[which.min(perdida)], rss_millones = min(perdida) / 1e6)
}
round(rbind(raiz = c(NA, rss(train$voto_petro) / 1e6),
            educ_superior = mejor_corte(train$educ_superior, train$voto_petro),
            edad_60_74 = mejor_corte(train$edad_60_74, train$voto_petro),
            afro = mejor_corte(train$afro, train$voto_petro)), 2)

## crecer un arbol grande y podarlo con la curva de CV (cp es el alpha de la poda)
set.seed(2026)
arbol_grande <- rpart(f_dpto, data = train, control = rpart.control(cp = 0.0001, xval = 5))
sum(arbol_grande$frame$var == "<leaf>")
tabla_cp <- as.data.frame(arbol_grande$cptable)
plotcp(arbol_grande)

## el minimo de la curva y la regla 1-SE, como en la semana 3
cp_min <- tabla_cp[which.min(tabla_cp$xerror), ]
cp_1se <- tabla_cp[which(tabla_cp$xerror <= cp_min$xerror + cp_min$xstd)[1], ]
rbind(minimo = cp_min, una_se = cp_1se)
arbol_podado <- prune(arbol_grande, cp = cp_min$CP)
arbol_1se <- prune(arbol_grande, cp = cp_1se$CP)
rbind(grande = c(hojas = sum(arbol_grande$frame$var == "<leaf>"), prueba = rmse(test$voto_petro, predict(arbol_grande, test))),
      podado = c(sum(arbol_podado$frame$var == "<leaf>"), rmse(test$voto_petro, predict(arbol_podado, test))),
      una_se = c(sum(arbol_1se$frame$var == "<leaf>"), rmse(test$voto_petro, predict(arbol_1se, test))))

##============================================================================##
##=== 4. El bosque: promediar arboles descorrelacionados                   ===##
##============================================================================##

## la matriz de predictores: la ficha, el departamento y, ahora si, las coordenadas
## (el arbol las corta como a cualquier variable: un corte en lon y uno en lat es un rectangulo)
x_train <- train[, c(x_ficha, "lon", "lat")] %>% mutate(departamento = factor(train$departamento))
x_test <- test[, c(x_ficha, "lon", "lat")] %>% mutate(departamento = factor(test$departamento, levels = levels(x_train$departamento)))

## 400 arboles sobre remuestras bootstrap, mtry = p/3 variables por corte (tarda ~2 min)
set.seed(2026)
bosque <- randomForest(x = x_train, y = train$voto_petro, ntree = 400, importance = T)
bosque

## el error out-of-bag por numero de arboles: cae rapido y se aplana; un arbol solo, 18
curva_oob <- data.frame(arboles = c(1, 5, 10, 25, 50, 100, 200, 400), oob = sqrt(bosque$mse[c(1, 5, 10, 25, 50, 100, 200, 400)]))
round(curva_oob, 2)
plot(sqrt(bosque$mse), type = "l", xlab = "arboles", ylab = "RMSE out-of-bag", main = "El bosque: promediar baja la varianza")

## el OOB anticipa la prueba sin abrirla
c(oob = sqrt(tail(bosque$mse, 1)), prueba = rmse(test$voto_petro, predict(bosque, x_test)))

## mtry: cuantas variables sortea cada corte (menos variables = arboles mas distintos)
for (m in c(2, 13, 40)) {
  set.seed(2026)
  b <- randomForest(x = x_train, y = train$voto_petro, ntree = 150, mtry = m)
  cat("mtry", m, ": OOB", round(sqrt(tail(b$mse, 1)), 2), "\n")
}

## la importancia nativa: cuanto sube el error OOB al permutar cada variable
importancia <- importance(bosque)[, "%IncMSE"]
round(sort(importancia, decreasing = T)[1:10], 1)

## ¿y sin ubicacion? el bosque solo con la ficha y el departamento: la liga de la estructura
set.seed(2026)
bosque_ficha <- randomForest(x = x_train %>% select(-lon, -lat), y = train$voto_petro, ntree = 400)
c(oob = sqrt(tail(bosque_ficha$mse, 1)), prueba = rmse(test$voto_petro, predict(bosque_ficha, x_test %>% select(-lon, -lat))))

##============================================================================##
##=== 5. Boosting: arboles chicos sobre los residuos                       ===##
##============================================================================##

## xgboost pide una matriz numerica: el departamento se abre en columnas
m_train <- model.matrix(~ . - 1, data = x_train)
m_test <- model.matrix(~ . - 1, data = x_test)
d_train <- xgb.DMatrix(m_train, label = train$voto_petro)

## la ronda se elige por CV con parada temprana: si el CV no mejora en 50 rondas, parar
parametros <- list(eta = 0.05, max_depth = 5, subsample = 0.8, colsample_bytree = 0.8)
set.seed(2026)
cv_boost <- xgb.cv(params = parametros, data = d_train, nrounds = 2000, nfold = 5, early_stopping_rounds = 50, verbose = 0)
rondas <- cv_boost$early_stop$best_iteration
rondas

## la curva que hay que saber leer: el error dentro baja sin parar; el CV se aplana
curva <- cv_boost$evaluation_log
plot(curva$iter, curva$train_rmse_mean, type = "l", ylim = c(0, 25), xlab = "rondas", ylab = "RMSE",
     main = "Boosting: dentro (gris) y CV (rojo)", col = "gray50")
lines(curva$iter, curva$test_rmse_mean, col = "red", lwd = 2)
abline(v = rondas, lty = 2)
round(curva[c(10, 50, 100, 200, 400, rondas), c("iter", "train_rmse_mean", "test_rmse_mean")], 2)

## reajustar con las rondas elegidas y abrir la prueba
boost <- xgb.train(params = parametros, data = d_train, nrounds = rondas)
rmse(test$voto_petro, predict(boost, xgb.DMatrix(m_test)))

## la perilla que mas importa: eta grande aprende rapido y sobreajusta
## (con eta = 0,3 el CV toca su minimo antes de la ronda 120 y despues empeora)
set.seed(2026)
cv_rapido <- xgb.cv(params = list(eta = 0.3, max_depth = 5), data = d_train, nrounds = 300, nfold = 5, verbose = 0)
curva_rapida <- cv_rapido$evaluation_log
c(mejor_ronda = which.min(curva_rapida$test_rmse_mean), cv_minimo = min(curva_rapida$test_rmse_mean),
  cv_ronda_300 = curva_rapida$test_rmse_mean[300], dentro_ronda_300 = curva_rapida$train_rmse_mean[300])

##============================================================================##
##=== 6. La otra pregunta: gana_petro con los mismos tres metodos          ===##
##============================================================================##

## el arbol de clasificacion: gini en vez de suma de cuadrados; se deja leer
arbol_clas <- rpart(as.formula(paste("gana_petro ~", paste(x_ficha, collapse = " + "))),
                    data = train, method = "class", control = rpart.control(cp = 0.01))
arbol_clas
p_arbol <- predict(arbol_clas, test)[, 2]

## el bosque y el boosting clasificadores (~1 min)
set.seed(2026)
bosque_clas <- randomForest(x = x_train, y = factor(train$gana_petro), ntree = 400)
p_bosque <- predict(bosque_clas, x_test, type = "prob")[, 2]
parametros_clas <- c(parametros, list(objective = "binary:logistic", eval_metric = "auc"))
d_train_clas <- xgb.DMatrix(m_train, label = train$gana_petro)
set.seed(2026)
cv_clas <- xgb.cv(params = parametros_clas, data = d_train_clas, nrounds = 2000, nfold = 5, early_stopping_rounds = 50, verbose = 0)
boost_clas <- xgb.train(params = parametros_clas, data = d_train_clas, nrounds = cv_clas$early_stop$best_iteration)
p_boost <- predict(boost_clas, xgb.DMatrix(m_test))

## la tabla de clasificacion: AUC y exactitud con umbral 0,5
round(rbind(arbol = c(auc = auc(test$gana_petro, p_arbol), exactitud = mean((p_arbol >= 0.5) == test$gana_petro)),
            bosque = c(auc(test$gana_petro, p_bosque), mean((p_bosque >= 0.5) == test$gana_petro)),
            boosting = c(auc(test$gana_petro, p_boost), mean((p_boost >= 0.5) == test$gana_petro))), 3)

## el umbral de la campaña de la semana 4 (0,83), ahora con el bosque
costo <- function(p, umbral) 5 * sum(p >= umbral & test$gana_petro == 0) + sum(p < umbral & test$gana_petro == 1)
c(logistica_semana_4 = 695, bosque_0.5 = costo(p_bosque, 0.5), bosque_0.83 = costo(p_bosque, 5/6))

##============================================================================##
##=== 7. La tabla del líder                                                 ===##
##============================================================================##

## regresion: el RMSE de prueba, los mismos 2.398 puestos de las semanas 3 y 4
lider <- data.frame(semana = c(3, 3, 3, 5, 5, 5, 5, 5),
                    modelo = c("Lasso, ficha larga", "ficha + departamentos", "k-NN en el mapa (k = 10)",
                               "arbol podado por CV", "bosque (ficha + dpto)", "bosque (+ coordenadas)",
                               "boosting (+ coordenadas)", "arbol de 10 hojas"),
                    rmse_prueba = c(14.75, 15.63, 12.13,
                                    rmse(test$voto_petro, predict(arbol_podado, test)),
                                    rmse(test$voto_petro, predict(bosque_ficha, x_test %>% select(-lon, -lat))),
                                    rmse(test$voto_petro, predict(bosque, x_test)),
                                    rmse(test$voto_petro, predict(boost, xgb.DMatrix(m_test))),
                                    rmse(test$voto_petro, predict(arbol, test))))
lider %>% arrange(rmse_prueba)

## clasificacion: la AUC de prueba
lider_clas <- data.frame(semana = c(4, 4, 5, 5, 5),
                         modelo = c("logistica penalizada, ficha larga", "k-NN en el mapa (k = 10)",
                                    "arbol de clasificacion", "bosque (+ coordenadas)", "boosting (+ coordenadas)"),
                         auc_prueba = c(0.937, 0.945, auc(test$gana_petro, p_arbol),
                                        auc(test$gana_petro, p_bosque), auc(test$gana_petro, p_boost)))
lider_clas %>% arrange(desc(auc_prueba))
