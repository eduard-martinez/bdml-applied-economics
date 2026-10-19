## Big Data y Machine Learning para Economia Aplicada
## week-05-application: la clase practica de arboles. Un arbol sencillo que se puede
## leer, la profundidad tuneada con la validacion cruzada que caret trae adentro, el
## arbol sin frenos que sobreajusta, un bosque aleatorio tuneado y un xgboost. La
## replica completa de las diapositivas es week-05-replication.R
## Last run: Oct 19, 2026

##==: 0. initial setup :==##

## clean environment
rm(list=ls())

## load/install packages
require(pacman)
p_load(rio , dplyr , rpart , caret , randomForest , xgboost)

##==: 1. prepare data :==##

## 1.1. load data (la base publica del curso: la carpeta input/ del zip o github)
url <- "https://raw.githubusercontent.com/eduard-martinez/bdml-applied-economics/main/applications/week-05/input/"
puestos <- import("input/puestos_colombia_2022.rds" , trust=T)
censo37 <- import("input/censo37_puestos_2022.rds" , trust=T)

## 1.2. la base de hoy: el voto por petro (%) de cada puesto y las 37 variables del censo
db <- puestos %>%
      select(puesto , nombre , departamento , voto_petro , muestra) %>%
      inner_join(censo37 , by="puesto") %>%
      subset(!is.na(educ_superior) & !is.na(afro))
x_censo <- setdiff(names(censo37) , "puesto")
f_censo <- as.formula(paste0("voto_petro ~ " , paste(x_censo , collapse=" + ")))

## 1.3. split data (la particion fija del curso: 9.602 para aprender y 2.398 para probar)
train <- db %>% subset(muestra=="entrenamiento")
test <- db %>% subset(muestra=="prueba")
c(train = nrow(train) , test = nrow(test))

## 1.4. la referencia: predecir todos los puestos con el promedio del train
## mira: rmse 27,3 / 26,7 y mae 23,6 / 22,9. Todo lo que sigue se compara con esto
c(rmse_train = sqrt(mean((train$voto_petro - mean(train$voto_petro))^2)) ,
  rmse_test = sqrt(mean((test$voto_petro - mean(train$voto_petro))^2)) ,
  mae_test = mean(abs(test$voto_petro - mean(train$voto_petro))))

##==: 2. arbol A: sencillo y se puede leer :==##

## 2.1. estimar el arbol con las opciones por defecto de rpart
arbol_a <- rpart(f_censo , data=train)

## 2.2. el arbol en texto y dibujado: cada nodo es una pregunta de si o no
## mira: el primer corte es la poblacion afro en 4,6% (la lamina 9 del deck)
arbol_a
plot(arbol_a , uniform=T , margin=0.05)
text(arbol_a , use.n=T , cex=0.7)

## 2.3. que hay en cada hojita: cuantos puestos caen y que voto promedio predice
hojas_a <- arbol_a$frame[arbol_a$frame$var=="<leaf>" , c("n","yval")]
colnames(hojas_a) <- c("puestos" , "voto_predicho")
round(hojas_a , 1)

## 2.4. predecir: a cada puesto le toca el promedio de su hoja
train$pred_arbol_a <- predict(arbol_a , train)
test$pred_arbol_a <- predict(arbol_a , test)

## 2.5. rmse y mae, en el train y en el test
## mira: 21,2 / 21,4 de rmse: casi iguales. Este arbol no se aprendio nada de memoria
c(rmse_train = sqrt(mean((train$voto_petro - train$pred_arbol_a)^2)) ,
  rmse_test = sqrt(mean((test$voto_petro - test$pred_arbol_a)^2)) ,
  mae_test = mean(abs(test$voto_petro - test$pred_arbol_a)))

##==: 3. arbol B: la profundidad tuneada con validacion cruzada :==##

## 3.1. la perilla del arbol es la profundidad (cuantas preguntas encadena). caret
## la tunea solo: trainControl pide cv de 5 pliegues y tuneGrid da la grilla; no
## hay que programar la validacion cruzada a mano
set.seed(2026)
arbol_b <- train(f_censo , data=train , method="rpart2" ,
                 trControl=trainControl(method="cv" , number=5) ,
                 tuneGrid=expand.grid(maxdepth=c(1,2,3,4,5,6,8,10,15,20,30)) ,
                 control=rpart.control(cp=0))

## 3.2. la tabla del tuning: rmse y mae de validacion cruzada por profundidad
## mira: baja hasta la profundidad 20 y en 30 ya no mejora; caret se queda con 20
arbol_b$results[,c("maxdepth","RMSE","MAE")]
arbol_b$bestTune
plot(arbol_b)

## 3.3. predecir con el arbol elegido
train$pred_arbol_b <- predict(arbol_b , train)
test$pred_arbol_b <- predict(arbol_b , test)

## 3.4. rmse y mae
## mira: 20,0 en el test: mejor que el arbol A (21,4). Tunear la perilla paga
c(rmse_train = sqrt(mean((train$voto_petro - train$pred_arbol_b)^2)) ,
  rmse_test = sqrt(mean((test$voto_petro - test$pred_arbol_b)^2)) ,
  mae_test = mean(abs(test$voto_petro - test$pred_arbol_b)))

##==: 4. el arbol sin frenos: sobreajusta demasiado :==##

## 4.1. mismo rpart, pero aceptando cualquier corte (cp = 0) y hojas de un solo
## puesto (minsplit = 2, minbucket = 1): el arbol crece hasta el fondo
arbol_sf <- rpart(f_censo , data=train ,
                  control=rpart.control(maxdepth=30 , cp=0 , minsplit=2 , minbucket=1))

## 4.2. cuantas hojas tiene
## mira: 9.402 hojas para 9.602 puestos: casi una hoja por puesto
sum(arbol_sf$frame$var=="<leaf>")

## 4.3. rmse y mae
## mira: en el train 0,4: casi perfecto. En el test 22,7: peor que el arbol A.
## Aprenderse el train de memoria no sirve para puestos nuevos; la cv de la
## seccion 3 jamas elige esto
c(rmse_train = sqrt(mean((train$voto_petro - predict(arbol_sf , train))^2)) ,
  rmse_test = sqrt(mean((test$voto_petro - predict(arbol_sf , test))^2)) ,
  mae_test = mean(abs(test$voto_petro - predict(arbol_sf , test))))

##==: 5. bosque aleatorio: muchos arboles y dos perillas :==##

## 5.1. las dos perillas de la clase: cuantos arboles (ntree) y cuantas variables
## compiten en cada corte (mtry). ntree no se tunea: mas arboles nunca empeoran el
## promedio (se fija en 200). mtry si: caret prueba la grilla y elige con el error
## fuera de la bolsa, la validacion que el bosque trae gratis (tarda unos 2 minutos)
set.seed(2026)
bosque <- train(x=train[,x_censo] , y=train$voto_petro , method="rf" ,
                trControl=trainControl(method="oob") ,
                tuneGrid=expand.grid(mtry=c(6,12,24)) , ntree=200)

## 5.2. el tuning y el bosque elegido
## mira: con mas variables por corte el error fuera de la bolsa baja un poco; la
## curva es casi plana (el bosque casi no necesita tuning)
bosque$results
bosque$finalModel

## 5.3. que hay adentro: ya no se puede leer un arbol, pero si ver que variables
## usa mas. mira: afro, indigena y estrato 2, las mismas del arbol A. La proxima
## clase abre esta caja en serio
varImpPlot(bosque$finalModel , n.var=10 , main="que variables usa el bosque")

## 5.4. predecir
train$pred_bosque <- predict(bosque , train[,x_censo])
test$pred_bosque <- predict(bosque , test[,x_censo])

## 5.5. rmse y mae
## mira: 15,8 en el test: de 20,0 del mejor arbol a 15,8 por promediar 200
c(rmse_train = sqrt(mean((train$voto_petro - train$pred_bosque)^2)) ,
  rmse_test = sqrt(mean((test$voto_petro - test$pred_bosque)^2)) ,
  mae_test = mean(abs(test$voto_petro - test$pred_bosque)))

##==: 6. xgboost: arboles en fila y la perilla de las rondas :==##

## 6.1. xgboost no recibe un data.frame: hay que pasarle matrices de numeros
x_train <- xgb.DMatrix(as.matrix(train[,x_censo]) , label=train$voto_petro)
x_test <- xgb.DMatrix(as.matrix(test[,x_censo]))

## 6.2. la perilla del boosting son las rondas (B), y aqui B si sobreajusta: se
## elige con validacion cruzada y parada temprana. caret aun no habla con esta
## version de xgboost, pero xgboost trae su propia cv (tarda medio minuto)
set.seed(2026)
cv_boost <- xgb.cv(params=xgb.params(objective="reg:squarederror" , eta=0.05 , max_depth=6 ,
                                     subsample=0.8 , colsample_bytree=0.8 ,
                                     eval_metric=c("mae","rmse") , nthread=1) ,
                   data=x_train , nrounds=1500 , nfold=5 ,
                   early_stopping_rounds=50 , verbose=0)

## 6.3. la curva del tuning: dentro del train el error solo baja; en validacion
## cruzada toca fondo y la parada temprana se detiene sola en la mejor ronda
log_cv <- as.data.frame(cv_boost$evaluation_log)
log_cv[c(1,10,50,100,300,600,cv_boost$early_stop$best_iteration) ,
       c("iter","train_rmse_mean","test_rmse_mean","test_mae_mean")]
plot(log_cv$iter , log_cv$train_rmse_mean , type="l" , col="gray60" , lwd=2 ,
     xlab="rondas (B)" , ylab="rmse" , ylim=c(0,28))
lines(log_cv$iter , log_cv$test_rmse_mean , col="red" , lwd=2)
legend("topright" , legend=c("train" , "validacion cruzada") , col=c("gray60","red") , lwd=2 , bty="n")

## 6.4. estimar con la mejor ronda y predecir
set.seed(2026)
boost <- xgb.train(params=xgb.params(objective="reg:squarederror" , eta=0.05 , max_depth=6 ,
                                     subsample=0.8 , colsample_bytree=0.8 , nthread=1) ,
                   data=x_train , nrounds=cv_boost$early_stop$best_iteration , verbose=0)
train$pred_boost <- predict(boost , x_train)
test$pred_boost <- predict(boost , x_test)

## 6.5. rmse y mae
## mira: 15,7 en el test: le gana por un pelo al bosque (15,8) y con mejor mae
c(rmse_train = sqrt(mean((train$voto_petro - train$pred_boost)^2)) ,
  rmse_test = sqrt(mean((test$voto_petro - test$pred_boost)^2)) ,
  mae_test = mean(abs(test$voto_petro - test$pred_boost)))

##==: 7. la tabla de hoy :==##

## 7.1. target original vs las cuatro predicciones pegadas
test %>%
     select(nombre , voto_petro , pred_arbol_a , pred_arbol_b , pred_bosque , pred_boost) %>%
     mutate(across(where(is.numeric) , ~ round(.x,1))) %>%
     head(5)

## 7.2. rmse y mae de todo lo que corrimos, en el train y en el test
## mira: el arbol sin frenos es el mejor en el train y el peor en el test; el
## bosque y el xgboost son los unicos que bajan de 16
tabla <- rbind(media = c(sqrt(mean((train$voto_petro - mean(train$voto_petro))^2)) , sqrt(mean((test$voto_petro - mean(train$voto_petro))^2)) , mean(abs(test$voto_petro - mean(train$voto_petro)))) ,
               arbol_a = c(sqrt(mean((train$voto_petro - train$pred_arbol_a)^2)) , sqrt(mean((test$voto_petro - test$pred_arbol_a)^2)) , mean(abs(test$voto_petro - test$pred_arbol_a))) ,
               arbol_b_cv = c(sqrt(mean((train$voto_petro - train$pred_arbol_b)^2)) , sqrt(mean((test$voto_petro - test$pred_arbol_b)^2)) , mean(abs(test$voto_petro - test$pred_arbol_b))) ,
               arbol_sin_frenos = c(sqrt(mean((train$voto_petro - predict(arbol_sf , train))^2)) , sqrt(mean((test$voto_petro - predict(arbol_sf , test))^2)) , mean(abs(test$voto_petro - predict(arbol_sf , test)))) ,
               bosque = c(sqrt(mean((train$voto_petro - train$pred_bosque)^2)) , sqrt(mean((test$voto_petro - test$pred_bosque)^2)) , mean(abs(test$voto_petro - test$pred_bosque))) ,
               xgboost = c(sqrt(mean((train$voto_petro - train$pred_boost)^2)) , sqrt(mean((test$voto_petro - test$pred_boost)^2)) , mean(abs(test$voto_petro - test$pred_boost))))
colnames(tabla) <- c("rmse_train" , "rmse_test" , "mae_test")
round(tabla , 1)

## 7.3. la misma tabla en un grafico: gris el train, rojo el test
barplot(t(tabla[,c("rmse_train","rmse_test")]) , beside=T , col=c("gray70","red") ,
        ylab="rmse (puntos del voto)" , cex.names=0.75)
legend("topright" , legend=c("train","test") , fill=c("gray70","red") , bty="n")

##==: 8. para la casa: el espacio de busqueda completo :==##

## 8.1. hasta aqui tuneamos una perilla a la vez. La busqueda de verdad recorre una
## grilla con todas las combinaciones: por ejemplo, de 500 a 1.000 arboles de 50 en
## 50, y de 5 a 20 variables por corte
grilla_casa <- expand.grid(ntree = seq(500 , 1000 , by=50) , mtry = 5:20)
nrow(grilla_casa)

## 8.2. eso son 176 bosques: horas de computador, no de clase. En caret la grilla de
## mtry entra en tuneGrid (ntree se fija en el valor mas alto, porque no sobreajusta):
## descomentar y correr en la casa
# bosque_casa <- train(x=train[,x_censo] , y=train$voto_petro , method="rf" ,
#                      trControl=trainControl(method="oob") ,
#                      tuneGrid=expand.grid(mtry=5:20) , ntree=1000)

## 8.3. en xgboost la grilla cruza la tasa y la profundidad, y cada combinacion
## corre su xgb.cv como el de la seccion 6: eta en c(0.01, 0.05, 0.1, 0.3) y
## max_depth en c(2, 4, 6, 8), por ejemplo. Y el menu queda abierto: en caret,
## method="glmnet" es el lasso de la semana 3 y method="knn" los vecinos;
## combinar modelos tambien se vale
