## Big Data y Machine Learning para Economia Aplicada
## week-05-application: la clase practica de arboles, con censo + departamento

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

## 1.2. la base de hoy: el voto por petro (%), las 37 del censo y el departamento
db <- puestos %>%
      select(puesto , nombre , departamento , voto_petro , muestra) %>%
      inner_join(censo37 , by="puesto") %>%
      mutate(dpto = factor(departamento)) %>%
      subset(!is.na(educ_superior) & !is.na(afro))
x_censo <- setdiff(names(censo37) , "puesto")
x_vars <- c(x_censo , "dpto")
f <- as.formula(paste0("voto_petro ~ " , paste(x_vars , collapse=" + ")))

## 1.3. split data (la particion fija del curso: 9.602 para aprender y 2.398 para probar)
train <- db %>% subset(muestra=="entrenamiento")
test <- db %>% subset(muestra=="prueba")
c(train = nrow(train) , test = nrow(test))

## 1.4. la referencia: predecir todos los puestos con el promedio del train
c(rmse_train = sqrt(mean((train$voto_petro - mean(train$voto_petro))^2)) ,
  rmse_test = sqrt(mean((test$voto_petro - mean(train$voto_petro))^2)) ,
  mae_test = mean(abs(test$voto_petro - mean(train$voto_petro))))

##==: 2. arbol A: sencillo y se puede leer :==##

## 2.1. estimar el arbol con las opciones por defecto de rpart
arbol_a <- rpart(f , data=train)

## 2.2. el arbol en texto y dibujado: cada nodo es una pregunta de si o no
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
c(rmse_train = sqrt(mean((train$voto_petro - train$pred_arbol_a)^2)) ,
  rmse_test = sqrt(mean((test$voto_petro - test$pred_arbol_a)^2)) ,
  mae_test = mean(abs(test$voto_petro - test$pred_arbol_a)))

##==: 3. arbol B: la profundidad tuneada con validacion cruzada :==##

## 3.1. la perilla del arbol es la profundidad (cuantas preguntas encadena). caret
## con x e y (no formula) el arbol usa el departamento entero, sin dummies
set.seed(2026)
arbol_b <- train(x=train[,x_vars] , y=train$voto_petro , method="rpart2" ,
                 trControl=trainControl(method="cv" , number=5) ,
                 tuneGrid=expand.grid(maxdepth=c(1,2,3,4,5,6,8,10,15,20,30)) ,
                 control=rpart.control(cp=0))

## 3.2. la tabla del tuning: rmse y mae de validacion cruzada por profundidad
arbol_b$results[,c("maxdepth","RMSE","MAE")]
arbol_b$bestTune
plot(arbol_b)

## 3.3. predecir con el arbol elegido
train$pred_arbol_b <- predict(arbol_b , train[,x_vars])
test$pred_arbol_b <- predict(arbol_b , test[,x_vars])

## 3.4. rmse y mae
c(rmse_train = sqrt(mean((train$voto_petro - train$pred_arbol_b)^2)) ,
  rmse_test = sqrt(mean((test$voto_petro - test$pred_arbol_b)^2)) ,
  mae_test = mean(abs(test$voto_petro - test$pred_arbol_b)))

##==: 4. el arbol sin frenos: sobreajusta demasiado :==##

## 4.1. mismo rpart, pero aceptando cualquier corte (cp = 0) y hojas de un solo
arbol_sf <- rpart(f , data=train ,
                  control=rpart.control(maxdepth=30 , cp=0 , minsplit=2 , minbucket=1))

## 4.2. cuantas hojas tiene
## mira: 9.484 hojas para 9.602 puestos: casi una hoja por puesto
sum(arbol_sf$frame$var=="<leaf>")

## 4.3. rmse y mae
c(rmse_train = sqrt(mean((train$voto_petro - predict(arbol_sf , train))^2)) ,
  rmse_test = sqrt(mean((test$voto_petro - predict(arbol_sf , test))^2)) ,
  mae_test = mean(abs(test$voto_petro - predict(arbol_sf , test))))

##==: 5. bosque aleatorio: muchos arboles y dos perillas :==##

## 5.1. las dos perillas de la clase: cuantos arboles (ntree) y cuantas variables
set.seed(2026)
bosque <- train(x=train[,x_vars] , y=train$voto_petro , method="rf" ,
                trControl=trainControl(method="oob") ,
                tuneGrid=expand.grid(mtry=c(6,12,24)) , ntree=200)

## 5.2. el tuning y el bosque elegido
bosque$results
bosque$finalModel

## 5.3. que hay adentro: ya no se puede leer un arbol, pero si ver que variables
varImpPlot(bosque$finalModel , n.var=10 , main="que variables usa el bosque")

## 5.4. predecir
train$pred_bosque <- predict(bosque , train[,x_vars])
test$pred_bosque <- predict(bosque , test[,x_vars])

## 5.5. rmse y mae
c(rmse_train = sqrt(mean((train$voto_petro - train$pred_bosque)^2)) ,
  rmse_test = sqrt(mean((test$voto_petro - test$pred_bosque)^2)) ,
  mae_test = mean(abs(test$voto_petro - test$pred_bosque)))

##==: 6. xgboost: arboles en fila y la perilla de las rondas :==##

## 6.1. xgboost no recibe factores: el departamento entra en 33 columnas de ceros y unos
x_train <- xgb.DMatrix(model.matrix(~ . - 1 , data=train[,x_vars]) , label=train$voto_petro)
x_test <- xgb.DMatrix(model.matrix(~ . - 1 , data=test[,x_vars]))

## 6.2. la perilla del boosting son las rondas (B), y aqui B si sobreajusta: se
## los 10 pliegues del curso (semilla 2026, los mismos de la semana 3)
set.seed(2026)
pliegue <- sample(rep(1:10 , length.out=nrow(train)))

set.seed(2026)
cv_boost <- xgb.cv(params=xgb.params(objective="reg:squarederror" , eta=0.05 , max_depth=6 ,
                                     subsample=0.8 , colsample_bytree=0.8 ,
                                     eval_metric=c("mae","rmse") , nthread=1) ,
                   data=x_train , nrounds=1500 , folds=split(1:nrow(train) , pliegue) ,
                   early_stopping_rounds=50 , verbose=0)

## 6.3. la curva del tuning: dentro del train el error solo baja; en validacion

## mejor iteraccion
cv_boost$early_stop$best_iteration

## todas las iteraciones
log_cv <- as.data.frame(cv_boost$evaluation_log)
nrow(log_cv)
head(log_cv)

## veamos lo que hay en cada iteraccion
log_cv[c(1,100,300,500,738) ,
       c("iter","train_rmse_mean","test_rmse_mean")]

## plot iteracciones
ggplot(log_cv , aes(x=iter)) +
geom_line(aes(y=train_rmse_mean , color="Train") , linewidth=1) +
geom_line(aes(y=test_rmse_mean , color="Test") , linewidth=1) +
scale_color_manual(values=c("Train"="gray60" , "Test"="red") ,breaks=c("Train","Test")) +
labs(x="rondas (B)" , y="rmse" , color=NULL) +
theme_classic()

## 6.4. estimar con la mejor ronda y predecir
set.seed(2026)
boost <- xgb.train(params=xgb.params(objective="reg:squarederror" , eta=0.05 , max_depth=6 ,
                                     subsample=0.8 , colsample_bytree=0.8 , nthread=1) ,
                   data=x_train , nrounds=738 , verbose=0)
train$pred_boost <- predict(boost , x_train)
test$pred_boost <- predict(boost , x_test)

## 6.5. rmse y mae
c(rmse_train = sqrt(mean((train$voto_petro - train$pred_boost)^2)) ,
  rmse_test = sqrt(mean((test$voto_petro - test$pred_boost)^2)) ,
  mae_test = mean(abs(test$voto_petro - test$pred_boost)))

##==: 7. la tabla de hoy :==##

## 7.1. target original vs las cuatro predicciones pegadas
head(test,5) %>%
select(nombre , voto_petro , pred_arbol_a , pred_arbol_b , pred_bosque , pred_boost)

## 7.2. rmse y mae de todo lo que corrimos, en el train y en el test
tabla <- rbind(media = c(sqrt(mean((train$voto_petro - mean(train$voto_petro))^2)) , sqrt(mean((test$voto_petro - mean(train$voto_petro))^2))) ,
               arbol_a = c(sqrt(mean((train$voto_petro - train$pred_arbol_a)^2)) , sqrt(mean((test$voto_petro - test$pred_arbol_a)^2))) ,
               arbol_b_cv = c(sqrt(mean((train$voto_petro - train$pred_arbol_b)^2)) , sqrt(mean((test$voto_petro - test$pred_arbol_b)^2))) ,
               arbol_sin_frenos = c(sqrt(mean((train$voto_petro - predict(arbol_sf , train))^2)) , sqrt(mean((test$voto_petro - predict(arbol_sf , test))^2))) ,
               bosque = c(sqrt(mean((train$voto_petro - train$pred_bosque)^2)) , sqrt(mean((test$voto_petro - test$pred_bosque)^2))) ,
               xgboost = c(sqrt(mean((train$voto_petro - train$pred_boost)^2)) , sqrt(mean((test$voto_petro - test$pred_boost)^2))))
colnames(tabla) <- c("rmse_train" , "rmse_test")
tabla

## 7.3. la misma tabla en un grafico: gris el train, rojo el test
barplot(t(tabla[,c("rmse_train","rmse_test")]) , beside=T , col=c("gray70","red") ,
        ylab="rmse (puntos del voto)" , cex.names=0.75)
legend("topright" , legend=c("train","test") , fill=c("gray70","red") , bty="n")

##==: 8. para la casa: el espacio de busqueda completo :==##

## 8.1. hasta aqui tuneamos una perilla a la vez. La busqueda de verdad recorre una
grilla_casa <- expand.grid(ntree = seq(500 , 1000 , by=50) , mtry = 5:20)
nrow(grilla_casa)

## 8.2. eso son 176 bosques: horas de computador, no de clase. En caret la grilla de
## mtry entra en tuneGrid (ntree se fija en el valor mas alto, porque no sobreajusta):
## descomentar y correr en la casa
# bosque_casa <- train(x=train[,x_vars] , y=train$voto_petro , method="rf" ,
#                      trControl=trainControl(method="oob") ,
#                      tuneGrid=expand.grid(mtry=5:20) , ntree=1000)

## 8.3. en xgboost la grilla cruza la tasa y la profundidad, y cada combinacion
## corre su xgb.cv como el de la seccion 6: eta en c(0.01, 0.05, 0.1, 0.3) y
## max_depth en c(2, 4, 6, 8), por ejemplo. Y el menu queda abierto: en caret,
## method="glmnet" es el lasso de la semana 3 y method="knn" los vecinos;
## combinar modelos tambien se vale
