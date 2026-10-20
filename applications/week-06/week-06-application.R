## Big Data y Machine Learning para Economia Aplicada
## week-06-application: el modelo de la semana 5, abierto con permutacion, ablacion y shap

##==: 0. initial setup :==##

## clean environment
rm(list=ls())

## load/install packages
require(pacman)
p_load(rio , dplyr , xgboost)

##==: 1. prepare data :==##

## 1.1. load data (la base publica del curso: la carpeta input/ del zip o github)
url <- "https://raw.githubusercontent.com/eduard-martinez/bdml-applied-economics/main/applications/week-06/input/"
puestos <- import("input/puestos_colombia_2022.rds" , trust=T)
censo37 <- import("input/censo37_puestos_2022.rds" , trust=T)

## 1.2. la base: el voto por petro (%), las 37 del censo y el departamento
db <- puestos %>%
      select(puesto , nombre , departamento , municipio , voto_petro , muestra) %>%
      inner_join(censo37 , by="puesto") %>%
      mutate(dpto = factor(departamento)) %>%
      subset(!is.na(educ_superior) & !is.na(afro))
x_censo <- setdiff(names(censo37) , "puesto")

## 1.3. split data (la particion fija del curso: 9.602 para aprender y 2.398 para probar)
train <- db %>% subset(muestra=="entrenamiento")
test <- db %>% subset(muestra=="prueba")
c(train = nrow(train) , test = nrow(test))

## 1.4. para xgboost, el departamento en 33 columnas de ceros y unos: 70 columnas
x_train <- model.matrix(~ . - 1 , data=train[,c(x_censo,"dpto")])
x_test <- model.matrix(~ . - 1 , data=test[,c(x_censo,"dpto")])
col_dpto <- grep("^dpto" , colnames(x_train) , value=T)
dim(x_train)

##==: 2. el modelo de la semana 5, repetido completo :==##

## 2.1. los 10 pliegues del curso (semilla 2026, los mismos de la semana 5)
set.seed(2026)
pliegue <- sample(rep(1:10 , length.out=nrow(train)))

## 2.2. la validacion cruzada elige las rondas, igual que cerro la semana 5 (tarda unos 3 minutos)
set.seed(2026)
cv_boost <- xgb.cv(params=xgb.params(objective="reg:squarederror" , eta=0.05 , max_depth=6 ,
                                     subsample=0.8 , colsample_bytree=0.8 ,
                                     eval_metric=c("mae","rmse") , nthread=1) ,
                   data=xgb.DMatrix(x_train , label=train$voto_petro) , nrounds=1500 ,
                   folds=split(1:nrow(train) , pliegue) , early_stopping_rounds=50 , verbose=0)

## mejor iteracion
cv_boost$early_stop$best_iteration

## 2.3. estimar con la mejor ronda y predecir
parametros <- xgb.params(objective="reg:squarederror" , 
                         eta=0.05 , max_depth=6 ,
                         subsample=0.8 , colsample_bytree=0.8 , nthread=1)

## train model
set.seed(2026)
modelo <- xgb.train(params=parametros , data=xgb.DMatrix(x_train , label=train$voto_petro) ,
                    nrounds=738 , verbose=0)

## 2.4. predecir y medir
test$voto_pred <- predict(modelo , xgb.DMatrix(x_test))
sqrt(mean((test$voto_petro - test$voto_pred)^2))

## 2.5. el tamano del modelo: nadie lee esto, hoy lo abrimos sin leerlo
arboles <- xgb.model.dt.tree(model=modelo)
head(arboles)

## plot arboles
c(arboles = length(unique(arboles$Tree)) , 
  hojas = sum(arboles$Feature=="Leaf"))

##==: 3. importancia por ganancia: la que trae el modelo :==##

## 3.1. la ganancia por variable (las 33 columnas del departamento cuentan como una)
imp <- xgb.importance(model=modelo)
imp$variable <- ifelse(grepl("^dpto" , imp$Feature) , "departamento" , imp$Feature)
ganancia <- imp %>%
            group_by(variable) %>%
            summarise(ganancia = round(100*sum(Gain) , 1)) %>%
            arrange(desc(ganancia))

## 3.2. el top 10
head(ganancia , 10)

## plot top 10
par(mar=c(4,8,2,1))
barplot(rev(ganancia$ganancia[1:10]) , names.arg=rev(ganancia$variable[1:10]) , horiz=T ,
        las=1 , cex.names=0.8 , col="blue" , xlab="ganancia (% del total)")
par(mar=c(5,4,4,2))

##==: 4. importancia por permutacion: cuanto duele barajar :==##

## 4.1. barajar una columna rompe su relacion con el voto; primero el rmse sin barajar
rmse_modelo <- sqrt(mean((test$voto_petro - test$voto_pred)^2))

## 4.2. barajar la poblacion afro
set.seed(2026)
x_barajada <- x_test
x_barajada[,"afro"] <- sample(x_barajada[,"afro"])
sube_afro <- sqrt(mean((test$voto_petro - predict(modelo , xgb.DMatrix(x_barajada)))^2)) - rmse_modelo
sube_afro

## 4.3. barajar la educacion superior
x_barajada <- x_test
x_barajada[,"educ_superior"] <- sample(x_barajada[,"educ_superior"])
sube_educ <- sqrt(mean((test$voto_petro - predict(modelo , xgb.DMatrix(x_barajada)))^2)) - rmse_modelo
sube_educ

## 4.4. barajar el departamento entero: las 33 columnas juntas, con el mismo orden de filas
x_barajada <- x_test
orden <- sample(nrow(x_barajada))
x_barajada[,col_dpto] <- x_barajada[orden,col_dpto]
sube_dpto <- sqrt(mean((test$voto_petro - predict(modelo , xgb.DMatrix(x_barajada)))^2)) - rmse_modelo
sube_dpto

## 4.5. una sola barajada es ruidosa: el deck promedia diez (el loop esta en la replica)

##==: 5. un ejemplo de ablacion: quitar el departamento y reentrenar :==##

## 5.1. el modelo sin departamento ya lo corrimos en la semana 5: solo censo, 1.052 rondas de su cv
## mira: 15,7, el mismo del barplot final de la semana 5
set.seed(2026)
modelo_sin <- xgb.train(params=parametros ,
                        data=xgb.DMatrix(as.matrix(train[,x_censo]) , label=train$voto_petro) ,
                        nrounds=1052 , verbose=0)
pred_sin <- predict(modelo_sin , xgb.DMatrix(as.matrix(test[,x_censo])))
sqrt(mean((test$voto_petro - pred_sin)^2))

## 5.2. la ablacion del departamento: cuanto se pierde sin esa informacion
ablacion_dpto <- sqrt(mean((test$voto_petro - pred_sin)^2)) - rmse_modelo
ablacion_dpto

## 5.3. permutar vs ablacion: danar la entrada duele mas que reentrenar sin ella
c(permutar = sube_dpto , ablacion = ablacion_dpto)

##==: 6. valores shap: el reparto de cada prediccion :==##

## 6.1. shap: cada prediccion = una base mas una contribucion por variable
shap <- predict(modelo , xgb.DMatrix(x_test) , predcontrib=T)
colnames(shap) <- c(colnames(x_test) , "BIAS")

## 6.2. la base y la eficiencia: base + suma de contribuciones = la prediccion
shap[1,"BIAS"]
max(abs(rowSums(shap) - test$voto_pred))

## 6.3. el departamento como un bloque: sus 33 contribuciones sumadas
shap_variable <- cbind(shap[,x_censo] , departamento = rowSums(shap[,col_dpto]))

## 6.4. un puesto explicado: el Colegio Fundacion Compartir, en Cali
i <- which(test$nombre=="COLEGIO FUNDACION COMPARTIR" & test$municipio=="CALI")
c(voto_real = test$voto_petro[i] , voto_pred = test$voto_pred[i])
round(sort(shap_variable[i,] , decreasing=T)[1:6] , 1)

## 6.5. otro puesto de la misma ciudad: el Colegio Comfandi
i2 <- which(test$nombre=="COLEGIO COMFANDI" & test$municipio=="CALI")
c(voto_real = test$voto_petro[i2] , voto_pred = test$voto_pred[i2])
contribucion <- shap_variable[i2,]
round(contribucion[order(-abs(contribucion))][1:6] , 1)

## 6.6. de lo local a lo global: el promedio del valor absoluto en el test
importancia_shap <- sort(colMeans(abs(shap_variable)) , decreasing=T)
round(head(importancia_shap , 10) , 1)

## 6.7. la dependencia shap: el valor de la variable contra su contribucion
plot(test$afro , shap_variable[,"afro"] , pch=16 , col="gray50" , cex=0.6 ,
     xlab="poblacion afro del puesto (%)" , ylab="contribucion shap (puntos de voto)")
abline(h=0 , lty=2)

##==: 7. la tabla de hoy :==##

## 7.1. las tres importancias lado a lado, para las variables que medimos
tabla <- data.frame(variable = c("departamento" , "afro" , "educ_superior") ,
                    ganancia_pct = ganancia$ganancia[match(c("departamento","afro","educ_superior") , ganancia$variable)] ,
                    permutacion = c(sube_dpto , sube_afro , sube_educ) ,
                    shap_promedio = importancia_shap[c("departamento","afro","educ_superior")])
tabla %>% mutate(across(where(is.numeric) , ~ round(.x,1)))
