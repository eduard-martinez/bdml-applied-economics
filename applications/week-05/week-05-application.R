## Big Data y Machine Learning para Economia Aplicada
## week-05-application: la aplicacion de clase. Cuatro modelos con las 37 variables del
## censo: un arbol sencillo, un arbol muy profundo que se aprende el train de memoria,
## un bosque aleatorio y un xgboost. La replica completa de las diapositivas es
## week-05-replication.R
## Last run: Oct 13, 2026

##==: 0. initial setup :==##

## clean environment
rm(list=ls())

## load/install packages
require(pacman)
p_load(rio , dplyr , rpart , randomForest , xgboost)

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

## 1.3. los nombres de las 37 variables
x_censo <- setdiff(names(censo37) , "puesto")
length(x_censo)

## 1.4. la formula que usan los arboles: voto_petro contra las 37
f_censo <- as.formula(paste0("voto_petro ~ " , paste(x_censo , collapse=" + ")))

## 1.5. split data (la particion fija del curso: 9.602 para aprender y 2.398 para probar)
train <- db %>% subset(muestra=="entrenamiento")
test <- db %>% subset(muestra=="prueba")
c(train = nrow(train) , test = nrow(test))

## 1.6. la referencia: predecir todos los puestos con el promedio del train
## mira: 27,3 en el train y 26,7 en el test. Todo lo que sigue se compara con esto
rmse_train_media <- sqrt(mean((train$voto_petro - mean(train$voto_petro))^2))
rmse_test_media <- sqrt(mean((test$voto_petro - mean(train$voto_petro))^2))
c(train = rmse_train_media , test = rmse_test_media)

##==: 2. un arbol sencillo :==##

## 2.1. estimar el arbol con las opciones por defecto de rpart: solo acepta un corte
## si mejora el ajuste al menos 1% (cp = 0.01)
arbol <- rpart(f_censo , data=train)

## 2.2. el arbol en texto: cada fila es un nodo; las que terminan en * son hojas
## mira: el primer corte es la poblacion afro en 4,6% (la lamina 9)
arbol

## 2.3. el arbol dibujado: cada nodo es una pregunta de si o no; cada hoja, un promedio
plot(arbol , uniform=T , margin=0.05)
text(arbol , use.n=T , cex=0.7)

## 2.4. cuantas hojas tiene
## mira: 10 hojas
sum(arbol$frame$var=="<leaf>")

## 2.5. predecir: a cada puesto le toca el promedio de su hoja
train$pred_arbol <- predict(arbol , train)
test$pred_arbol <- predict(arbol , test)

## 2.6. target original vs target predicho
test %>% select(nombre , departamento , afro , indigena , voto_petro , pred_arbol) %>% head(5)

## 2.7. el rmse en el train y en el test
## mira: 21,2 y 21,4. Casi iguales: este arbol no se aprendio nada de memoria
rmse_train_arbol <- sqrt(mean((train$voto_petro - train$pred_arbol)^2))
rmse_test_arbol <- sqrt(mean((test$voto_petro - test$pred_arbol)^2))
c(train = rmse_train_arbol , test = rmse_test_arbol)

##==: 3. un arbol muy profundo: se aprende el train de memoria :==##

## 3.1. el mismo arbol sin frenos: acepta cualquier corte (cp = 0), parte grupos de
## solo 2 puestos (minsplit = 2) y puede bajar 30 niveles (maxdepth = 30)
arbol_grande <- rpart(f_censo , data=train ,
                      control=rpart.control(cp=0 , minsplit=2 , minbucket=1 , maxdepth=30))

## 3.2. cuantas hojas tiene
## mira: 9.402 hojas para 9.602 puestos: casi una hoja por puesto
sum(arbol_grande$frame$var=="<leaf>")

## 3.3. predecir
train$pred_grande <- predict(arbol_grande , train)
test$pred_grande <- predict(arbol_grande , test)

## 3.4. el rmse en el train y en el test
## mira: en el train baja de 21,2 a 0,4: casi perfecto. En el test sube de 21,4 a
## 22,7: empeora. Aprenderse el train de memoria no sirve para puestos nuevos
rmse_train_grande <- sqrt(mean((train$voto_petro - train$pred_grande)^2))
rmse_test_grande <- sqrt(mean((test$voto_petro - test$pred_grande)^2))
c(train = rmse_train_grande , test = rmse_test_grande)

##==: 4. bosque aleatorio :==##

## 4.1. en vez de escoger un arbol, se promedian 300 arboles grandes. Cada arbol se
## estima con una muestra distinta del train y en cada corte solo compiten 12 de las
## 37 variables, sorteadas (mtry = 12). Tarda cerca de un minuto
set.seed(2026)
bosque <- randomForest(x=train[,x_censo] , y=train$voto_petro , ntree=300 , mtry=12)

## 4.2. predecir: el promedio de lo que dicen los 300 arboles
train$pred_bosque <- predict(bosque , train[,x_censo])
test$pred_bosque <- predict(bosque , test[,x_censo])

## 4.3. el rmse en el train y en el test
## mira: 15,6 en el test, contra 21,4 del arbol sencillo y 22,7 del arbol profundo
rmse_train_bosque <- sqrt(mean((train$voto_petro - train$pred_bosque)^2))
rmse_test_bosque <- sqrt(mean((test$voto_petro - test$pred_bosque)^2))
c(train = rmse_train_bosque , test = rmse_test_bosque)

##==: 5. xgboost :==##

## 5.1. xgboost no recibe un data.frame: hay que pasarle matrices de numeros
x_train <- xgb.DMatrix(as.matrix(train[,x_censo]) , label=train$voto_petro)
x_test <- xgb.DMatrix(as.matrix(test[,x_censo]))

## 5.2. las tres decisiones del boosting: cuanto se suma de cada arbol (eta = 0.05),
## que tan grande es cada arbol (max_depth = 6) y cuantas rondas (nrounds = 1167).
## Ademas, cada arbol ve el 80% de los puestos y el 80% de las variables
parametros <- xgb.params(objective="reg:squarederror" , eta=0.05 , max_depth=6 ,
                         subsample=0.8 , colsample_bytree=0.8 , nthread=1)

## 5.3. estimar: 1.167 arboles pequenos en fila, cada uno corrige lo que dejaron los
## anteriores (las 1.167 rondas las eligio la validacion cruzada de la replica)
set.seed(2026)
boost <- xgb.train(params=parametros , data=x_train , nrounds=1167 , verbose=0)

## 5.4. predecir: el promedio del train mas la suma de los 1.167 arboles
train$pred_boost <- predict(boost , x_train)
test$pred_boost <- predict(boost , x_test)

## 5.5. el rmse en el train y en el test
## mira: 15,7 en el test: empata con el bosque
rmse_train_boost <- sqrt(mean((train$voto_petro - train$pred_boost)^2))
rmse_test_boost <- sqrt(mean((test$voto_petro - test$pred_boost)^2))
c(train = rmse_train_boost , test = rmse_test_boost)

##==: 6. la tabla de hoy :==##

## 6.1. target original vs target predicho por los cuatro modelos
test %>%
     select(nombre , voto_petro , pred_arbol , pred_grande , pred_bosque , pred_boost) %>%
     mutate(across(where(is.numeric) , ~ round(.x,1))) %>%
     head(5)

## 6.2. el rmse de cada modelo en el train y en el test
## mira: el arbol profundo es el mejor en el train y el peor en el test; el bosque
## y el xgboost son los unicos que bajan de 16 en el test
tabla <- data.frame(modelo = c("promedio" , "arbol sencillo" , "arbol profundo" , "bosque" , "xgboost") ,
                    rmse_train = c(rmse_train_media , rmse_train_arbol , rmse_train_grande ,
                                   rmse_train_bosque , rmse_train_boost) ,
                    rmse_test = c(rmse_test_media , rmse_test_arbol , rmse_test_grande ,
                                  rmse_test_bosque , rmse_test_boost))
tabla %>% mutate(rmse_train = round(rmse_train,1) , rmse_test = round(rmse_test,1))

## 6.3. la misma tabla en un grafico: gris el train, rojo el test
barplot(t(as.matrix(tabla[,c("rmse_train","rmse_test")])) , beside=T , names.arg=tabla$modelo ,
        col=c("gray70","red") , ylab="rmse (puntos del voto)" , cex.names=0.8)
legend("topright" , legend=c("train","test") , fill=c("gray70","red") , bty="n")
