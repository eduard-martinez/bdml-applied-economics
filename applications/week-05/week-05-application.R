## Big Data y Machine Learning para Economia Aplicada
## week-05-application: la aplicacion de clase. Un arbol que se puede leer, el bosque
## que promedia y el boosting que corrige, contra el Lasso de la semana 3. La replica
## completa de las diapositivas es week-05-replication.R
## Last run: Oct 8, 2026

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

## 1.2. la base de hoy: el voto, las 37 del censo y el departamento como factor
## (el arbol usa la variable de texto entera, sin dummies)
db <- puestos %>%
      select(puesto , nombre , municipio , departamento , voto_petro , muestra) %>%
      inner_join(censo37 , by="puesto") %>%
      mutate(dpto = factor(departamento)) %>%
      subset(!is.na(educ_superior) & !is.na(afro))
x_censo <- setdiff(names(censo37) , "puesto")

## 1.3. split data (la particion fija del curso: 9.602 para aprender y 2.398 bajo llave)
test <- db %>% subset(muestra=="prueba")
train <- db %>% subset(muestra=="entrenamiento")
c(train = nrow(train) , test = nrow(test))

## 1.4. la referencia de siempre: la media del train deja 26,7 en la prueba
sqrt(mean((test$voto_petro - mean(train$voto_petro))^2))

##==: 2. el primer corte del pais :==##

## 2.1. la formula con las 37 del censo, y cuantos cortes candidatos hay en la raiz
## (cada variable aporta tantos cortes como valores distintos tenga, menos uno)
f_censo <- as.formula(paste0("voto_petro ~ " , paste(x_censo , collapse=" + ")))
sum(sapply(train[,x_censo] , function(x) length(unique(x)) - 1))

## 2.2. un arbol de un solo corte: la maquina prueba los 208.236 y se queda con el mejor
## mira: corta la poblacion afro en 4,6%. A la izquierda quedan 7.508 puestos con voto
## medio 47,3; a la derecha, 2.094 con 71,2. Nadie le dijo donde esta el pacifico
arbol_1 <- rpart(f_censo , data=train , control=rpart.control(maxdepth=1 , cp=0 , xval=0))
arbol_1

##==: 3. el arbol que se puede leer: seis hojas :==##

## 3.1. crecer el arbol grande (hojas chicas, cp = 0) y podarlo a seis hojas, es decir
## cinco cortes: el arbol de la lamina 9 (tarda unos segundos)
arbol_g <- rpart(f_censo , data=train , control=rpart.control(cp=0 , minsplit=20 , xval=0))
arbol_6 <- prune(arbol_g , cp=arbol_g$cptable[arbol_g$cptable[,"nsplit"]==5,"CP"])
arbol_6

## 3.2. el dibujo: cada nodo es una pregunta de si o no; cada hoja, un promedio
## mira: la indigena corta dos veces (una no linealidad) y el estrato 2 y el gas solo
## aparecen en la rama con poca poblacion afro: interacciones que nadie escribio
plot(arbol_6 , uniform=T , margin=0.05)
text(arbol_6 , use.n=T , cex=0.7)

## 3.3. la prediccion es el promedio de la hoja: siga un puesto por el dibujo
## con sus valores de afro e indigena, y llegue a su voto_pred
test$voto_pred <- predict(arbol_6 , test)
test %>%
     select(nombre , departamento , afro , indigena , voto_petro , voto_pred) %>%
     mutate(across(where(is.numeric) , ~ round(.x,1))) %>%
     head(5)

## el error de prueba del arbol de seis hojas
sqrt(mean((test$voto_petro - test$voto_pred)^2))

##==: 4. arboles malos, promedio bueno: el bosque :==##

## 4.1. un arbol solo es inestable: entre remuestras del entrenamiento, la prediccion
## de un puesto se mueve 9,2 puntos (lamina 13). La salida del bosque: promediar 500
## arboles hondos, cada uno sobre una remuestra bootstrap y con m = 12 variables
## candidatas por corte, para que no salgan todos iguales

## 4.2. el bosque con las 37 del censo + el departamento (tarda un minuto largo)
set.seed(2026)
bosque <- randomForest(x=train[,c(x_censo,"dpto")] , y=train$voto_petro , ntree=500 , mtry=12 ,
                       xtest=test[,c(x_censo,"dpto")] , ytest=test$voto_petro)

## 4.3. la curva del promedio: el error de prueba segun cuantos arboles se promedian
## mira: un arbol solo deja 19,8; con diez el promedio ya va en 13,6; los 500 dejan
## 12,9. Mas arboles nunca empeoran el promedio
curva <- sqrt(bosque$test$mse)
round(curva[c(1,10,50,100,500)] , 1)
plot(1:500 , curva , type="l" , log="x" , col="red" , lwd=2 ,
     xlab="numero de arboles promediados" , ylab="rmse de prueba")

## 4.4. el juez gratis: cada arbol deja fuera de su remuestra a un tercio de los
## puestos, y ese error fuera de la bolsa casi calca el de la prueba sin gastar pliegues
c(rmse_oob = sqrt(tail(bosque$mse,1)) , rmse_test = sqrt(tail(bosque$test$mse,1)))

## target predicho vs target original
test$voto_pred <- bosque$test$predicted
test %>% select(nombre , departamento , voto_petro , voto_pred) %>% head(5)

##==: 5. el boosting a mano: aprender despacio en cali :==##

## 5.1. los 257 puestos de cali y su area metropolitana (la particion de la semana 2)
metro <- db %>% subset(departamento=="VALLE" & municipio %in% c("CALI","PALMIRA","YUMBO","JAMUNDI"))
set.seed(2026)
prueba_m <- sample(x = nrow(metro) , round(nrow(metro)*0.25))
test_m <- metro[prueba_m,]
train_m <- metro[-prueba_m,]
c(train = nrow(train_m) , test = nrow(test_m))

## 5.2. el algoritmo 8.2 de isl con tocones (arboles de un solo corte) y tasa 0,1:
## arrancar en la media, ajustar un tocon al residuo, sumarlo encogido y repetir
lambda <- 0.1
grilla <- data.frame(educ_superior = seq(0 , 60 , by=0.5))
f_train <- rep(mean(train_m$voto_petro) , nrow(train_m))
f_test <- rep(mean(train_m$voto_petro) , nrow(test_m))
f_grilla <- rep(mean(train_m$voto_petro) , nrow(grilla))
aprende <- data.frame(b = 1:500 , rmse_train = NA , rmse_test = NA)
curvas <- list()
for (b in 1:500){
     residuo <- train_m$voto_petro - f_train
     tocon <- rpart(residuo ~ educ_superior , data=data.frame(residuo = residuo , educ_superior = train_m$educ_superior) ,
                    control=rpart.control(maxdepth=1 , cp=0 , minsplit=10 , minbucket=5 , xval=0))
     f_train <- f_train + lambda*predict(tocon)
     f_test <- f_test + lambda*predict(tocon , test_m)
     f_grilla <- f_grilla + lambda*predict(tocon , grilla)
     aprende$rmse_train[b] <- sqrt(mean((train_m$voto_petro - f_train)^2))
     aprende$rmse_test[b] <- sqrt(mean((test_m$voto_petro - f_test)^2))
     if (b %in% c(1,10,50,500)) curvas[[as.character(b)]] <- f_grilla
}

## 5.3. dentro de muestra el error solo baja; en la prueba toca fondo y vuelve a subir
## mira: el fondo llega en la ronda 48 (7,1) y en la 500 ya va en 7,3: a diferencia
## del bosque, aqui B si sobreajusta (por eso la parada temprana)
aprende[c(1,10,50,100,500),] %>% mutate(rmse_train = round(rmse_train,2) , rmse_test = round(rmse_test,2))
aprende[which.min(aprende$rmse_test),]

## 5.4. las escaleras: con 1 tocon casi la media; con 50, la forma; con 500, una
## escalera fina que persigue los puntos
plot(train_m$educ_superior , train_m$voto_petro , pch=16 , col="gray60" ,
     xlab="educacion superior (%)" , ylab="voto por petro (%)")
colores <- c("gray40" , "lightblue" , "blue" , "red")
for (i in 1:4){
     lines(grilla$educ_superior , curvas[[i]] , type="s" , col=colores[i] , lwd=2)
}
legend("topright" , legend=paste(names(curvas) , "tocones") , col=colores , lwd=2 , bty="n")

##==: 6. la carrera: ¿encontro la maquina las interacciones? :==##

## 6.1. el boosting del pais, con los mismos ingredientes que el lasso de la semana 3
## (censo + departamento; xgboost necesita numeros: model.matrix crea las 33 dummies)
x_train <- model.matrix(~ . - 1 , data=train[,c(x_censo,"dpto")])
x_test <- model.matrix(~ . - 1 , data=test[,c(x_censo,"dpto")])
dim(x_train)

## 6.2. las perillas de la lamina 23: tasa 0,05, profundidad 6, 80% de filas y de
## columnas por arbol. Las 738 rondas las eligio el juez con parada temprana en
## week-05-replication.R: aqui no esperamos esa cv (tarda un minuto largo)
parametros <- xgb.params(objective="reg:squarederror" , eta=0.05 , max_depth=6 ,
                         subsample=0.8 , colsample_bytree=0.8 , eval_metric="rmse" , nthread=1)
set.seed(2026)
boost <- xgb.train(params=parametros , data=xgb.DMatrix(x_train , label=train$voto_petro) ,
                   nrounds=738 , verbose=0)

## 6.3. target predicho vs target original, y el error de prueba
test$voto_pred <- predict(boost , xgb.DMatrix(x_test))
test %>% select(nombre , departamento , voto_petro , voto_pred) %>% head(5)
sqrt(mean((test$voto_petro - test$voto_pred)^2))

## 6.4. tabla final: el rmse de prueba de lo que corrimos hoy
## mira: con los mismos ingredientes, el boosting (12,8) le gana por 2,8 a la mejor
## recta de la semana 3 (censo + dummies, 15,6): los cruces que encontro la maquina
## valen lo que las dummies no alcanzan a contar. El promedio del boosting con el
## k-nn del mapa deja 11,4: lo corre la replica
tabla <- data.frame(modelo = c("la media del train (referencia)" ,
                               "un arbol de seis hojas (censo)" ,
                               "bosque (censo + departamento)" ,
                               "boosting (censo + departamento)") ,
                    rmse_test = c(sqrt(mean((test$voto_petro - mean(train$voto_petro))^2)) ,
                                  sqrt(mean((test$voto_petro - predict(arbol_6 , test))^2)) ,
                                  sqrt(tail(bosque$test$mse,1)) ,
                                  sqrt(mean((test$voto_petro - test$voto_pred)^2))))
tabla %>% mutate(rmse_test = round(rmse_test,1))
