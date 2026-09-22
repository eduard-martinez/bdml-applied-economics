## Big Data y Machine Learning para Economia Aplicada
## week-02: la regresion lineal como maquina de predecir
## Last run: Sep 22, 2026

##==: 0. initial setup :==##

## clean environment
rm(list=ls())

## load/install packages
require(pacman)
p_load(rio , dplyr)

##==: 1. prepare data :==##

## 1.1. load data (la base publica del curso, directo desde github)
url <- "https://raw.githubusercontent.com/eduard-martinez/bdml-applied-economics/main/applications/week-02/input/puestos_valle_2022.rds"
db <- import(url , trust=T)

## 1.2. subset data: cali y su area metropolitana
df <- db %>%
      subset(municipio %in% c("CALI","PALMIRA","YUMBO","JAMUNDI")) %>%
      mutate(log_registrados = log(registrados))
nrow(df)

## 1.3. split data (la particion de la clase: semilla 2026, 25% a la prueba)
set.seed(2026)
sample <- sample(x = nrow(df) , round(nrow(df)*0.25))
test <- df[sample,]
train <- df[-sample,]

##==: 2. baseline: la media del target :==##

## 2.1. target predicho: la media del train para todos
test$voto_pred <- mean(train$voto_petro)

## 2.2. comparar target original vs target predicho
test %>% select(nombre , municipio , voto_petro , voto_pred) %>% head(5)

## 2.3. rmse en train y test
rmse_train_media <- sqrt(mean((train$voto_petro - mean(train$voto_petro))^2))
rmse_test_media <- sqrt(mean((test$voto_petro - test$voto_pred)^2))
c(train = rmse_train_media , test = rmse_test_media)

##==: 3. regresion simple :==##

## 3.1. estimar el modelo: voto contra educacion superior
modelo_simple <- lm(voto_petro ~ educ_superior , data=train)
summary(modelo_simple)

## 3.2. la nube y la recta
plot(train$educ_superior , train$voto_petro , pch=16 , col="gray50",
     xlab="educacion superior (%)" , ylab="voto por petro (%)")
abline(modelo_simple , col="blue" , lwd=2)

## 3.3. target predicho vs target original
test$voto_pred <- predict(modelo_simple , test)
test %>% select(nombre , municipio , voto_petro , voto_pred) %>% head(5)

## 3.4. rmse en train y test
rmse_train_simple <- sqrt(mean((train$voto_petro - predict(modelo_simple , train))^2))
rmse_test_simple <- sqrt(mean((test$voto_petro - test$voto_pred)^2))
c(train = rmse_train_simple , test = rmse_test_simple)

##==: 4. k-nn: el promedio de los k puestos mas cercanos :==##

## 4.1. la funcion de la clase: promediar los k vecinos en el mapa
knn_reg <- function(coord_train , y_train , coord_nueva , k){
           pred <- rep(NA , nrow(coord_nueva))
           for (j in 1:nrow(coord_nueva)){
                dist <- sqrt((coord_train[,1] - coord_nueva[j,1])^2 + (coord_train[,2] - coord_nueva[j,2])^2)
                pred[j] <- mean(y_train[order(dist)[1:k]])
           }
           return(pred)
}

## 4.2. coordenadas
coord_train <- as.matrix(train[,c("lon","lat")])
coord_test <- as.matrix(test[,c("lon","lat")])

## 4.3. rmse en train y test para k = 1, 5 y 25
knn <- data.frame(k = c(1,5,25) , rmse_train = NA , rmse_test = NA)
for (i in 1:nrow(knn)){
     pred_train <- knn_reg(coord_train , train$voto_petro , coord_train , k=knn$k[i])
     pred_test <- knn_reg(coord_train , train$voto_petro , coord_test , k=knn$k[i])
     knn$rmse_train[i] <- sqrt(mean((train$voto_petro - pred_train)^2))
     knn$rmse_test[i] <- sqrt(mean((test$voto_petro - pred_test)^2))
}
knn

##==: 5. regresion multiple: todas las combinaciones :==##

## 5.1. covariables candidatas (las nueve de la clase)
covars <- c("educ_superior","estrato_bajo","estrato_alto","edad_20_29","edad_60_74",
            "mujeres","afro","servicios","log_registrados")

## 5.2. todas las combinaciones posibles (2^9 - 1 = 511 modelos)
combos <- list()
for (k in 1:length(covars)){
     combos <- c(combos , combn(x=covars , m=k , simplify=F))
}

## 5.3. grid de modelos
grid <- data.frame(id_modelo = 1:length(combos),
                   n_covars = sapply(combos , length),
                   covars = sapply(combos , paste , collapse=" + "),
                   rmse_train = NA,
                   rmse_test = NA)

## 5.4. correr los 511 modelos
for (i in 1:nrow(grid)){
     modelo_i <- lm(as.formula(paste0("voto_petro ~ " , grid$covars[i])) , data=train)
     grid$rmse_train[i] <- sqrt(mean((train$voto_petro - predict(modelo_i , train))^2))
     grid$rmse_test[i] <- sqrt(mean((test$voto_petro - predict(modelo_i , test))^2))
}

## 5.5. ranking por rmse de test (el train siempre mejora con mas covariables; el test no)
grid <- grid %>%
        arrange(rmse_test) %>%
        mutate(ranking = row_number())

## top 10 modelos
grid %>%
     mutate(rmse_train = round(rmse_train,2) , rmse_test = round(rmse_test,2)) %>%
     select(ranking , n_covars , covars , rmse_train , rmse_test) %>%
     head(10)

## best model
top_1 <- grid %>% filter(ranking==1)
top_1

##==: 6. el mejor modelo :==##

## 6.1. reestimar el mejor modelo
modelo_best <- lm(as.formula(paste0("voto_petro ~ " , top_1$covars[1])) , data=train)
summary(modelo_best)

## 6.2. target predicho vs target original
test$voto_pred <- predict(modelo_best , test)
test %>%
     select(nombre , municipio , voto_petro , voto_pred) %>%
     mutate(voto_petro = round(voto_petro,1) , voto_pred = round(voto_pred,1)) %>%
     head(5)

## 6.3. la diagonal es la prediccion perfecta
plot(test$voto_petro , test$voto_pred , pch=16 , col="blue" , xlim=c(10,100) , ylim=c(10,100),
     xlab="target original (%)" , ylab="target predicho (%)")
abline(a=0 , b=1 , lty=2)

##==: 7. tabla final :==##

## rmse en train y test de todos los modelos de la clase
tabla <- data.frame(modelo = c("media del train (baseline)",
                               "regresion simple (educacion superior)",
                               paste0("k-nn en el mapa (k = " , knn$k , ")"),
                               paste0("mejor multiple (" , top_1$n_covars , " covariables)")),
                    rmse_train = c(rmse_train_media , rmse_train_simple , knn$rmse_train , top_1$rmse_train),
                    rmse_test = c(rmse_test_media , rmse_test_simple , knn$rmse_test , top_1$rmse_test))
tabla %>%
     mutate(rmse_train = round(rmse_train,2) , rmse_test = round(rmse_test,2)) %>%
     arrange(rmse_test)
