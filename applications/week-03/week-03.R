## Big Data y Machine Learning para Economia Aplicada
## week-03: validacion cruzada y regularizacion (el juez y la perilla)
## Last run: Sep 29, 2026

##==: 0. initial setup :==##

## clean environment
rm(list=ls())

## load/install packages
require(pacman)
p_load(rio , dplyr , glmnet)

##==: 1. prepare data :==##

## 1.1. load data (la base publica del curso: la carpeta input/ del zip o github)
url <- "https://raw.githubusercontent.com/eduard-martinez/bdml-applied-economics/main/applications/week-03/input/"
puestos <- import("input/puestos_colombia_2022.rds" , trust=T)
censo37 <- import("input/censo37_puestos_2022.rds" , trust=T)

## 1.2. la base de hoy: el puesto, el target, las variables de la semana 2 y las 37
## variables del censo (vienen listas en censo37: proporciones + log_registrados + rural)
db <- puestos %>%
      select(puesto , nombre , departamento , municipio , cod_municipio , zona , lon , lat , registrados ,
             voto_petro , estrato_bajo , estrato_alto , servicios , muestra) %>%
      inner_join(censo37 , by="puesto") %>%
      subset(!is.na(educ_superior) & !is.na(afro))
nrow(db)

## los nombres de las 37 (los predictores de la perilla en la parte B)
x_censo <- setdiff(names(censo37) , "puesto")
length(x_censo)

## 1.3. subset data: cali y su area metropolitana (los mismos 343 puestos de la semana 2)
df <- db %>% subset(departamento=="VALLE" & municipio %in% c("CALI","PALMIRA","YUMBO","JAMUNDI"))
nrow(df)

## 1.4. split data (la particion de la semana 2: semilla 2026, 25% a la prueba)
set.seed(2026)

## sample
sample <- sample(x = nrow(df) , round(nrow(df)*0.25))

## subset test
test <- df[sample,]

## subset train
train <- df[-sample,]

## validar
c(train = nrow(train) , test = nrow(test))

##============================================================================##
##  PARTE A. EL JUEZ: estimar el error sin gastar la prueba                   ##
##============================================================================##

##==: 2. el problema: el concurso de los 511 con otras particiones :==##

## 2.1. las nueve covariables de la semana 2 y sus 511 combinaciones (2^9 - 1 modelos)
covars <- c("educ_superior","estrato_bajo","estrato_alto","edad_20_29","edad_60_74",
            "mujeres","afro","servicios","log_registrados")
combos <- list()
for (k in 1:length(covars)){
     combos <- c(combos , combn(x=covars , m=k , simplify=F))
}
grid <- data.frame(id_modelo = 1:length(combos),
                   n_covars = sapply(combos , length),
                   covars = sapply(combos , paste , collapse=" + "),
                   rmse_test = NA)

## 2.2. el concurso de la semana 2: cada modelo se estima en el train y se elige con el test
for (i in 1:nrow(grid)){
     modelo_i <- lm(as.formula(paste0("voto_petro ~ " , grid$covars[i])) , data=train)
     grid$rmse_test[i] <- sqrt(mean((test$voto_petro - predict(modelo_i , test))^2))
}
campeon_clase <- grid$covars[which.min(grid$rmse_test)]

## top 3: el campeon se eligio con la misma prueba que lo califica (la regla de oro #2 quedo rota)
grid %>% arrange(rmse_test) %>% select(n_covars , covars , rmse_test) %>% head(3)

## 2.3. el mismo concurso con 100 particiones 75/25 distintas (tarda cerca de medio minuto)
concurso <- data.frame(semilla = 1:100 , campeon = NA , n_covars = NA , rmse_campeon = NA , rmse_recta = NA)
for (s in 1:nrow(concurso)){
     set.seed(concurso$semilla[s])
     sample_s <- sample(x = nrow(df) , round(nrow(df)*0.25))
     test_s <- df[sample_s,]
     train_s <- df[-sample_s,]
     rmse_s <- rep(NA , nrow(grid))
     for (i in 1:nrow(grid)){
          modelo_i <- lm(as.formula(paste0("voto_petro ~ " , grid$covars[i])) , data=train_s)
          rmse_s[i] <- sqrt(mean((test_s$voto_petro - predict(modelo_i , test_s))^2))
     }
     concurso$campeon[s] <- grid$covars[which.min(rmse_s)]
     concurso$n_covars[s] <- grid$n_covars[which.min(rmse_s)]
     concurso$rmse_campeon[s] <- min(rmse_s)
     concurso$rmse_recta[s] <- rmse_s[1]
}

## cuantos campeones distintos hay, cuantas veces gana el de la clase y cuanto baila la recta
length(unique(concurso$campeon))
sum(concurso$campeon==campeon_clase)
table(concurso$n_covars)
range(concurso$rmse_recta)

## el campeon siempre queda debajo de la diagonal: es el minimo de 511 intentos (parte de su ventaja es suerte)
plot(concurso$rmse_recta , concurso$rmse_campeon , pch=16 , col="blue" ,
     xlab="rmse de prueba de la recta" , ylab="rmse de prueba del campeon")
abline(a=0 , b=1 , lty=2)

##==: 3. el juez (I): una particion de validacion es una loteria :==##

## 3.1. k-nn: la funcion de la semana 2, ahora devuelve una prediccion por cada k (una columna por k)
knn_reg <- function(coord_train , y_train , coord_nueva , k){
           pred <- matrix(NA , nrow(coord_nueva) , length(k))
           for (j in 1:nrow(coord_nueva)){
                dist <- (coord_train[,1] - coord_nueva[j,1])^2 + (coord_train[,2] - coord_nueva[j,2])^2
                vecinos <- y_train[order(dist)[1:max(k)]]
                pred[j,] <- cumsum(vecinos)[k]/k
           }
           return(pred)
}

## 3.2. coordenadas y la grilla de k (la perilla del k-nn)
coord_train <- as.matrix(train[,c("lon","lat")])
coord_test <- as.matrix(test[,c("lon","lat")])
ks <- 1:40

## 3.3. diez particiones de validacion del train: 193 puestos para ajustar y 64 para validar
loteria <- expand.grid(k = ks , semilla = 1:10 , rmse_valida = NA)
for (s in 1:10){
     set.seed(s)
     valida <- sample(x = nrow(train) , 64)
     pred <- knn_reg(coord_train[-valida,] , train$voto_petro[-valida] , coord_train[valida,] , ks)
     loteria$rmse_valida[loteria$semilla==s] <- sqrt(colMeans((train$voto_petro[valida] - pred)^2))
}

## cada semilla corona un k distinto, y el error en k = 10 depende de la particion que toco
loteria %>% group_by(semilla) %>% filter(rmse_valida==min(rmse_valida)) %>% ungroup()
loteria %>% subset(k==10) %>% summarise(minimo = min(rmse_valida) , maximo = max(rmse_valida))

## las diez curvas: el mismo train, otra particion, otro veredicto
plot(ks , loteria$rmse_valida[loteria$semilla==1] , type="l" , col="blue" , ylim=c(5,13) ,
     xlab="k: vecinos en el mapa" , ylab="rmse de validacion")
for (s in 2:10){
     lines(ks , loteria$rmse_valida[loteria$semilla==s] , col="blue")
}

##==: 4. el juez (II): validacion cruzada con J pliegues :==##

## 4.1. cinco pliegues, diez asignaciones distintas: cada puesto valida una sola vez
pliegues_5 <- expand.grid(k = ks , asignacion = 1:10 , rmse_cv = NA)
for (s in 1:10){
     set.seed(s)
     pliegue_s <- sample(rep(1:5 , length.out=nrow(train)))
     error2 <- matrix(NA , nrow(train) , length(ks))
     for (j in 1:5){
          pred <- knn_reg(coord_train[pliegue_s!=j,] , train$voto_petro[pliegue_s!=j] , coord_train[pliegue_s==j,] , ks)
          error2[pliegue_s==j,] <- (train$voto_petro[pliegue_s==j] - pred)^2
     }
     pliegues_5$rmse_cv[pliegues_5$asignacion==s] <- sqrt(colMeans(error2))
}

## el k elegido casi no se mueve y el rango en k = 10 se encoge (comparar con la loteria de la seccion 3)
pliegues_5 %>% group_by(asignacion) %>% filter(rmse_cv==min(rmse_cv)) %>% ungroup()
pliegues_5 %>% subset(k==10) %>% summarise(minimo = min(rmse_cv) , maximo = max(rmse_cv))

## 4.2. los pliegues de hoy: 10 pliegues con la semilla 2026 (los mismos para todos los modelos del metro)
set.seed(2026)
pliegue <- sample(rep(1:10 , length.out=nrow(train)))
table(pliegue)

## 4.3. cuantos pliegues: la recta con J = 2, 5 y 10 (50 asignaciones de cada uno)
cuantos <- expand.grid(asignacion = 1:50 , J = c(2,5,10) , rmse_cv = NA)
for (i in 1:nrow(cuantos)){
     set.seed(cuantos$asignacion[i])
     pliegue_i <- sample(rep(1:cuantos$J[i] , length.out=nrow(train)))
     error2 <- rep(NA , nrow(train))
     for (j in 1:cuantos$J[i]){
          modelo_j <- lm(voto_petro ~ educ_superior , data=train[pliegue_i!=j,])
          error2[pliegue_i==j] <- (train$voto_petro[pliegue_i==j] - predict(modelo_j , train[pliegue_i==j,]))^2
     }
     cuantos$rmse_cv[i] <- sqrt(mean(error2))
}

## con mas pliegues cada modelo se entrena con mas datos: el error estimado baja y se estabiliza
cuantos %>% group_by(J) %>% summarise(media = mean(rmse_cv) , sd = sd(rmse_cv))

## 4.4. loocv: el extremo J = n son 257 ajustes, uno por puesto...
modelo_simple <- lm(voto_petro ~ educ_superior , data=train)
loo <- rep(NA , nrow(train))
for (i in 1:nrow(train)){
     modelo_i <- lm(voto_petro ~ educ_superior , data=train[-i,])
     loo[i] <- train$voto_petro[i] - predict(modelo_i , train[i,])
}
sqrt(mean(loo^2))

## ...o un solo ajuste con el atajo del apalancamiento (isl, ec. 5.2)
sqrt(mean((residuals(modelo_simple)/(1 - hatvalues(modelo_simple)))^2))

##==: 5. el juez (III): el concurso de los 511 sin mirar la prueba :==##

## 5.1. los 511 modelos con los mismos 10 pliegues: el promedio de los 10 errores y su error estandar
grid$cv <- NA
grid$ee <- NA
for (i in 1:nrow(grid)){
     mse_i <- rep(NA , 10)
     for (j in 1:10){
          modelo_j <- lm(as.formula(paste0("voto_petro ~ " , grid$covars[i])) , data=train[pliegue!=j,])
          mse_i[j] <- mean((train$voto_petro[pliegue==j] - predict(modelo_j , train[pliegue==j,]))^2)
     }
     grid$cv[i] <- mean(mse_i)
     grid$ee[i] <- sd(mse_i)/sqrt(10)
}

## 5.2. ranking por cv: el juez corona otro campeon sin haber abierto la prueba
grid <- grid %>%
        arrange(cv) %>%
        mutate(ranking = row_number() , rmse_cv = sqrt(cv))
grid %>%
     mutate(rmse_cv = round(rmse_cv,2) , rmse_test = round(rmse_test,2)) %>%
     select(ranking , n_covars , covars , rmse_cv , rmse_test) %>%
     head(5)

## el campeon de la clase, juzgado sin mirar la prueba, cae al fondo de la tabla
grid %>% subset(covars==campeon_clase) %>% select(ranking , n_covars , rmse_cv , rmse_test)

## 5.3. la regla de un error estandar: entre lo que empata con el mejor, el modelo mas simple
top_1se <- grid %>%
           subset(cv <= grid$cv[1] + grid$ee[1]) %>%
           arrange(n_covars , cv) %>%
           head(1)
top_1se %>% select(ranking , n_covars , covars , rmse_cv , rmse_test)

## 5.4. que tan firme es el veredicto: otras 20 asignaciones de los pliegues (tarda cerca de un minuto)
robustez <- data.frame(asignacion = 1:20 , elegido = NA)
for (s in 1:20){
     set.seed(s)
     pliegue_s <- sample(rep(1:10 , length.out=nrow(train)))
     cv_s <- rep(NA , nrow(grid))
     ee_s <- rep(NA , nrow(grid))
     for (i in 1:nrow(grid)){
          mse_i <- rep(NA , 10)
          for (j in 1:10){
               modelo_j <- lm(as.formula(paste0("voto_petro ~ " , grid$covars[i])) , data=train[pliegue_s!=j,])
               mse_i[j] <- mean((train$voto_petro[pliegue_s==j] - predict(modelo_j , train[pliegue_s==j,]))^2)
          }
          cv_s[i] <- mean(mse_i)
          ee_s[i] <- sd(mse_i)/sqrt(10)
     }
     candidatos <- which(cv_s <= min(cv_s) + ee_s[which.min(cv_s)])
     robustez$elegido[s] <- grid$covars[candidatos[order(grid$n_covars[candidatos] , cv_s[candidatos])][1]]
}

## la recta gana en 19 de 20 asignaciones: el veredicto no depende de la suerte de los pliegues
table(robustez$elegido)

##============================================================================##
##  PARTE B. LA PERILLA: regularizar con las 37 variables del censo           ##
##============================================================================##

##==: 6. la perilla: ridge, lasso y elastic net con las 37 del censo :==##

## 6.1. ols con las 37: el mejor ajuste dentro de la muestra (una columna sobra y R la deja en NA)
f_censo <- as.formula(paste0("voto_petro ~ " , paste(x_censo , collapse=" + ")))
modelo_ols <- lm(f_censo , data=train)
sum(is.na(coef(modelo_ols)))
summary(modelo_ols)$r.squared

## el mismo ols juzgado con los 10 pliegues: el ajuste dentro no viaja
error2 <- rep(NA , nrow(train))
for (j in 1:10){
     modelo_j <- lm(f_censo , data=train[pliegue!=j,])
     error2[pliegue==j] <- (train$voto_petro[pliegue==j] - predict(modelo_j , train[pliegue==j,]))^2
}
rmse_cv_ols <- sqrt(mean(error2))
c(rmse_train = sqrt(mean(residuals(modelo_ols)^2)) , rmse_cv = rmse_cv_ols)

## 6.2. las matrices de glmnet (glmnet estandariza cada columna por dentro)
x_train <- as.matrix(train[,x_censo])
x_test <- as.matrix(test[,x_censo])
y_train <- train$voto_petro

## 6.3. ridge (alpha = 0): todas las variables encogen, ninguna sale (la grilla se alarga para ver el minimo)
ridge <- cv.glmnet(x=x_train , y=y_train , alpha=0 , foldid=pliegue , lambda.min.ratio=1e-6 , nlambda=150)

## el camino de los coeficientes y la curva del juez
plot(ridge$glmnet.fit , xvar="lambda")
plot(ridge)

## 6.4. lasso (alpha = 1): las variables entran de a una; las demas quedan en cero exacto
lasso <- cv.glmnet(x=x_train , y=y_train , alpha=1 , foldid=pliegue)
plot(lasso$glmnet.fit , xvar="lambda")
plot(lasso)

## los dos lambda del juez: el minimo y el de la regla de un error estandar
c(lambda_min = lasso$lambda.min , lambda_1se = lasso$lambda.1se)

## el orden de entrada al camino: que compra primero el lasso cuando el precio baja
betas <- as.matrix(lasso$glmnet.fit$beta)
entrada <- apply(betas!=0 , 1 , function(x) which(x)[1])
names(sort(entrada))[1:6]

## lo que sobrevive con la regla de un error estandar
coef_1se <- coef(lasso , s="lambda.1se")
round(coef_1se[coef_1se[,1]!=0 , , drop=F] , 3)

## 6.5. elastic net (alpha = 0,5): las dos penalizaciones a la vez
en <- cv.glmnet(x=x_train , y=y_train , alpha=0.5 , foldid=pliegue)

## 6.6. la tabla del metro: coeficientes, rmse dentro, de cv y de prueba (la prueba se abre una sola vez)
tabla_metro <- data.frame(modelo = c("recta (educacion superior)","ols con las 37 del censo","ridge (lambda min)",
                                     "lasso (lambda min)","lasso (lambda 1se)","elastic net (lambda min)"),
                          coeficientes = NA , rmse_train = NA , rmse_cv = NA , rmse_test = NA)

## la recta y ols
tabla_metro[1,2:5] <- c(1 , sqrt(mean(residuals(modelo_simple)^2)) , grid$rmse_cv[grid$covars=="educ_superior"] ,
                        sqrt(mean((test$voto_petro - predict(modelo_simple , test))^2)))
tabla_metro[2,2:5] <- c(sum(!is.na(coef(modelo_ols))) - 1 , sqrt(mean(residuals(modelo_ols)^2)) , rmse_cv_ols ,
                        sqrt(mean((test$voto_petro - predict(modelo_ols , test))^2)))

## las penalizadas, con el lambda que eligio el juez
penalizados <- list(ridge , lasso , lasso , en)
regla <- c("lambda.min","lambda.min","lambda.1se","lambda.min")
for (i in 1:4){
     modelo_i <- penalizados[[i]]
     indice <- which.min(abs(modelo_i$lambda - modelo_i[[regla[i]]]))
     tabla_metro$coeficientes[i+2] <- modelo_i$nzero[indice]
     tabla_metro$rmse_train[i+2] <- sqrt(mean((y_train - predict(modelo_i , x_train , s=regla[i]))^2))
     tabla_metro$rmse_cv[i+2] <- sqrt(modelo_i$cvm[indice])
     tabla_metro$rmse_test[i+2] <- sqrt(mean((test$voto_petro - predict(modelo_i , x_test , s=regla[i]))^2))
}
tabla_metro %>% mutate(rmse_train = round(rmse_train,2) , rmse_cv = round(rmse_cv,2) , rmse_test = round(rmse_test,2))

## target predicho vs target original (lasso con lambda min)
test$voto_pred <- as.numeric(predict(lasso , x_test , s="lambda.min"))
test %>% select(nombre , municipio , voto_petro , voto_pred) %>% head(5)

## 6.7. la gemela de la semana 2: la educacion superior mas un ruido minimo (semilla 99)
set.seed(99)
train$educ_gemela <- train$educ_superior + rnorm(nrow(train) , sd=0.5)
cor(train$educ_superior , train$educ_gemela)

## ols reparte al azar entre las gemelas, con errores estandar enormes
summary(lm(voto_petro ~ educ_superior + educ_gemela , data=train))$coefficients
summary(modelo_simple)$coefficients

## ridge reparte entre las dos, el lasso sortea una, elastic net las agrupa: los tres caminos
z_gemela <- scale(as.matrix(train[,c("educ_superior","educ_gemela")]))
par(mfrow=c(1,3))
plot(glmnet(z_gemela , y_train , alpha=0 , standardize=F) , xvar="lambda" , main="ridge")
plot(glmnet(z_gemela , y_train , alpha=1 , standardize=F) , xvar="lambda" , main="lasso")
plot(glmnet(z_gemela , y_train , alpha=0.5 , standardize=F) , xvar="lambda" , main="elastic net")
par(mfrow=c(1,1))

##============================================================================##
##  PARTE C. EL PAIS: censo x departamento, y el juez de la pregunta del uso  ##
##============================================================================##

##==: 7. el pais: censo x departamento, 1.291 columnas :==##

## 7.1. la particion fija del curso: 9.602 puestos para aprender, 2.398 bajo llave
train_p <- db %>% subset(muestra=="entrenamiento")
test_p <- db %>% subset(muestra=="prueba")
c(train = nrow(train_p) , test = nrow(test_p))

## validar: en el pais el target es casi plano (ganarle a la media ya es dificil)
hist(train_p$voto_petro)
c(pais = sd(train_p$voto_petro) , metro = sd(train$voto_petro))

## 7.2. la media y la recta de la semana 2, en el pais: la educacion superior no ordena el voto nacional
recta_p <- lm(voto_petro ~ educ_superior , data=train_p)
summary(recta_p)$r.squared
c(media = sqrt(mean((test_p$voto_petro - mean(train_p$voto_petro))^2)) ,
  recta = sqrt(mean((test_p$voto_petro - predict(recta_p , test_p))^2)))

## 7.3. una recta por departamento: la pendiente de la educacion superior cambia de signo
dptos <- sort(unique(db$departamento))
pendientes <- data.frame(departamento = dptos , n = NA , b = NA , ee = NA)
for (d in 1:length(dptos)){
     train_d <- train_p %>% subset(departamento==dptos[d])
     modelo_d <- lm(voto_petro ~ educ_superior , data=train_d)
     pendientes$n[d] <- nrow(train_d)
     pendientes$b[d] <- coef(modelo_d)[2]
     pendientes$ee[d] <- summary(modelo_d)$coefficients[2,2]
}
pendientes %>% arrange(desc(n)) %>% mutate(b = round(b,2) , ee = round(ee,2)) %>% head(8)

## el embudo: con pocos puestos la pendiente es ruido; con muchos, es precisa y distinta
plot(log(pendientes$n) , pendientes$b , pch=16 , col="blue" , ylim=c(-6,2) ,
     xlab="log de los puestos del departamento" , ylab="pendiente de la educacion superior")
abline(h=coef(recta_p)[2] , lty=2)

## 7.4. el censo x departamento: 37 efectos nacionales + 33 constantes + 37 x 33 desviaciones
## (una columna por departamento, sin categoria de referencia: cada uno se encoge hacia lo nacional)
nombres <- c(x_censo , paste0("dpto:" , dptos) , paste0(rep(x_censo , each=length(dptos)) , ":" , dptos))
x_larga <- matrix(0 , nrow(train_p) , length(nombres) , dimnames=list(NULL , nombres))
x_larga_test <- matrix(0 , nrow(test_p) , length(nombres) , dimnames=list(NULL , nombres))
x_larga[,x_censo] <- as.matrix(train_p[,x_censo])
x_larga_test[,x_censo] <- as.matrix(test_p[,x_censo])
for (d in dptos){
     x_larga[,paste0("dpto:",d)] <- as.numeric(train_p$departamento==d)
     x_larga_test[,paste0("dpto:",d)] <- as.numeric(test_p$departamento==d)
     for (v in x_censo){
          x_larga[,paste0(v,":",d)] <- train_p[[v]]*(train_p$departamento==d)
          x_larga_test[,paste0(v,":",d)] <- test_p[[v]]*(test_p$departamento==d)
     }
}
dim(x_larga)

## 7.5. los 10 pliegues del pais (semilla 2026): los mismos para todos los modelos
set.seed(2026)
pliegue_p <- sample(rep(1:10 , length.out=nrow(train_p)))

## 7.6. ols con las 1.291 columnas: dentro casi perfecto, fuera se desploma (tarda cerca de un minuto)
ols_larga <- lm.fit(cbind(1 , x_larga) , train_p$voto_petro)
b_larga <- ols_larga$coefficients
b_larga[is.na(b_larga)] <- 0
error2 <- rep(NA , nrow(train_p))
for (j in 1:10){
     b_j <- lm.fit(cbind(1 , x_larga[pliegue_p!=j,]) , train_p$voto_petro[pliegue_p!=j])$coefficients
     b_j[is.na(b_j)] <- 0
     error2[pliegue_p==j] <- (train_p$voto_petro[pliegue_p==j] - cbind(1 , x_larga[pliegue_p==j,]) %*% b_j)^2
}
rmse_cv_ols_larga <- sqrt(mean(error2))
c(rmse_train = sqrt(mean((train_p$voto_petro - cbind(1 , x_larga) %*% b_larga)^2)) ,
  rmse_cv = rmse_cv_ols_larga ,
  rmse_test = sqrt(mean((test_p$voto_petro - cbind(1 , x_larga_test) %*% b_larga)^2)))

## 7.7. ridge, lasso y elastic net con los mismos pliegues (cada juez tarda unos minutos)
ridge_p <- cv.glmnet(x=x_larga , y=train_p$voto_petro , alpha=0 , foldid=pliegue_p , lambda.min.ratio=1e-7 , nlambda=150)
lasso_p <- cv.glmnet(x=x_larga , y=train_p$voto_petro , alpha=1 , foldid=pliegue_p)
en_p <- cv.glmnet(x=x_larga , y=train_p$voto_petro , alpha=0.5 , foldid=pliegue_p)
plot(lasso_p)

## 7.8. la tabla del pais: la media, las lineales de siempre y las penalizadas
f_dpto <- update(f_censo , . ~ . + departamento)
modelo_censo_p <- lm(f_censo , data=train_p)
modelo_dpto_p <- lm(f_dpto , data=train_p)
tabla_pais <- data.frame(modelo = c("media del train (baseline)","recta (educacion superior)","las 37 del censo",
                                    "censo + dummies de departamento","ols, censo x departamento","ridge, censo x departamento",
                                    "lasso, censo x departamento","elastic net, censo x departamento"),
                         coeficientes = NA , rmse_train = NA , rmse_cv = NA , rmse_test = NA)

## las lineales, con los mismos 10 pliegues
formulas_p <- list(voto_petro ~ 1 , voto_petro ~ educ_superior , f_censo , f_dpto)
for (i in 1:4){
     modelo_i <- lm(formulas_p[[i]] , data=train_p)
     error2_i <- rep(NA , nrow(train_p))
     for (j in 1:10){
          modelo_j <- lm(formulas_p[[i]] , data=train_p[pliegue_p!=j,])
          error2_i[pliegue_p==j] <- (train_p$voto_petro[pliegue_p==j] - predict(modelo_j , train_p[pliegue_p==j,]))^2
     }
     tabla_pais$coeficientes[i] <- sum(!is.na(coef(modelo_i))) - 1
     tabla_pais$rmse_train[i] <- sqrt(mean(residuals(modelo_i)^2))
     tabla_pais$rmse_cv[i] <- sqrt(mean(error2_i))
     tabla_pais$rmse_test[i] <- sqrt(mean((test_p$voto_petro - predict(modelo_i , test_p))^2))
}

## la fila de ols con las 1.291 columnas (ya calculada en 7.6)
tabla_pais[5,2:5] <- c(sum(!is.na(ols_larga$coefficients)) - 1 , sqrt(mean((train_p$voto_petro - cbind(1 , x_larga) %*% b_larga)^2)) ,
                       rmse_cv_ols_larga , sqrt(mean((test_p$voto_petro - cbind(1 , x_larga_test) %*% b_larga)^2)))

## las penalizadas, con el lambda minimo de cada juez
penalizados_p <- list(ridge_p , lasso_p , en_p)
for (i in 1:3){
     modelo_i <- penalizados_p[[i]]
     indice <- which.min(abs(modelo_i$lambda - modelo_i$lambda.min))
     tabla_pais$coeficientes[i+5] <- modelo_i$nzero[indice]
     tabla_pais$rmse_train[i+5] <- sqrt(mean((train_p$voto_petro - predict(modelo_i , x_larga , s="lambda.min"))^2))
     tabla_pais$rmse_cv[i+5] <- sqrt(modelo_i$cvm[indice])
     tabla_pais$rmse_test[i+5] <- sqrt(mean((test_p$voto_petro - predict(modelo_i , x_larga_test , s="lambda.min"))^2))
}
tabla_pais %>% mutate(rmse_train = round(rmse_train,1) , rmse_cv = round(rmse_cv,1) , rmse_test = round(rmse_test,1))

## 7.9. el juez pliegue por pliegue: ridge acierta en la prueba, pero se dispara en algunos pliegues
por_pliegue <- data.frame(pliegue = 1:10 , ridge = NA , lasso = NA)
for (j in 1:10){
     ridge_j <- glmnet(x_larga[pliegue_p!=j,] , train_p$voto_petro[pliegue_p!=j] , alpha=0 , lambda=ridge_p$lambda.min)
     lasso_j <- glmnet(x_larga[pliegue_p!=j,] , train_p$voto_petro[pliegue_p!=j] , alpha=1 , lambda=lasso_p$lambda.min)
     por_pliegue$ridge[j] <- sqrt(mean((train_p$voto_petro[pliegue_p==j] - predict(ridge_j , x_larga[pliegue_p==j,]))^2))
     por_pliegue$lasso[j] <- sqrt(mean((train_p$voto_petro[pliegue_p==j] - predict(lasso_j , x_larga[pliegue_p==j,]))^2))
}
por_pliegue %>% arrange(desc(ridge))

## el peor pliegue de ridge: de donde sale su error (unos pocos puestos de san andres)
peor <- which.max(por_pliegue$ridge)
ridge_peor <- glmnet(x_larga[pliegue_p!=peor,] , train_p$voto_petro[pliegue_p!=peor] , alpha=0 , lambda=ridge_p$lambda.min)
train_p[pliegue_p==peor,] %>%
     mutate(error2 = (voto_petro - as.numeric(predict(ridge_peor , x_larga[pliegue_p==peor,])))^2) %>%
     group_by(departamento) %>%
     summarise(puestos = n() , parte_del_error = sum(error2)) %>%
     mutate(parte_del_error = round(parte_del_error/sum(parte_del_error)*100,1)) %>%
     arrange(desc(parte_del_error)) %>%
     head(3)

## 7.10. la ventaja del lasso sobre censo + dummies de departamento: 2.000 remuestras de la prueba
test_p$voto_pred <- as.numeric(predict(lasso_p , x_larga_test , s="lambda.min"))
pred_dpto <- predict(modelo_dpto_p , test_p)
set.seed(2026)
dif <- rep(NA , 2000)
for (b in 1:2000){
     i <- sample(nrow(test_p) , replace=T)
     dif[b] <- sqrt(mean((test_p$voto_petro[i] - test_p$voto_pred[i])^2)) - sqrt(mean((test_p$voto_petro[i] - pred_dpto[i])^2))
}

## el intervalo de la diferencia queda entero por debajo de cero: la ventaja no es suerte de la prueba
quantile(dif , c(0.025,0.975))

## target predicho vs target original, y la diagonal
test_p %>% select(nombre , departamento , voto_petro , voto_pred) %>% head(5)
plot(test_p$voto_petro , test_p$voto_pred , pch=16 , col="blue" , cex=0.5 , xlim=c(0,100) , ylim=c(0,100),
     xlab="target original (%)" , ylab="target predicho (%)")
abline(a=0 , b=1 , lty=2)

##==: 8. dos jueces, dos preguntas: pliegues al azar o municipios enteros :==##

## 8.1. el segundo juez: pliegues que sacan municipios enteros (10 grupos, semilla 2026)
municipios <- unique(train_p$cod_municipio)
set.seed(2026)
grupo <- sample(rep(1:10 , length.out=length(municipios)))
pliegue_m <- grupo[match(train_p$cod_municipio , municipios)]

## las lineales con matrices (un departamento que falte en el pliegue queda en cero)
x_media <- matrix(1 , nrow(train_p) , 1)
x_censo_p <- cbind(1 , as.matrix(train_p[,x_censo]))
x_dpto <- cbind(1 , as.matrix(train_p[,x_censo]) , model.matrix(~ departamento , train_p)[,-1])
matrices <- list(x_media , x_censo_p , x_dpto)
esquemas <- data.frame(modelo = c("media del train","las 37 del censo","censo + dummies de departamento","lasso, censo x departamento","k-nn en el mapa"),
                       al_azar = NA , por_municipio = NA)
for (i in 1:3){
     for (esquema in c("al_azar","por_municipio")){
          if (esquema=="al_azar") pl <- pliegue_p else pl <- pliegue_m
          error2 <- rep(NA , nrow(train_p))
          for (j in 1:10){
               b_j <- lm.fit(matrices[[i]][pl!=j,,drop=F] , train_p$voto_petro[pl!=j])$coefficients
               b_j[is.na(b_j)] <- 0
               error2[pl==j] <- (train_p$voto_petro[pl==j] - matrices[[i]][pl==j,,drop=F] %*% b_j)^2
          }
          esquemas[i,esquema] <- sqrt(mean(error2))
     }
}

## el lasso del censo x departamento, validado por municipio (tarda un par de minutos)
lasso_m <- cv.glmnet(x=x_larga , y=train_p$voto_petro , alpha=1 , foldid=pliegue_m)
esquemas[4,2:3] <- c(sqrt(min(lasso_p$cvm)) , sqrt(min(lasso_m$cvm)))

## k-nn en el mapa del pais: el juez al azar elige k, y el juez por municipio lo evalua en municipios nuevos
coord_p <- as.matrix(train_p[,c("lon","lat")])
ks_p <- c(1:20 , 25 , 30 , 40 , 50)
knn_p <- data.frame(k = ks_p , al_azar = NA , por_municipio = NA)
for (esquema in c("al_azar","por_municipio")){
     if (esquema=="al_azar") pl <- pliegue_p else pl <- pliegue_m
     error2 <- matrix(NA , nrow(train_p) , length(ks_p))
     for (j in 1:10){
          pred <- knn_reg(coord_p[pl!=j,] , train_p$voto_petro[pl!=j] , coord_p[pl==j,] , ks_p)
          error2[pl==j,] <- (train_p$voto_petro[pl==j] - pred)^2
     }
     knn_p[,esquema] <- sqrt(colMeans(error2))
}
k_pais <- knn_p$k[which.min(knn_p$al_azar)]
esquemas[5,2:3] <- c(min(knn_p$al_azar) , knn_p$por_municipio[knn_p$k==k_pais])

## al azar: completar el mapa de un municipio que ya conozco; por municipio: predecir uno nuevo
esquemas %>% mutate(aumento = round(por_municipio - al_azar,1) , al_azar = round(al_azar,1) , por_municipio = round(por_municipio,1))

##==: 9. tabla final :==##

## el k-nn del mapa en la prueba, con el k que eligio el juez
rmse_knn_p <- sqrt(mean((test_p$voto_petro - knn_reg(coord_p , train_p$voto_petro , as.matrix(test_p[,c("lon","lat")]) , k_pais)[,1])^2))

## rmse de prueba de todos los modelos del pais, ordenados (la prueba se abrio una sola vez por modelo)
tabla_pais %>%
     select(modelo , rmse_cv , rmse_test) %>%
     rbind(data.frame(modelo = paste0("k-nn en el mapa (k = " , k_pais , ")") ,
                      rmse_cv = min(knn_p$al_azar) , rmse_test = rmse_knn_p)) %>%
     mutate(rmse_cv = round(rmse_cv,1) , rmse_test = round(rmse_test,1)) %>%
     arrange(rmse_test)
