## Big Data y Machine Learning para Economia Aplicada
## week-03: validacion cruzada y regularizacion (el juez y la perilla)
## Last run: Sep 24, 2026

##==: 0. initial setup :==##

## clean environment
rm(list=ls())

## load/install packages
require(pacman)
p_load(rio , dplyr , glmnet , doParallel)

## cluster: glmnet valida los pliegues en paralelo (mismo resultado, menos tiempo)
cores <- parallel::detectCores()
cl <- makeCluster(max(1 , cores - 2))
registerDoParallel(cl)

##==: 1. prepare data :==##

## 1.1. load data (la base publica del curso: la carpeta input/ del zip o github)
url <- "https://raw.githubusercontent.com/eduard-martinez/bdml-applied-economics/main/applications/week-03/input/"
puestos <- import("input/puestos_colombia_2022.rds" , trust=T)
censo <- import("input/censo_puestos_2022.rds" , trust=T)

## 1.2. la ficha del censo: de conteos a proporciones (una categoria por bloque queda por fuera)
ficha <- censo %>%
         mutate(con_nivel = educ_ninguna + educ_basica + educ_tecnica + educ_superior,
                con_etnia = etnia_ninguna + indigena + rom + raizal + palenquero + afro) %>%
         transmute(puesto,
                   educ_ninguna = educ_ninguna/con_nivel*100 , educ_tecnica = educ_tecnica/con_nivel*100 ,
                   educ_superior = educ_superior/con_nivel*100 ,
                   edad_0_19 = edad_0_19/personas*100 , edad_20_29 = edad_20_29/personas*100 ,
                   edad_45_59 = edad_45_59/personas*100 , edad_60_74 = edad_60_74/personas*100 ,
                   edad_75 = edad_75/personas*100 ,
                   mujeres = mujeres/personas*100 , edad_votar = edad_votar/personas*100 ,
                   indigena = indigena/con_etnia*100 , rom = rom/con_etnia*100 , raizal = raizal/con_etnia*100 ,
                   palenquero = palenquero/con_etnia*100 , afro = afro/con_etnia*100 ,
                   serv_agua = serv_agua/viviendas*100 , serv_alcantarillado = serv_alcantarillado/viviendas*100 ,
                   serv_energia = serv_energia/viviendas*100 , serv_gas = serv_gas/viviendas*100 ,
                   serv_internet = serv_internet/viviendas*100 ,
                   estrato_0 = estrato_0/viviendas*100 , estrato_1 = estrato_1/viviendas*100 ,
                   estrato_2 = estrato_2/viviendas*100 , estrato_3 = estrato_3/viviendas*100 ,
                   estrato_4 = estrato_4/viviendas*100 , estrato_5 = estrato_5/viviendas*100 ,
                   estrato_6 = estrato_6/viviendas*100 ,
                   hogares_por_vivienda = hogares/viviendas , personas_por_hogar = personas/hogares ,
                   log_poblacion = log(personas) , log_viviendas = log(viviendas) , log_manzanas = log(manzanas) ,
                   manzanas_compartidas = manzanas_compartidas/manzanas*100 ,
                   log_distancia = log(1 + distancia_media) ,
                   desfase = pmin(pmax(desfase , -100) , 300))

## 1.3. la base: el puesto, el target, las variables de la semana 2 y la ficha
db <- puestos %>%
      select(puesto , nombre , departamento , municipio , cod_municipio , zona , lon , lat , registrados ,
             voto_petro , estrato_bajo , estrato_alto , servicios , muestra) %>%
      inner_join(ficha , by="puesto") %>%
      mutate(log_registrados = log(registrados) , rural = as.integer(zona=="rural")) %>%
      subset(!is.na(educ_superior) & !is.na(afro))
nrow(db)

## las 37 variables de la ficha
x_ficha <- c(setdiff(names(ficha) , "puesto") , "log_registrados" , "rural")
length(x_ficha)

## 1.4. cali y su area metropolitana (los mismos 343 puestos de la semana 2)
df <- db %>% subset(departamento=="VALLE" & municipio %in% c("CALI","PALMIRA","YUMBO","JAMUNDI"))
nrow(df)

## 1.5. split data (la particion de la semana 2: semilla 2026, 25% a la prueba)
set.seed(2026)

## sample
sample <- sample(x = nrow(df) , round(nrow(df)*0.25))

## subset test
test <- df[sample,]

## subset train
train <- df[-sample,]

## validar
c(train = nrow(train) , test = nrow(test))

##==: 2. el problema: el concurso de los 511 con otras particiones :==##

## 2.1. las nueve covariables de la semana 2 y sus 511 combinaciones
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

## 2.2. el concurso de la clase: el campeon se eligio con la misma prueba que lo califica
for (i in 1:nrow(grid)){
     modelo_i <- lm(as.formula(paste0("voto_petro ~ " , grid$covars[i])) , data=train)
     grid$rmse_test[i] <- sqrt(mean((test$voto_petro - predict(modelo_i , test))^2))
}
campeon_clase <- grid$covars[which.min(grid$rmse_test)]
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

## campeones distintos, cuantas veces gana el de la clase y cuanto cambia la misma recta
length(unique(concurso$campeon))
sum(concurso$campeon==campeon_clase)
table(concurso$n_covars)
range(concurso$rmse_recta)

## el campeon siempre queda debajo de la diagonal: es el minimo de 511 intentos
plot(concurso$rmse_recta , concurso$rmse_campeon , pch=16 , col="blue" ,
     xlab="rmse de prueba de la recta" , ylab="rmse de prueba del campeon")
abline(a=0 , b=1 , lty=2)

##==: 3. el juez (I): una particion de validacion es una loteria :==##

## 3.1. k-nn: la funcion de la semana 2, ahora para varios k a la vez (una columna por k)
knn_reg <- function(coord_train , y_train , coord_nueva , k){
           pred <- matrix(NA , nrow(coord_nueva) , length(k))
           for (j in 1:nrow(coord_nueva)){
                dist <- (coord_train[,1] - coord_nueva[j,1])^2 + (coord_train[,2] - coord_nueva[j,2])^2
                vecinos <- y_train[order(dist)[1:max(k)]]
                pred[j,] <- cumsum(vecinos)[k]/k
           }
           return(pred)
}

## 3.2. coordenadas y los k de la curva
coord_train <- as.matrix(train[,c("lon","lat")])
coord_test <- as.matrix(test[,c("lon","lat")])
ks <- 1:40

## 3.3. diez particiones del train: 193 puestos para ajustar y 64 para validar
loteria <- expand.grid(k = ks , semilla = 1:10 , rmse_valida = NA)
for (s in 1:10){
     set.seed(s)
     valida <- sample(x = nrow(train) , 64)
     pred <- knn_reg(coord_train[-valida,] , train$voto_petro[-valida] , coord_train[valida,] , ks)
     loteria$rmse_valida[loteria$semilla==s] <- sqrt(colMeans((train$voto_petro[valida] - pred)^2))
}

## el k que elige cada semilla, y el error en k = 10 segun la semilla
loteria %>% group_by(semilla) %>% filter(rmse_valida==min(rmse_valida)) %>% ungroup()
loteria %>% subset(k==10) %>% summarise(minimo = min(rmse_valida) , maximo = max(rmse_valida))

## las diez curvas
plot(ks , loteria$rmse_valida[loteria$semilla==1] , type="l" , col="blue" , ylim=c(5,13) ,
     xlab="k: vecinos en el mapa" , ylab="rmse de validacion")
for (s in 2:10){
     lines(ks , loteria$rmse_valida[loteria$semilla==s] , col="blue")
}

##==: 4. el juez (II): validacion cruzada con K pliegues :==##

## 4.1. cinco pliegues con diez asignaciones distintas: las curvas casi se superponen
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
pliegues_5 %>% group_by(asignacion) %>% filter(rmse_cv==min(rmse_cv)) %>% ungroup()
pliegues_5 %>% subset(k==10) %>% summarise(minimo = min(rmse_cv) , maximo = max(rmse_cv))

## 4.2. los pliegues de hoy: 10 pliegues con la semilla 2026 (los mismos para todo el metro)
set.seed(2026)
pliegue <- sample(rep(1:10 , length.out=nrow(train)))
table(pliegue)

## 4.3. la curva del juez: el error de cada pliegue, para cada k
mse_pliegue <- matrix(NA , 10 , length(ks))
for (j in 1:10){
     pred <- knn_reg(coord_train[pliegue!=j,] , train$voto_petro[pliegue!=j] , coord_train[pliegue==j,] , ks)
     mse_pliegue[j,] <- colMeans((train$voto_petro[pliegue==j] - pred)^2)
}
curva <- data.frame(k = ks,
                    cv = colMeans(mse_pliegue),
                    ee = apply(mse_pliegue , 2 , sd)/sqrt(10),
                    rmse_train = sqrt(colMeans((train$voto_petro - knn_reg(coord_train , train$voto_petro , coord_train , ks))^2)))

## 4.4. el minimo y la regla de un error estandar (el k mas grande a menos de un ee del minimo)
k_min <- curva$k[which.min(curva$cv)]
k_1se <- max(curva$k[curva$cv <= min(curva$cv) + curva$ee[which.min(curva$cv)]])
c(k_min = k_min , k_1se = k_1se)
sqrt(curva$cv[c(k_min , k_1se)])

## dentro de muestra (gris) siempre elige k = 1; la validacion cruzada (rojo), no
plot(ks , sqrt(curva$cv) , type="l" , lwd=2 , col="red" , ylim=c(0,13) ,
     xlab="k: vecinos en el mapa" , ylab="rmse")
lines(ks , curva$rmse_train , lwd=2 , col="gray50")
abline(v=c(k_min , k_1se) , lty=2 , col="blue")

## 4.5. solo entonces se abre la prueba, una vez: target predicho vs target original
test$voto_pred <- knn_reg(coord_train , train$voto_petro , coord_test , k_1se)[,1]
test %>% select(nombre , municipio , voto_petro , voto_pred) %>% head(5)
c(k_min = sqrt(mean((test$voto_petro - knn_reg(coord_train , train$voto_petro , coord_test , k_min)[,1])^2)) ,
  k_1se = sqrt(mean((test$voto_petro - test$voto_pred)^2)))

## 4.6. cuantos pliegues: la recta con K = 2, 5 y 10 (50 asignaciones de cada uno)
cuantos <- expand.grid(asignacion = 1:50 , K = c(2,5,10) , rmse_cv = NA)
for (i in 1:nrow(cuantos)){
     set.seed(cuantos$asignacion[i])
     pliegue_i <- sample(rep(1:cuantos$K[i] , length.out=nrow(train)))
     error2 <- rep(NA , nrow(train))
     for (j in 1:cuantos$K[i]){
          modelo_j <- lm(voto_petro ~ educ_superior , data=train[pliegue_i!=j,])
          error2[pliegue_i==j] <- (train$voto_petro[pliegue_i==j] - predict(modelo_j , train[pliegue_i==j,]))^2
     }
     cuantos$rmse_cv[i] <- sqrt(mean(error2))
}
cuantos %>% group_by(K) %>% summarise(media = mean(rmse_cv) , sd = sd(rmse_cv))

## 4.7. loocv: 257 ajustes...
modelo_simple <- lm(voto_petro ~ educ_superior , data=train)
loo <- rep(NA , nrow(train))
for (i in 1:nrow(train)){
     modelo_i <- lm(voto_petro ~ educ_superior , data=train[-i,])
     loo[i] <- train$voto_petro[i] - predict(modelo_i , train[i,])
}
sqrt(mean(loo^2))

## ...o uno solo con el atajo del apalancamiento (isl, ec. 5.2)
sqrt(mean((residuals(modelo_simple)/(1 - hatvalues(modelo_simple)))^2))

##==: 5. el juez (III): el concurso de los 511 sin mirar la prueba :==##

## 5.1. los 511 modelos con los mismos 10 pliegues: el error de cada pliegue
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

## 5.2. ranking por cv: el campeon del juez y el puesto del campeon de la clase
grid <- grid %>%
        arrange(cv) %>%
        mutate(ranking = row_number() , rmse_cv = sqrt(cv))
grid %>%
     mutate(rmse_cv = round(rmse_cv,2) , rmse_test = round(rmse_test,2)) %>%
     select(ranking , n_covars , covars , rmse_cv , rmse_test) %>%
     head(5)
grid %>% subset(covars==campeon_clase) %>% select(ranking , n_covars , rmse_cv , rmse_test)

## 5.3. la regla de un error estandar: el modelo mas simple a menos de un ee del mejor
top_1se <- grid %>%
           subset(cv <= grid$cv[1] + grid$ee[1]) %>%
           arrange(n_covars , cv) %>%
           head(1)
top_1se %>% select(ranking , n_covars , covars , rmse_cv , rmse_test)

## 5.4. con otras 20 asignaciones de los pliegues, que elige la regla (tarda cerca de un minuto)
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
table(robustez$elegido)

##==: 6. la perilla: ridge, lasso y elastic net con la ficha de 37 :==##

## 6.1. ols con las 37: el mejor ajuste dentro (una columna sobra: R la deja en NA)
f_ficha <- as.formula(paste0("voto_petro ~ " , paste(x_ficha , collapse=" + ")))
modelo_ols <- lm(f_ficha , data=train)
sum(is.na(coef(modelo_ols)))
summary(modelo_ols)$r.squared

## el mismo modelo con los 10 pliegues
error2 <- rep(NA , nrow(train))
for (j in 1:10){
     modelo_j <- lm(f_ficha , data=train[pliegue!=j,])
     error2[pliegue==j] <- (train$voto_petro[pliegue==j] - predict(modelo_j , train[pliegue==j,]))^2
}
rmse_cv_ols <- sqrt(mean(error2))
c(rmse_train = sqrt(mean(residuals(modelo_ols)^2)) , rmse_cv = rmse_cv_ols)

## 6.2. las matrices de glmnet (glmnet estandariza cada columna por dentro)
x_train <- as.matrix(train[,x_ficha])
x_test <- as.matrix(test[,x_ficha])
y_train <- train$voto_petro

## 6.3. ridge (alpha = 0): todas encogen, ninguna sale (la grilla se alarga para ver el minimo)
ridge <- cv.glmnet(x=x_train , y=y_train , alpha=0 , foldid=pliegue , lambda.min.ratio=1e-6 , nlambda=150)
plot(ridge$glmnet.fit , xvar="lambda")
plot(ridge)

## 6.4. lasso (alpha = 1): entran de a una
lasso <- cv.glmnet(x=x_train , y=y_train , alpha=1 , foldid=pliegue)
plot(lasso$glmnet.fit , xvar="lambda")
plot(lasso)
c(lambda_min = lasso$lambda.min , lambda_1se = lasso$lambda.1se)

## el orden de entrada al camino del lasso
betas <- as.matrix(lasso$glmnet.fit$beta)
entrada <- apply(betas!=0 , 1 , function(x) which(x)[1])
names(sort(entrada))[1:6]

## lo que queda con la regla de un error estandar
coef_1se <- coef(lasso , s="lambda.1se")
round(coef_1se[coef_1se[,1]!=0 , , drop=F] , 3)

## 6.5. elastic net (alpha = 0,5): las dos penalizaciones a la vez
en <- cv.glmnet(x=x_train , y=y_train , alpha=0.5 , foldid=pliegue)

## 6.6. la tabla: coeficientes, rmse dentro, de cv y de prueba (la prueba, una sola vez)
tabla_metro <- data.frame(modelo = c("recta (educacion superior)","ols con la ficha","ridge (lambda min)",
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

## ols reparte al azar, con errores estandar enormes
summary(lm(voto_petro ~ educ_superior + educ_gemela , data=train))$coefficients
summary(modelo_simple)$coefficients

## ridge reparte, el lasso sortea, elastic net agrupa: los caminos de las dos gemelas
z_gemela <- scale(as.matrix(train[,c("educ_superior","educ_gemela")]))
par(mfrow=c(1,3))
plot(glmnet(z_gemela , y_train , alpha=0 , standardize=F) , xvar="lambda" , main="ridge")
plot(glmnet(z_gemela , y_train , alpha=1 , standardize=F) , xvar="lambda" , main="lasso")
plot(glmnet(z_gemela , y_train , alpha=0.5 , standardize=F) , xvar="lambda" , main="elastic net")
par(mfrow=c(1,1))

##==: 7. el pais: juez y perilla con 1.291 columnas :==##

## 7.1. la particion fija del curso: 9.602 puestos para aprender, 2.398 bajo llave
train_p <- db %>% subset(muestra=="entrenamiento")
test_p <- db %>% subset(muestra=="prueba")
c(train = nrow(train_p) , test = nrow(test_p))

## validar: en el pais el target es casi plano
hist(train_p$voto_petro)
c(pais = sd(train_p$voto_petro) , metro = sd(train$voto_petro))

## 7.2. la media y la recta de la semana 2, en el pais
recta_p <- lm(voto_petro ~ educ_superior , data=train_p)
summary(recta_p)$r.squared
c(media = sqrt(mean((test_p$voto_petro - mean(train_p$voto_petro))^2)) ,
  recta = sqrt(mean((test_p$voto_petro - predict(recta_p , test_p))^2)))

## 7.3. una recta por departamento: la pendiente de la educacion superior
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

## el embudo: pocos puestos, pendientes que son ruido; muchos, pendientes precisas y distintas
plot(log(pendientes$n) , pendientes$b , pch=16 , col="blue" , ylim=c(-6,2) ,
     xlab="log de los puestos del departamento" , ylab="pendiente de la educacion superior")
abline(h=coef(recta_p)[2] , lty=2)

## 7.4. la ficha larga: 37 efectos nacionales + 33 constantes + 37 x 33 desviaciones
## (una columna por departamento, sin categoria de referencia: cada uno se encoge hacia lo nacional)
nombres <- c(x_ficha , paste0("dpto:" , dptos) , paste0(rep(x_ficha , each=length(dptos)) , ":" , dptos))
x_larga <- matrix(0 , nrow(train_p) , length(nombres) , dimnames=list(NULL , nombres))
x_larga_test <- matrix(0 , nrow(test_p) , length(nombres) , dimnames=list(NULL , nombres))
x_larga[,x_ficha] <- as.matrix(train_p[,x_ficha])
x_larga_test[,x_ficha] <- as.matrix(test_p[,x_ficha])
for (d in dptos){
     x_larga[,paste0("dpto:",d)] <- as.numeric(train_p$departamento==d)
     x_larga_test[,paste0("dpto:",d)] <- as.numeric(test_p$departamento==d)
     for (v in x_ficha){
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

## 7.7. ridge, lasso y elastic net con los mismos pliegues (en paralelo: un par de minutos cada uno)
ridge_p <- cv.glmnet(x=x_larga , y=train_p$voto_petro , alpha=0 , foldid=pliegue_p , lambda.min.ratio=1e-7 , nlambda=150 , parallel=T)
lasso_p <- cv.glmnet(x=x_larga , y=train_p$voto_petro , alpha=1 , foldid=pliegue_p , parallel=T)
en_p <- cv.glmnet(x=x_larga , y=train_p$voto_petro , alpha=0.5 , foldid=pliegue_p , parallel=T)
plot(lasso_p)

## 7.8. la tabla del pais: la media, las lineales de siempre y las penalizadas
f_dpto <- update(f_ficha , . ~ . + departamento)
modelo_ficha_p <- lm(f_ficha , data=train_p)
modelo_dpto_p <- lm(f_dpto , data=train_p)
tabla_pais <- data.frame(modelo = c("media del train (baseline)","recta (educacion superior)","la ficha de 37",
                                    "ficha + departamentos","ols con la ficha larga","ridge, ficha larga",
                                    "lasso, ficha larga","elastic net, ficha larga"),
                         coeficientes = NA , rmse_train = NA , rmse_cv = NA , rmse_test = NA)

## las lineales, con los mismos 10 pliegues
formulas_p <- list(voto_petro ~ 1 , voto_petro ~ educ_superior , f_ficha , f_dpto)
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

## ols con la ficha larga (de 7.6)
tabla_pais[5,2:5] <- c(sum(!is.na(ols_larga$coefficients)) - 1 , sqrt(mean((train_p$voto_petro - cbind(1 , x_larga) %*% b_larga)^2)) ,
                       rmse_cv_ols_larga , sqrt(mean((test_p$voto_petro - cbind(1 , x_larga_test) %*% b_larga)^2)))

## las penalizadas, con lambda min
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

## el peor pliegue de ridge: de donde sale su error
peor <- which.max(por_pliegue$ridge)
ridge_peor <- glmnet(x_larga[pliegue_p!=peor,] , train_p$voto_petro[pliegue_p!=peor] , alpha=0 , lambda=ridge_p$lambda.min)
train_p[pliegue_p==peor,] %>%
     mutate(error2 = (voto_petro - as.numeric(predict(ridge_peor , x_larga[pliegue_p==peor,])))^2) %>%
     group_by(departamento) %>%
     summarise(puestos = n() , parte_del_error = sum(error2)) %>%
     mutate(parte_del_error = round(parte_del_error/sum(parte_del_error)*100,1)) %>%
     arrange(desc(parte_del_error)) %>%
     head(3)

## 7.10. la ventaja del lasso sobre la ficha con departamentos: 2.000 remuestras de la prueba
test_p$voto_pred <- as.numeric(predict(lasso_p , x_larga_test , s="lambda.min"))
pred_dpto <- predict(modelo_dpto_p , test_p)
set.seed(2026)
dif <- rep(NA , 2000)
for (b in 1:2000){
     i <- sample(nrow(test_p) , replace=T)
     dif[b] <- sqrt(mean((test_p$voto_petro[i] - test_p$voto_pred[i])^2)) - sqrt(mean((test_p$voto_petro[i] - pred_dpto[i])^2))
}
quantile(dif , c(0.025,0.975))

## target predicho vs target original, y la diagonal
test_p %>% select(nombre , departamento , voto_petro , voto_pred) %>% head(5)
plot(test_p$voto_petro , test_p$voto_pred , pch=16 , col="blue" , cex=0.5 , xlim=c(0,100) , ylim=c(0,100),
     xlab="target original (%)" , ylab="target predicho (%)")
abline(a=0 , b=1 , lty=2)

## 7.11. que eligio el lasso: efectos nacionales, constantes y desviaciones por departamento
coef_l <- coef(lasso_p , s="lambda.min")[-1,1]
elegidas <- names(coef_l)[coef_l!=0]
nacionales <- elegidas[elegidas %in% x_ficha]
desviaciones <- elegidas[grepl(":" , elegidas) & !grepl("^dpto:" , elegidas)]
c(total = length(elegidas) , nacionales = length(nacionales) ,
  constantes = sum(grepl("^dpto:" , elegidas)) , desviaciones = length(desviaciones))

## la educacion superior no entra como efecto nacional; lo que cambia por departamento es otra cosa
"educ_superior" %in% elegidas
sort(table(sub(":.*" , "" , desviaciones)) , decreasing=T)[1:5]
sort(table(sub(".*:" , "" , desviaciones)) , decreasing=T)[1:5]

## 7.12. cambia la lista, no la prediccion: el mismo lambda en 20 remuestras bootstrap (cerca de un minuto)
set.seed(2026)
frecuencia <- rep(0 , ncol(x_larga))
pred_boot <- matrix(NA , nrow(x_larga) , 20)
for (b in 1:20){
     i <- sample(nrow(x_larga) , replace=T)
     lasso_b <- glmnet(x_larga[i,] , train_p$voto_petro[i] , alpha=1 , lambda=lasso_p$lambda.min)
     frecuencia <- frecuencia + as.numeric(coef(lasso_b)[-1,1]!=0)
     pred_boot[,b] <- as.numeric(predict(lasso_b , x_larga))
}
names(frecuencia) <- colnames(x_larga)
c(alguna_vez = sum(frecuencia>0) , una_vez = sum(frecuencia==1) , siempre = sum(frecuencia==20))
frecuencia["educ_superior"]
hist(frecuencia[frecuencia>0] , breaks=0:20 , xlab="remuestras en las que entra la columna (de 20)" , main="")

## la educacion superior no hace falta: las nacionales que si entran ya la cuentan
summary(lm(as.formula(paste0("educ_superior ~ " , paste(nacionales , collapse=" + "))) , data=train_p))$r.squared

## la prediccion de un mismo puesto casi no se mueve entre remuestras
median(apply(pred_boot , 1 , sd))

##==: 8. dos trampas del juez :==##

## 8.1. una columna que ya sabe la respuesta: el voto promedio del municipio, calculado con todos
train_p$media_municipio <- ave(train_p$voto_petro , train_p$cod_municipio)
f_trampa <- update(f_ficha , . ~ . + media_municipio)
error2 <- rep(NA , nrow(train_p))
for (j in 1:10){
     modelo_j <- lm(f_trampa , data=train_p[pliegue_p!=j,])
     error2[pliegue_p==j] <- (train_p$voto_petro[pliegue_p==j] - predict(modelo_j , train_p[pliegue_p==j,]))^2
}
cv_trampa <- sqrt(mean(error2))

## lo honesto: el promedio se calcula dentro de cada pliegue, sin el pliegue
error2 <- rep(NA , nrow(train_p))
for (j in 1:10){
     a <- train_p[pliegue_p!=j,]
     b <- train_p[pliegue_p==j,]
     medias <- a %>% group_by(cod_municipio) %>% summarise(media_municipio = mean(voto_petro))
     a$media_municipio <- ave(a$voto_petro , a$cod_municipio)
     b <- b %>% select(-media_municipio) %>% left_join(medias , by="cod_municipio")
     b$media_municipio[is.na(b$media_municipio)] <- mean(a$voto_petro)
     error2[pliegue_p==j] <- (b$voto_petro - predict(lm(f_trampa , data=a) , b))^2
}
cv_honesto <- sqrt(mean(error2))

## y en la prueba, con el promedio calculado solo con el train
medias <- train_p %>% group_by(cod_municipio) %>% summarise(media_municipio = mean(voto_petro))
test_trampa <- test_p %>% left_join(medias , by="cod_municipio")
test_trampa$media_municipio[is.na(test_trampa$media_municipio)] <- mean(train_p$voto_petro)
c(cv_con_trampa = cv_trampa , cv_honesto = cv_honesto ,
  rmse_test = sqrt(mean((test_trampa$voto_petro - predict(lm(f_trampa , data=train_p) , test_trampa))^2)))
train_p$media_municipio <- NULL

## 8.2. vecinos que se copian: pliegues al azar o pliegues de municipios enteros
municipios <- unique(train_p$cod_municipio)
set.seed(2026)
grupo <- sample(rep(1:10 , length.out=length(municipios)))
pliegue_m <- grupo[match(train_p$cod_municipio , municipios)]

## las lineales con matrices (un departamento que falte en el pliegue queda en cero)
x_media <- matrix(1 , nrow(train_p) , 1)
x_ficha_p <- cbind(1 , as.matrix(train_p[,x_ficha]))
x_dpto <- cbind(1 , as.matrix(train_p[,x_ficha]) , model.matrix(~ departamento , train_p)[,-1])
matrices <- list(x_media , x_ficha_p , x_dpto)
esquemas <- data.frame(modelo = c("media del train","la ficha de 37","ficha + departamentos","lasso, ficha larga","k-nn en el mapa"),
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

## el lasso de la ficha larga, validado por municipio (en paralelo: un par de minutos)
lasso_m <- cv.glmnet(x=x_larga , y=train_p$voto_petro , alpha=1 , foldid=pliegue_m , parallel=T)
esquemas[4,2:3] <- c(sqrt(min(lasso_p$cvm)) , sqrt(min(lasso_m$cvm)))

## k-nn en el mapa del pais: el juez elige k con los pliegues al azar
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

## el k-nn del mapa en la prueba, con el k del juez
rmse_knn_p <- sqrt(mean((test_p$voto_petro - knn_reg(coord_p , train_p$voto_petro , as.matrix(test_p[,c("lon","lat")]) , k_pais)[,1])^2))

## rmse de prueba de todos los modelos del pais, ordenados (la prueba se abrio una sola vez por modelo)
tabla_pais %>%
     select(modelo , rmse_cv , rmse_test) %>%
     rbind(data.frame(modelo = paste0("k-nn en el mapa (k = " , k_pais , ")") ,
                      rmse_cv = min(knn_p$al_azar) , rmse_test = rmse_knn_p)) %>%
     mutate(rmse_cv = round(rmse_cv,1) , rmse_test = round(rmse_test,1)) %>%
     arrange(rmse_test)

## stop cluster
stopCluster(cl)
