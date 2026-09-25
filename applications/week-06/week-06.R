## Big Data y Machine Learning para Economia Aplicada
## week-06: abrir la caja negra (que aprendio la maquina)
## Last run: Sep 25, 2026

##==: 0. initial setup :==##

## clean environment
rm(list=ls())

## load/install packages
require(pacman)
p_load(rio , dplyr , xgboost , doParallel)

## cluster: los reentrenamientos del boosting corren en paralelo, cada tarea con su semilla (mismo resultado, menos tiempo)
cores <- parallel::detectCores()
cl <- makeCluster(max(1 , cores - 2))
registerDoParallel(cl)

##==: 1. prepare data :==##

## 1.1. load data (la base publica del curso: la carpeta input/ del zip o github)
url <- "https://raw.githubusercontent.com/eduard-martinez/bdml-applied-economics/main/applications/week-06/input/"
puestos <- import("input/puestos_colombia_2022.rds" , trust=T)
censo <- import("input/censo_puestos_2022.rds" , trust=T)

## 1.2. la ficha del censo de la semana 3: de conteos a proporciones (una categoria por bloque queda por fuera)
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

## 1.3. la base: el puesto, los dos targets de siempre y la ficha; el departamento como una sola variable
db <- puestos %>%
      select(puesto , nombre , departamento , municipio , cod_municipio , zona , lon , lat , registrados ,
             voto_petro , gana_petro , muestra) %>%
      inner_join(ficha , by="puesto") %>%
      mutate(log_registrados = log(registrados) , rural = as.integer(zona=="rural") ,
             dpto = factor(departamento)) %>%
      subset(!is.na(educ_superior) & !is.na(afro))
nrow(db)

## 1.4. los ingredientes del modelo de hoy: la ficha de 37 y el departamento (los de la semana 3)
x_ficha <- c(setdiff(names(ficha) , "puesto") , "log_registrados" , "rural")
x_dpto <- c(x_ficha , "dpto")
length(x_ficha)

## 1.5. el pais: la particion fija del curso y los 10 pliegues de la semana 3 (semilla 2026)
train_p <- db %>% subset(muestra=="entrenamiento")
test_p <- db %>% subset(muestra=="prueba")
set.seed(2026)
pliegue_p <- sample(rep(1:10 , length.out=nrow(train_p)))
pliegues_p <- lapply(1:10 , function(j) which(pliegue_p==j))
c(train = nrow(train_p) , test = nrow(test_p))

## 1.6. para xgboost, el departamento en 33 columnas de ceros y unos: 70 columnas en total
x_dpto_m <- model.matrix(~ . - 1 , data=train_p[,x_dpto])
x_dpto_t <- model.matrix(~ . - 1 , data=test_p[,x_dpto])
col_dpto <- grep("^dpto" , colnames(x_dpto_m) , value=T)
dim(x_dpto_m)

## 1.7. los siete bloques de la sesion: seis del censo y el departamento
bloques <- list(educacion = c("educ_ninguna","educ_tecnica","educ_superior") ,
                edad_genero = c("edad_0_19","edad_20_29","edad_45_59","edad_60_74","edad_75","mujeres","edad_votar") ,
                etnia = c("indigena","rom","raizal","palenquero","afro") ,
                servicios = c("serv_agua","serv_alcantarillado","serv_energia","serv_gas","serv_internet") ,
                estrato = paste0("estrato_" , 0:6) ,
                escala = c("hogares_por_vivienda","personas_por_hogar","log_poblacion","log_viviendas","log_manzanas",
                           "manzanas_compartidas","log_distancia","desfase","log_registrados","rural") ,
                departamento = col_dpto)
sapply(bloques , length)

##==: 2. el modelo que vamos a abrir: el boosting de la semana 5 :==##

## 2.1. el juez de xgboost de la semana 5: validacion cruzada con parada temprana (se detiene tras 50 rondas sin
## mejorar); devuelve las rondas que elige y los parametros. un hilo por tarea: el paralelo va por fuera, con foreach
cv_xgb <- function(x , y , pliegues , objetivo , d , eta){
          parametros <- xgb.params(objective=objetivo , eta=eta , max_depth=d , subsample=0.8 , colsample_bytree=0.8 ,
                                   eval_metric=ifelse(objetivo=="binary:logistic" , "logloss" , "rmse") , nthread=1)
          set.seed(2026)
          cv <- xgb.cv(params=parametros , data=xgb.DMatrix(x , label=y) , nrounds=10000 , folds=pliegues ,
                       early_stopping_rounds=50 , verbose=0)
          return(list(rondas=cv$early_stop$best_iteration , parametros=parametros))
}

## 2.2. el boosting de la ficha + departamento: d = 6 y tasa 0,05, con las rondas que elige el juez (cerca de un minuto)
juez <- cv_xgb(x_dpto_m , train_p$voto_petro , pliegues_p , "reg:squarederror" , 6 , 0.05)
juez$rondas
set.seed(2026)
modelo <- xgb.train(params=juez$parametros , data=xgb.DMatrix(x_dpto_m , label=train_p$voto_petro) ,
                    nrounds=juez$rondas , verbose=0)

## target predicho vs target original: el modelo que vamos a abrir, en la prueba
test_p$voto_pred <- predict(modelo , xgb.DMatrix(x_dpto_t))
test_p %>% select(nombre , departamento , voto_petro , voto_pred) %>% head(5)
rmse_modelo <- sqrt(mean((test_p$voto_petro - test_p$voto_pred)^2))
rmse_modelo

## 2.3. su tamano: nadie lee cientos de arboles con decenas de miles de hojas
arboles <- xgb.model.dt.tree(model=modelo)
c(arboles = length(unique(arboles$Tree)) , hojas = sum(arboles$Feature=="Leaf"))

##==: 3. la importancia que trae el modelo: la ganancia :==##

## 3.1. la ganancia de cada columna: cuanto bajo el error de entrenamiento en los cortes que la usan
## (el departamento con sus 33 columnas sumadas)
imp <- xgb.importance(model=modelo)
imp$var <- ifelse(grepl("^dpto" , imp$Feature) , "dpto" , imp$Feature)
ganancia <- imp %>% group_by(var) %>% summarise(ganancia = sum(Gain) , .groups="drop") %>% arrange(desc(ganancia))
ganancia %>% mutate(ganancia = round(100*ganancia , 1)) %>% head(10)
par(mar=c(4,8,2,1))
barplot(rev(100*ganancia$ganancia[1:10]) , names.arg=rev(ganancia$var[1:10]) , horiz=T , las=1 , cex.names=0.7 ,
        col="blue" , xlab="ganancia (% del total, en el entrenamiento)")
par(mar=c(5,4,4,2))

## 3.2. la pregunta para la sala: una columna de numeros al azar (semilla 99), el mismo modelo con las mismas rondas
set.seed(99)
ruido <- runif(nrow(x_dpto_m))
ruido_t <- runif(nrow(x_dpto_t))
x_ruido_m <- cbind(x_dpto_m , ruido = ruido)
x_ruido_t <- cbind(x_dpto_t , ruido = ruido_t)
set.seed(2026)
modelo_r <- xgb.train(params=juez$parametros , data=xgb.DMatrix(x_ruido_m , label=train_p$voto_petro) ,
                      nrounds=juez$rondas , verbose=0)
imp_r <- xgb.importance(model=modelo_r)
imp_r$var <- ifelse(grepl("^dpto" , imp_r$Feature) , "dpto" , imp_r$Feature)
ganancia_r <- imp_r %>% group_by(var) %>% summarise(ganancia = sum(Gain) , .groups="drop") %>% arrange(desc(ganancia)) %>%
              mutate(puesto = row_number())

## el ruido le gana a variables reales: una variable continua ofrece miles de umbrales (sesgo hacia la cardinalidad)
ganancia_r %>% subset(var %in% c("afro","educ_superior","ruido","rural"))
nrow(ganancia_r)

## 3.3. el mismo ruido, barajado en la prueba: no vale nada (diez barajadas, semilla 2026)
rmse_r <- sqrt(mean((test_p$voto_petro - predict(modelo_r , xgb.DMatrix(x_ruido_t)))^2))
error_ruido <- rep(NA , 10)
set.seed(2026)
for (k in 1:10){
     x_p <- x_ruido_t
     x_p[,"ruido"] <- sample(x_p[,"ruido"])
     error_ruido[k] <- sqrt(mean((test_p$voto_petro - predict(modelo_r , xgb.DMatrix(x_p)))^2))
}
mean(error_ruido) - rmse_r

##==: 4. la importancia por permutacion: cuanto duele barajar :==##

## 4.1. barajar una columna en la prueba (sin reentrenar) y medir cuanto sube el error; diez barajadas por
## variable (semilla 2026). el departamento se baraja entero: sus 33 columnas con el mismo orden
grid_perm <- data.frame(var = c(x_ficha , "dpto") , sube = NA , de = NA)
set.seed(2026)
for (i in 1:nrow(grid_perm)){
     if (grid_perm$var[i]=="dpto") columnas <- col_dpto else columnas <- grid_perm$var[i]
     error_k <- rep(NA , 10)
     for (k in 1:10){
          orden <- sample(nrow(x_dpto_t))
          x_p <- x_dpto_t
          x_p[,columnas] <- x_dpto_t[orden,columnas]
          error_k[k] <- sqrt(mean((test_p$voto_petro - predict(modelo , xgb.DMatrix(x_p)))^2))
     }
     grid_perm$sube[i] <- mean(error_k) - rmse_modelo
     grid_perm$de[i] <- sd(error_k)
}
grid_perm <- grid_perm %>% arrange(desc(sube))
grid_perm %>% mutate(sube = round(sube , 2) , de = round(de , 2)) %>% head(10)

## la educacion superior, la estrella de la semana 2, apenas mueve el error
grid_perm %>% subset(var=="educ_superior")

## las dos importancias, lado a lado: coinciden arriba y se separan abajo
par(mfrow=c(1,2) , mar=c(4,8,2,1))
barplot(rev(100*ganancia$ganancia[1:10]) , names.arg=rev(ganancia$var[1:10]) , horiz=T , las=1 , cex.names=0.7 ,
        col="blue" , main="ganancia (%)")
barplot(rev(grid_perm$sube[1:10]) , names.arg=rev(grid_perm$var[1:10]) , horiz=T , las=1 , cex.names=0.7 ,
        col="red" , main="permutacion (puntos)")
par(mfrow=c(1,1) , mar=c(5,4,4,2))

## 4.2. barajar un bloque entero: todas sus columnas juntas, con el mismo orden de filas (semilla 2026)
grid_bloque <- data.frame(bloque = names(bloques) , columnas = sapply(bloques , length) , permutar = NA)
set.seed(2026)
for (i in 1:nrow(grid_bloque)){
     columnas <- bloques[[i]]
     error_k <- rep(NA , 10)
     for (k in 1:10){
          orden <- sample(nrow(x_dpto_t))
          x_p <- x_dpto_t
          x_p[,columnas] <- x_dpto_t[orden,columnas]
          error_k[k] <- sqrt(mean((test_p$voto_petro - predict(modelo , xgb.DMatrix(x_p)))^2))
     }
     grid_bloque$permutar[i] <- mean(error_k) - rmse_modelo
}
grid_bloque

##==: 5. permutar contra reentrenar: la ablacion por bloques :==##

## 5.1. reentrenar sin cada bloque, con las rondas que elija el juez (en paralelo: un par de minutos)
ablacion <- foreach(b = names(bloques) , .packages="xgboost") %dopar% {
            quedan <- setdiff(colnames(x_dpto_m) , bloques[[b]])
            juez_b <- cv_xgb(x_dpto_m[,quedan] , train_p$voto_petro , pliegues_p , "reg:squarederror" , 6 , 0.05)
            set.seed(2026)
            modelo_b <- xgb.train(params=juez_b$parametros , data=xgb.DMatrix(x_dpto_m[,quedan] , label=train_p$voto_petro) ,
                                  nrounds=juez_b$rondas , verbose=0)
            c(rondas = juez_b$rondas ,
              rmse_test = sqrt(mean((test_p$voto_petro - predict(modelo_b , xgb.DMatrix(x_dpto_t[,quedan])))^2)))
}
grid_bloque$rondas <- NA
grid_bloque$reentrenar <- NA
for (i in 1:nrow(grid_bloque)){
     grid_bloque$rondas[i] <- ablacion[[i]]["rondas"]
     grid_bloque$reentrenar[i] <- ablacion[[i]]["rmse_test"] - rmse_modelo
}

## 5.2. permutar pregunta por el modelo que tenemos; reentrenar, por el mejor modelo sin ese bloque.
## los bloques del censo se reemplazan entre si; el departamento, no del todo
grid_bloque %>% mutate(permutar = round(permutar , 2) , reentrenar = round(reentrenar , 2))
par(mar=c(4,9,2,1))
barplot(t(as.matrix(grid_bloque[nrow(grid_bloque):1,c("reentrenar","permutar")])) , beside=T , horiz=T , las=1 ,
        names.arg=grid_bloque$bloque[nrow(grid_bloque):1] , cex.names=0.8 , col=c("blue","red") ,
        xlab="cuanto sube el rmse de prueba (puntos)" , legend.text=c("reentrenar sin el bloque","permutar el bloque") ,
        args.legend=list(x="bottomright" , cex=0.8))
par(mar=c(5,4,4,2))

##==: 6. la dependencia parcial y las curvas ice: la poblacion afro :==##

## 6.1. fijar la poblacion afro en cada valor de la malla para los 2.398 puestos de prueba y predecir (una curva ice
## por puesto); el pdp es su promedio
malla_a <- seq(0 , 100 , by=2.5)
ice_a <- matrix(NA , nrow(x_dpto_t) , length(malla_a))
for (g in 1:length(malla_a)){
     x_p <- x_dpto_t
     x_p[,"afro"] <- malla_a[g]
     ice_a[,g] <- predict(modelo , xgb.DMatrix(x_p))
}
pdp_a <- colMeans(ice_a)
data.frame(afro = malla_a , pdp = round(pdp_a , 1)) %>% subset(afro %in% c(0,10,20,50,70,80,100))

## 6.2. donde viven los datos: los deciles de la poblacion afro (el 80 % de los puestos tiene muy poca)
quantile(test_p$afro , seq(0.1 , 0.9 , 0.1))

## 6.3. cincuenta curvas ice al azar (semilla 2026) y el pdp: casi paralelas, cada una a su nivel
set.seed(2026)
muestra_ice <- sample(nrow(x_dpto_t) , 50)
matplot(malla_a , t(ice_a[muestra_ice,]) , type="l" , lty=1 , col="gray70" , ylim=c(0,100) ,
        xlab="poblacion afro del puesto (%)" , ylab="voto predicho (%)")
lines(malla_a , pdp_a , col="red" , lwd=3)
rug(quantile(test_p$afro , seq(0.1 , 0.9 , 0.1)) , col="blue" , lwd=2)

##==: 7. la estrella de cali, en el pais: la educacion superior por grupos :==##

## 7.1. las curvas ice de la educacion superior, de 0 a 60 %
malla_e <- seq(0 , 60 , by=2)
ice_e <- matrix(NA , nrow(x_dpto_t) , length(malla_e))
for (g in 1:length(malla_e)){
     x_p <- x_dpto_t
     x_p[,"educ_superior"] <- malla_e[g]
     ice_e[,g] <- predict(modelo , xgb.DMatrix(x_p))
}

## 7.2. el pdp del pais y el promedio de las ice de tres grupos de puestos
grupos_e <- data.frame(grupo = c("pais","cali","choco","antioquia") , n = NA , educ_0 = NA , educ_60 = NA ,
                       minimo = NA , maximo = NA)
curvas_e <- matrix(NA , length(malla_e) , 4)
for (i in 1:4){
     if (grupos_e$grupo[i]=="pais") filas <- rep(T , nrow(test_p))
     if (grupos_e$grupo[i]=="cali") filas <- test_p$municipio=="CALI"
     if (grupos_e$grupo[i]=="choco") filas <- test_p$departamento=="CHOCO"
     if (grupos_e$grupo[i]=="antioquia") filas <- test_p$departamento=="ANTIOQUIA"
     curvas_e[,i] <- colMeans(ice_e[filas,,drop=F])
     grupos_e$n[i] <- sum(filas)
     grupos_e$educ_0[i] <- curvas_e[1,i]
     grupos_e$educ_60[i] <- curvas_e[length(malla_e),i]
     grupos_e$minimo[i] <- min(curvas_e[,i])
     grupos_e$maximo[i] <- max(curvas_e[,i])
}

## en el pais casi plana; pero en cali subir la educacion baja la prediccion y en el choco la sube
grupos_e %>% mutate(across(-c(grupo , n) , ~ round(.x , 1)))
matplot(malla_e , curvas_e , type="l" , lty=1 , lwd=c(3,2,2,2) , col=c("black","red","darkgreen","blue") , ylim=c(20,100) ,
        xlab="educacion superior (%), fijada para todos los puestos del grupo" , ylab="voto predicho (%)")
legend("topright" , legend=grupos_e$grupo , col=c("black","red","darkgreen","blue") , lwd=2 , cex=0.8)

##==: 8. cuando el pdp pregunta por puestos que no existen: el ale :==##

## 8.1. la educacion superior y el estrato 1 van en contra: el pdp en 40 % pone ahi a todos los puestos,
## tambien a los que tienen mas de la mitad de sus viviendas en estrato 1, y ninguno de esos pasa de 35 %
cor(test_p$educ_superior , test_p$estrato_1)
c(estrato_1_alto = sum(test_p$estrato_1 > 50) , proporcion = mean(test_p$estrato_1 > 50) ,
  educ_max = max(test_p$educ_superior[test_p$estrato_1 > 50]))
plot(test_p$educ_superior , test_p$estrato_1 , pch=16 , cex=0.3 , col="gray50" ,
     xlab="educacion superior (%)" , ylab="estrato 1 (% de viviendas)")
abline(v=40 , col="red" , lwd=2)

## 8.2. el ale (apley y zhu): veinte franjas de la educacion superior (de a 5 % de los puestos); en cada franja,
## cuanto cambia la prediccion de sus propios puestos entre los dos bordes
z <- unique(quantile(test_p$educ_superior , seq(0 , 1 , length.out=21)))
franja <- cut(test_p$educ_superior , z , include.lowest=T , labels=F)
grid_ale <- data.frame(desde = z[-length(z)] , hasta = z[-1] , n = NA , diferencia = NA)
for (i in 1:nrow(grid_ale)){
     idx <- which(franja==i)
     x_arriba <- x_dpto_t[idx,,drop=F]
     x_abajo <- x_arriba
     x_arriba[,"educ_superior"] <- grid_ale$hasta[i]
     x_abajo[,"educ_superior"] <- grid_ale$desde[i]
     grid_ale$n[i] <- length(idx)
     grid_ale$diferencia[i] <- mean(predict(modelo , xgb.DMatrix(x_arriba)) - predict(modelo , xgb.DMatrix(x_abajo)))
}

## 8.3. acumular las diferencias y centrar (el promedio ponderado por puestos queda en cero)
acumulado <- c(0 , cumsum(grid_ale$diferencia))
ale <- acumulado - sum(((acumulado[-1] + acumulado[-length(acumulado)])/2)*grid_ale$n)/sum(grid_ale$n)

## 8.4. el pdp en los mismos puntos, centrado: esta vez los dos coinciden
pdp_z <- rep(NA , length(z))
for (g in 1:length(z)){
     x_p <- x_dpto_t
     x_p[,"educ_superior"] <- z[g]
     pdp_z[g] <- mean(predict(modelo , xgb.DMatrix(x_p)))
}
pdp_c <- pdp_z - mean(pdp_z)
data.frame(educ_superior = round(z , 1) , pdp = round(pdp_c , 2) , ale = round(ale , 2))
max(abs(pdp_c - ale))
plot(z , pdp_c , type="l" , col="red" , lwd=2 , ylim=c(-4,4) ,
     xlab="educacion superior (%)" , ylab="efecto sobre el voto (centrado)")
lines(z , ale , col="blue" , lwd=2 , lty=2)
abline(h=0 , col="gray50")
legend("topright" , legend=c("pdp","ale") , col=c("red","blue") , lwd=2 , lty=c(1,2))

##==: 9. los cruces que encontro la maquina: interacciones shap :==##

## 9.1. el reparto shap de un puesto (la parte 3) se parte en un efecto propio de cada columna y uno por cada par;
## es caro (una matriz de 71 x 71 por puesto): 500 puestos de prueba al azar (semilla 2026; cerca de un minuto)
set.seed(2026)
muestra_i <- sample(nrow(x_dpto_t) , 500)
inter <- predict(modelo , xgb.DMatrix(x_dpto_t[muestra_i,]) , predinteraction=T)
dim(inter)

## 9.2. el departamento como un solo bloque: se suman sus 33 filas y columnas; la fuerza de cada cruce es
## el promedio de |phi_jk + phi_kj| en los 500 puestos
grupo_col <- c(colnames(x_dpto_t) , "BIAS")
grupo_col[grepl("^dpto" , grupo_col)] <- "dpto"
variables <- setdiff(unique(grupo_col) , "BIAS")
fuerza <- matrix(0 , length(variables) , length(variables) , dimnames=list(variables , variables))
for (a in variables){
     for (b in variables){
          fuerza[a,b] <- mean(abs(apply(inter[,grupo_col==a,grupo_col==b,drop=F] , 1 , sum)))
     }
}
pares <- which(upper.tri(fuerza) , arr.ind=T)
cruces <- data.frame(a = variables[pares[,1]] , b = variables[pares[,2]] , fuerza = 2*fuerza[pares]) %>% arrange(desc(fuerza))

## los cruces de la semana 5 son, sobre todo, la etnia en cada departamento (lo que el lasso encontro en la semana 3)
cruces %>% mutate(fuerza = round(fuerza , 2)) %>% head(10)
par(mar=c(4,12,2,1))
barplot(rev(cruces$fuerza[1:10]) , names.arg=rev(paste(cruces$a[1:10] , "x" , cruces$b[1:10])) , horiz=T , las=1 ,
        cex.names=0.7 , col="orange" , xlab="fuerza del cruce (puntos de voto)")
par(mar=c(5,4,4,2))

##==: 10. es el choco o es la poblacion afro? shapley a mano en un puesto de quibdo :==##

## 10.1. el puesto y los tres jugadores: la educacion, la etnia y el departamento
quibdo <- test_p %>% subset(nombre=="COLEGIO PEDRO GRAU AROLA" & municipio=="QUIBDO")
quibdo %>% select(nombre , municipio , voto_petro , educ_superior , afro)
jugadores <- list(E = bloques$educacion , T = bloques$etnia , D = "departamento")

## 10.2. el juego: una regresion por cada coalicion de jugadores; v(S) es su prediccion para el puesto menos el
## promedio del entrenamiento (la coalicion vacia predice el promedio: v = 0)
media_train <- mean(train_p$voto_petro)
juego <- data.frame(coalicion = c("vacia","E","T","D","ET","ED","TD","ETD") , v = NA)
juego$v[1] <- 0
for (i in 2:nrow(juego)){
     S <- strsplit(juego$coalicion[i] , "")[[1]]
     f <- as.formula(paste("voto_petro ~" , paste(unlist(jugadores[S]) , collapse=" + ")))
     juego$v[i] <- predict(lm(f , data=train_p) , quibdo) - media_train
}
juego %>% mutate(v = round(v , 1))

## target predicho vs target original: la regresion con los tres bloques, en quibdo
c(voto_petro = quibdo$voto_petro , voto_pred = media_train + juego$v[juego$coalicion=="ETD"])

## 10.3. los seis ordenes de entrada: cuanto anade cada jugador al llegar. el que entra primero se lleva casi todo
ordenes <- data.frame(orden = c("ETD","EDT","TED","TDE","DET","DTE") , E = NA , T = NA , D = NA)
for (i in 1:nrow(ordenes)){
     dentro <- c()
     for (j in strsplit(ordenes$orden[i] , "")[[1]]){
          antes <- paste(c("E","T","D")[c("E","T","D") %in% dentro] , collapse="")
          despues <- paste(c("E","T","D")[c("E","T","D") %in% c(dentro , j)] , collapse="")
          if (antes=="") antes <- "vacia"
          ordenes[i,j] <- juego$v[juego$coalicion==despues] - juego$v[juego$coalicion==antes]
          dentro <- c(dentro , j)
     }
}
ordenes %>% mutate(across(-orden , ~ round(.x , 1)))

## 10.4. el valor de shapley: el promedio sobre los seis ordenes. suma lo que el puesto se aparta del promedio
shapley <- colMeans(ordenes[,c("E","T","D")])
round(c(shapley , suma = sum(shapley) , v_ETD = juego$v[juego$coalicion=="ETD"]) , 2)

##==: 11. shap: el reparto de cada prediccion del boosting :==##

## 11.1. treeshap exacto con predcontrib: una contribucion por columna y la base (BIAS, la prediccion promedio)
shap <- predict(modelo , xgb.DMatrix(x_dpto_t) , predcontrib=T)
colnames(shap) <- c(colnames(x_dpto_t) , "BIAS")
base <- shap[1,"BIAS"]
base

## la eficiencia: base + suma de las contribuciones = la prediccion, puesto a puesto (el error es de redondeo)
max(abs(rowSums(shap) - test_p$voto_pred))

## 11.2. el departamento como un bloque (sus 33 columnas sumadas); target predicho vs target original, con su reparto
shap_ag <- cbind(shap[,x_ficha] , dpto = rowSums(shap[,col_dpto]))
test_p %>% mutate(base = base , shap_dpto = shap_ag[,"dpto"] , shap_afro = shap_ag[,"afro"] , shap_resto = rowSums(shap_ag) - shap_dpto - shap_afro) %>%
           select(nombre , departamento , voto_petro , voto_pred , base , shap_dpto , shap_afro , shap_resto) %>% head(5)

## 11.3. de lo local a lo global: el promedio del valor absoluto, la misma jerarquia que la permutacion
media_abs <- sort(colMeans(abs(shap_ag)) , decreasing=T)
round(head(media_abs , 10) , 2)

## 11.4. dos colegios de cali: la base, las seis columnas que mas mueven cada prediccion, el resto y la prediccion
colegios <- data.frame(nombre = c("COLEGIO COMFANDI","COLEGIO FUNDACION COMPARTIR") , fila = NA , voto_petro = NA , voto_pred = NA)
for (i in 1:2){
     colegios$fila[i] <- which(test_p$nombre==colegios$nombre[i] & test_p$municipio=="CALI")
     colegios$voto_petro[i] <- test_p$voto_petro[colegios$fila[i]]
     colegios$voto_pred[i] <- test_p$voto_pred[colegios$fila[i]]
}
colegios
test_p[colegios$fila,] %>% select(nombre , educ_superior , estrato_1 , estrato_2 , estrato_3 , estrato_4 , estrato_5 , afro)
par(mfrow=c(1,2) , mar=c(4,8,3,1))
for (i in 1:2){
     s <- shap_ag[colegios$fila[i],]
     s <- s[order(-abs(s))]
     reparto <- c(s[1:6] , resto = sum(s[-(1:6)]))
     print(round(c(base = base , reparto , prediccion = base + sum(reparto)) , 2))
     barplot(rev(reparto) , horiz=T , las=1 , cex.names=0.7 , col=ifelse(rev(reparto) > 0 , "red" , "blue") ,
             main=colegios$nombre[i] , cex.main=0.7 , xlab="shap (puntos de voto)")
}
par(mfrow=c(1,1) , mar=c(5,4,4,2))

## el mismo departamento pesa distinto en los dos colegios: sin cruces pesaria igual
shap_ag[colegios$fila,"dpto"]

## 11.5. el beeswarm, a lo simple: una fila por variable, un punto por puesto; rojo si el puesto tiene mas que la
## mediana de esa variable, azul si menos (el departamento no tiene orden: gris)
top_8 <- names(media_abs)[1:8]
set.seed(7)
plot(NA , xlim=c(-25,30) , ylim=c(0.5,8.5) , yaxt="n" , xlab="valor shap: cuanto mueve la prediccion (puntos de voto)" , ylab="")
axis(2 , at=8:1 , labels=top_8 , las=1 , cex.axis=0.7)
for (r in 1:8){
     v <- top_8[r]
     if (v=="dpto") color <- "gray60" else color <- ifelse(test_p[[v]] > median(test_p[[v]]) , "red" , "blue")
     points(shap_ag[,v] , 9 - r + runif(nrow(shap_ag) , -0.3 , 0.3) , pch=16 , cex=0.3 , col=color)
}
abline(v=0 , lty=2)

##==: 12. cuando entra el mapa: la explicacion es del modelo :==##

## 12.1. el boosting con coordenadas de la semana 5, y el mapa sin el censo (en paralelo: cerca de un minuto)
x_mapa_m <- cbind(x_dpto_m , lon = train_p$lon , lat = train_p$lat)
x_mapa_t <- cbind(x_dpto_t , lon = test_p$lon , lat = test_p$lat)
x_solo_m <- cbind(x_dpto_m[,col_dpto] , lon = train_p$lon , lat = train_p$lat)
x_solo_t <- cbind(x_dpto_t[,col_dpto] , lon = test_p$lon , lat = test_p$lat)
mapas <- foreach(ingredientes = c("con_mapa","mapa_sin_censo") , .packages="xgboost") %dopar% {
         if (ingredientes=="con_mapa") x <- x_mapa_m else x <- x_solo_m
         if (ingredientes=="con_mapa") x_t <- x_mapa_t else x_t <- x_solo_t
         juez_m <- cv_xgb(x , train_p$voto_petro , pliegues_p , "reg:squarederror" , 6 , 0.05)
         set.seed(2026)
         modelo_m <- xgb.train(params=juez_m$parametros , data=xgb.DMatrix(x , label=train_p$voto_petro) ,
                               nrounds=juez_m$rondas , verbose=0)
         shap_m <- predict(modelo_m , xgb.DMatrix(x_t) , predcontrib=T)
         colnames(shap_m) <- c(colnames(x_t) , "BIAS")
         list(rondas = juez_m$rondas , pred = predict(modelo_m , xgb.DMatrix(x_t)) , shap = shap_m)
}

## casi el mismo error; el censo aporta poco una vez se conoce el mapa
c(sin_mapa = rmse_modelo ,
  con_mapa = sqrt(mean((test_p$voto_petro - mapas[[1]]$pred)^2)) ,
  mapa_sin_censo = sqrt(mean((test_p$voto_petro - mapas[[2]]$pred)^2)))

## target predicho vs target original: los dos modelos, puesto a puesto
test_p %>% mutate(voto_pred_mapa = mapas[[1]]$pred) %>% select(nombre , departamento , voto_petro , voto_pred , voto_pred_mapa) %>% head(5)

## 12.2. otro reparto: las coordenadas se llevan lo que era del departamento y de la poblacion afro (efecto rashomon)
shap_mapa <- cbind(mapas[[1]]$shap[,x_ficha] , dpto = rowSums(mapas[[1]]$shap[,col_dpto]) ,
                   lon = mapas[[1]]$shap[,"lon"] , lat = mapas[[1]]$shap[,"lat"])
media_abs_mapa <- colMeans(abs(shap_mapa))
vars_9 <- c("lon","lat","dpto","afro","indigena","estrato_2","educ_tecnica","edad_20_29")
rashomon <- data.frame(var = vars_9 , sin_mapa = 0 , con_mapa = media_abs_mapa[vars_9])
for (i in 1:nrow(rashomon)){
     if (rashomon$var[i] %in% names(media_abs)) rashomon$sin_mapa[i] <- media_abs[rashomon$var[i]]
}
rashomon %>% mutate(sin_mapa = round(sin_mapa , 1) , con_mapa = round(con_mapa , 1))
par(mar=c(4,7,2,1))
barplot(t(as.matrix(rashomon[nrow(rashomon):1,c("con_mapa","sin_mapa")])) , beside=T , horiz=T , las=1 ,
        names.arg=rashomon$var[nrow(rashomon):1] , cex.names=0.8 , col=c("darkcyan","red") ,
        xlab="|shap| promedio en la prueba (puntos de voto)" , legend.text=c("con el mapa","sin el mapa") ,
        args.legend=list(x="bottomright" , cex=0.8))
par(mar=c(5,4,4,2))

##==: 13. a quien le falla: el error por subgrupos :==##

## 13.1. por zona: el error vive en el campo, donde el censo dice menos
test_p$residuo <- test_p$voto_petro - test_p$voto_pred
test_p %>% group_by(rural) %>% summarise(n = n() , rmse = sqrt(mean(residuo^2)))

## 13.2. por nivel del voto: donde petro saco muy poco el modelo predice de mas, y donde saco mucho, de menos
test_p %>% mutate(tramo = cut(voto_petro , c(0,20,40,60,80,100) , include.lowest=T)) %>%
           group_by(tramo) %>% summarise(n = n() , real = mean(voto_petro) , predicho = mean(voto_pred))

## 13.3. por departamento: el promedio del pais esconde errores del doble (ojo con los que tienen pocos puestos)
por_dpto <- test_p %>% group_by(departamento) %>%
            summarise(n = n() , rmse = sqrt(mean(residuo^2)) , sesgo = mean(residuo) , .groups="drop") %>%
            arrange(desc(rmse))
por_dpto %>% head(5)
por_dpto %>% tail(5)
par(mar=c(4,9,2,1))
barplot(rev(por_dpto$rmse) , names.arg=rev(paste0(por_dpto$departamento , " (" , por_dpto$n , ")")) , horiz=T , las=1 ,
        cex.names=0.5 , col=ifelse(rev(por_dpto$rmse) > rmse_modelo , "red" , "blue") , xlab="rmse de prueba")
abline(v=rmse_modelo , lty=2)
par(mar=c(5,4,4,2))

## 13.4. quitar la variable sensible no basta: sin el bloque de etnia el modelo casi no pierde (seccion 5),
## porque el departamento y el resto del censo la reconstruyen
grid_bloque %>% subset(bloque=="etnia") %>% select(bloque , permutar , reentrenar)

##==: 14. la otra pregunta: gana_petro y shap en log-odds (apendice a7) :==##

## 14.1. el boosting de clasificacion de la semana 5 (ficha + departamento; la log-loss como perdida y como juez)
juez_c <- cv_xgb(x_dpto_m , train_p$gana_petro , pliegues_p , "binary:logistic" , 6 , 0.05)
set.seed(2026)
modelo_c <- xgb.train(params=juez_c$parametros , data=xgb.DMatrix(x_dpto_m , label=train_p$gana_petro) ,
                      nrounds=juez_c$rondas , verbose=0)

## target predicho vs target original: la probabilidad de que gane petro
test_p$prob_pred <- predict(modelo_c , xgb.DMatrix(x_dpto_t))
test_p %>% select(nombre , departamento , gana_petro , prob_pred) %>% head(5)

## la auc de la semana 4: la probabilidad de ordenar bien un par (un puesto que gana y uno que pierde)
r <- rank(test_p$prob_pred)
n1 <- sum(test_p$gana_petro==1)
n0 <- sum(test_p$gana_petro==0)
(sum(r[test_p$gana_petro==1]) - n1*(n1 + 1)/2)/(n1*n0)

## 14.2. shap en la escala de los log-odds: base + suma = log(p/(1 - p))
shap_c <- predict(modelo_c , xgb.DMatrix(x_dpto_t) , predcontrib=T)
colnames(shap_c) <- c(colnames(x_dpto_t) , "BIAS")
shap_c[1,"BIAS"]
max(abs(rowSums(shap_c) - log(test_p$prob_pred/(1 - test_p$prob_pred))))

## lo que mas mueve la probabilidad de que gane petro: la misma jerarquia que en el voto
shap_c_ag <- cbind(shap_c[,x_ficha] , dpto = rowSums(shap_c[,col_dpto]))
round(head(sort(colMeans(abs(shap_c_ag)) , decreasing=T) , 8) , 2)

##==: 15. tabla final :==##

## las tres importancias, lado a lado: ganancia (entrenamiento), permutacion y |shap| promedio (prueba)
importancias <- data.frame(var = names(media_abs) , ganancia = NA , permutacion = NA , shap = media_abs)
for (i in 1:nrow(importancias)){
     if (importancias$var[i] %in% ganancia$var) importancias$ganancia[i] <- 100*ganancia$ganancia[ganancia$var==importancias$var[i]]
     importancias$permutacion[i] <- grid_perm$sube[grid_perm$var==importancias$var[i]]
}
importancias %>% mutate(across(-var , ~ round(.x , 2))) %>% head(10)

## los bloques: permutar contra reentrenar
grid_bloque %>% mutate(permutar = round(permutar , 1) , reentrenar = round(reentrenar , 1)) %>% select(bloque , permutar , reentrenar)

## el error: con y sin el mapa, en la prueba y en el campo
c(sin_mapa = rmse_modelo , con_mapa = sqrt(mean((test_p$voto_petro - mapas[[1]]$pred)^2)) ,
  rural = sqrt(mean(test_p$residuo[test_p$rural==1]^2)) , cabecera = sqrt(mean(test_p$residuo[test_p$rural==0]^2)))

## target predicho vs target original: la compresion hacia la media, y los puestos rurales (rojo) mas lejos de la diagonal
plot(test_p$voto_petro , test_p$voto_pred , pch=16 , cex=0.4 , col=ifelse(test_p$rural==1 , "red" , "blue") ,
     xlim=c(0,100) , ylim=c(0,100) , xlab="target original (%)" , ylab="target predicho (%)")
abline(a=0 , b=1 , lty=2)
legend("topleft" , legend=c("rural","cabecera") , col=c("red","blue") , pch=16)

## stop cluster
stopCluster(cl)
